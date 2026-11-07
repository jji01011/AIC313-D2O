import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 16)))
import torch.nn.functional as F
from torch.utils.data import DataLoader

EDM_TEACHER_CODE = os.environ.get("EDM_TEACHER_CODE", "/workspace/refs/edm_teacher")
sys.path.insert(0, EDM_TEACHER_CODE)
from teacher import phema_tools

from dataset import PokemonDataModule


def load_train(exclude):
    module = PokemonDataModule(batch_size=256, num_workers=8, return_category=True)
    xs, ys = [], []
    for x, y in DataLoader(module.train_dataset, batch_size=512, num_workers=8):
        xs.append(x); ys.append(y)
    xs, ys = torch.cat(xs), torch.cat(ys)
    if exclude:
        keep = torch.ones(len(xs), dtype=torch.bool)
        keep[torch.load(exclude)["heldout_indices"]] = False
        xs, ys = xs[keep], ys[keep]
    return xs, ys


def nan_rows(*ts):
    m = None
    for t in ts:
        r = ~torch.isfinite(t.flatten(1)).all(dim=1)
        m = r if m is None else (m | r)
    return m


class LoRAFake(torch.nn.Module):

    def __init__(self, base, rank, decoder_only=False, freeze_base=False):
        super().__init__()
        self._base = (base,)
        self.freeze_base = freeze_base
        self.active = False
        self.down, self.up = torch.nn.ParameterList(), torch.nn.ParameterList()
        for name, m in base.unet.named_modules():
            if decoder_only and not (name.startswith("dec.") or name == "out_conv"):
                continue
            if type(m).__name__ == "MPConv" and m.weight.ndim == 4:
                cout, cin, k, _ = m.weight.shape
                d = torch.nn.Parameter(torch.randn(rank, cin, k, k, device=m.weight.device) / (cin * k * k) ** 0.5)
                u = torch.nn.Parameter(torch.zeros(cout, rank, 1, 1, device=m.weight.device))
                self.down.append(d); self.up.append(u)
                m.register_forward_hook(self._branch(d, u))

    def _branch(self, d, u):
        def hook(module, inputs, output):
            if not self.active:
                return None
            x = inputs[0]
            return output + torch.nn.functional.conv2d(torch.nn.functional.conv2d(x, d.to(x.dtype), padding=d.shape[-1] // 2), u.to(x.dtype))
        return hook

    def forward(self, *args, **kwargs):
        base = self._base[0]
        if self.freeze_base:
            flags, training = [p.requires_grad for p in base.parameters()], base.training
            base.requires_grad_(False).eval()
        self.active = True
        try:
            return base(*args, **kwargs)
        finally:
            self.active = False
            if self.freeze_base:
                for p, f in zip(base.parameters(), flags):
                    p.requires_grad_(f)
                base.train(training)

    def post_step_(self):
        pass


def main(a):
    torch.manual_seed(a.seed)
    if a.vram_limit_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_limit_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()

    teacher, data = phema_tools.load_network(a.teacher, "cuda")
    teacher.eval().requires_grad_(False); teacher.set_attention("sdpa")
    guide = phema_tools.load_network(a.guide, "cuda")[0].eval().requires_grad_(False) if a.guide else None
    if guide is not None:
        guide.set_attention("sdpa")
    if a.fake_lora > 0 and a.fake_lora_base == "teacher":
        fake = LoRAFake(teacher, a.fake_lora, decoder_only=bool(a.fake_lora_dec))
        print(f"fake score: teacher + LoRA rank {a.fake_lora}, {sum(p.numel() for p in fake.parameters()):,} trainable parameters", flush=True)
    elif a.fake_lora == 0:
        fake = copy.deepcopy(teacher).train().requires_grad_(True); fake.mean_image.requires_grad_(False)
    if a.one_ckpt:
        sys.path.append(a.one_code)
        from d2o.config import GENERATOR_EDM2_KWARGS
        from d2o.edm2 import EDM2Precond
        ck = torch.load(a.one_ckpt, map_location="cpu", weights_only=False)
        one_meta = {k: v for k, v in ck.items() if k != "state_dict"}
        sigma_g_t = ck["state_dict"]["sigma_g"].clone()
        sigma_g = float(sigma_g_t)
        G = EDM2Precond(dict(GENERATOR_EDM2_KWARGS)).cuda()
        G.load_state_dict({k[len("backbone."):]: v.float() if v.is_floating_point() else v
                           for k, v in ck["state_dict"].items() if k.startswith("backbone.")})
        del ck
        for mod in G.modules():
            if getattr(mod, "dropout", None):
                mod.dropout = 0.0
        G.train().requires_grad_(True)
        if a.freeze_encoder:
            for name, q in G.named_parameters():
                q.requires_grad_(not name.startswith(("unet.emb_", "unet.enc.")))
            G.encoder_frozen = True
            G.train()
        net_cfg = None
    else:
        G = copy.deepcopy(teacher) if not a.g_net else phema_tools.load_network(a.g_net, "cuda")[0]
        G.set_attention("sdpa")
        sd = torch.load(a.init, map_location="cuda", weights_only=False)
        net_cfg = sd.get("network", data["network"])
        G.load_state_dict({k[len("backbone."):]: v.float() if v.is_floating_point() else v for k, v in sd["state_dict"].items()})
        G.set_dropout({int(r): 0.0 for r in net_cfg["kwargs"]["dropout_res"]})
        G.train().requires_grad_(True); G.mean_image.requires_grad_(False)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    if a.fake_lora > 0 and a.fake_lora_base == "generator":
        fake = LoRAFake(G, a.fake_lora, decoder_only=bool(a.fake_lora_dec), freeze_base=True)
        shared = {id(p) for p in G.parameters()} & {id(p) for p in fake.parameters()}
        print(f"fake score: generator + LoRA rank {a.fake_lora}, {sum(p.numel() for p in fake.parameters()):,} trainable parameters; "
              f"framework G + LoRA = {sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in fake.parameters()):,} "
              f"distinct tensors (shared ids {len(shared)})", flush=True)
    gp = [p for p in G.parameters() if p.requires_grad]; fp = [p for p in fake.parameters() if p.requires_grad]
    opt_G = torch.optim.Adam(gp, lr=a.glr, betas=(0.0, 0.999), eps=1e-6)
    opt_F = torch.optim.Adam(fp, lr=a.lr, betas=(0.0, 0.999), eps=1e-6)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    label_w = [float(v) for v in a.label_w.split(",")] if a.label_w else [0.0] * len(sigmas)
    log_s = opt_S = sig_ema = None
    if a.learn_sigmas > 0:
        log_s = torch.nn.Parameter(torch.tensor([float(s) for s in sigmas[1:]], device="cuda").log())
        opt_S = torch.optim.Adam([log_s], lr=a.learn_sigmas, betas=(0.9, 0.999))
        sig_ema = log_s.detach().clone()
    mean = G.mean_image.detach().float()
    sigma_d = float(teacher.sigma_data)
    print(f"G {sum(p.numel() for p in G.parameters()):,} (trainable {sum(p.numel() for p in gp):,}) | one-step {bool(a.one_ckpt)} | sigmas {sigmas} | grad_steps {a.grad_steps} "
          f"| setup memory {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)

    cur = 0
    state_path = run / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        G.load_state_dict(st["G"]); fake.load_state_dict(st["fake"]); G_ema.load_state_dict(st["G_ema"])
        opt_G.load_state_dict(st["opt_G"]); opt_F.load_state_dict(st["opt_F"]); cur = st["cur_nimg"]
        print(f"resumed at {cur}", flush=True)
        del st

    if a.lsg_gan > 0 or a.div_w > 0:
        if a.real_pt:
            d = torch.load(a.real_pt)
            images, labels = d["x_u8"].cuda().float() / 127.5 - 1, d["y"].cuda()
        else:
            images, labels = (t.cuda() for t in load_train(a.exclude))
        print(f"SiDA: {len(images)} real train-split images", flush=True)

    div_store = {}
    if a.div_w > 0 and a.div_space == "stats":
        for name in ("16x16_block2", "8x8_block2"):
            teacher.unet.enc[name].register_forward_hook(
                lambda m, i, o, n=name: div_store.__setitem__(n, o) if div_store.get("_on") else None)

    def div_feat(x, c, gen_noise=None):
        s = torch.full((len(x),), a.div_sigma, device="cuda")
        n = torch.randn_like(x) if gen_noise is None else gen_noise
        if a.div_space == "stats":
            div_store["_on"] = True
            teacher(x + a.div_sigma * n, s, teacher.one_hot(c), force_fp32=True)
            div_store["_on"] = False
            out = torch.cat([torch.cat([div_store[k].float().mean((2, 3)), div_store[k].float().std((2, 3))], 1)
                             for k in ("16x16_block2", "8x8_block2")], 1)
            div_store.pop("16x16_block2"); div_store.pop("8x8_block2")
            return out
        _, e = teacher(x + a.div_sigma * n, s, teacher.one_hot(c), force_fp32=True, return_enc=True)
        return e.float().flatten(1)

    d_real = None
    if a.div_w > 0:
        g_div = torch.Generator(device="cuda").manual_seed(1234)
        ds = []
        with torch.no_grad():
            for _ in range(a.div_ref_pairs // 64):
                c = labels[torch.randint(0, len(labels), (64,), device="cuda", generator=g_div)]
                xi, xj = [], []
                for cc in c.tolist():
                    pool = (labels == cc).nonzero().flatten()
                    k = pool[torch.randperm(len(pool), device="cuda", generator=g_div)[:2]]
                    xi.append(images[k[0]]); xj.append(images[k[1]])
                fi, fj = div_feat(torch.stack(xi), c), div_feat(torch.stack(xj), c)
                ds.append((fi - fj).abs().mean(1))
        d_real = torch.cat(ds).mean().item()
        print(f"diversity floor: d_real {d_real:.4f} over {len(torch.cat(ds))} same-class real pairs (sigma {a.div_sigma})", flush=True)
        div_diag = True

    def real_batch(n):
        idx = torch.randint(0, len(images), (n,), device="cuda")
        x, c = images[idx], labels[idx]
        flip = torch.rand(n, device="cuda") < 0.5
        return torch.where(flip[:, None, None, None], x.flip(3), x), c

    def gen(c, grad):
        if a.one_ckpt:
            n = len(c)
            with torch.set_grad_enabled(grad):
                return G(sigma_g * torch.randn(n, 3, 64, 64, device="cuda"), torch.full((n,), sigma_g, device="cuda"),
                         class_labels=F.one_hot(c, 151).float(), force_fp32=True).float()
        n = len(c); oh = G.one_hot(c); null = G.null_labels(n, "cuda")
        K = int(torch.randint(1, len(sigmas) + 1, (1,)).item()) if a.uniform_steps else len(sigmas)
        first_grad = K - a.grad_steps if grad else K
        x = mean + sigmas[0] * torch.randn(n, 3, 64, 64, device="cuda")
        for j, s in enumerate(sigmas[:K]):
            if j > 0 and log_s is not None:
                s = log_s[j - 1].exp() if grad else log_s[j - 1].detach().exp()
            st = s.expand(n) if torch.is_tensor(s) else torch.full((n,), s, device="cuda")
            if j > 0:
                x = x + s * torch.randn_like(x)
            with torch.set_grad_enabled(j >= first_grad):
                x = G(x, st, (1 + label_w[j]) * oh - label_w[j] * null).float()
        return x

    def fake_step(stats, use_gan):
        opt_F.zero_grad(set_to_none=True)
        for _ in range(B // a.fake_micro):
            if use_gan:
                real, c = real_batch(a.fake_micro)
            else:
                c = torch.randint(0, 151, (a.fake_micro,), device="cuda")
            with torch.no_grad():
                y = gen(c, grad=False)
            sig = (torch.randn(len(c), device="cuda") * a.P_std + a.P_mean).exp()
            w = ((sig ** 2 + sigma_d ** 2) / (sig * sigma_d) ** 2).view(-1, 1, 1, 1)
            n = torch.randn_like(y) * sig.view(-1, 1, 1, 1)
            y_fake, logvar, enc_f = fake(y + n, sig, teacher.one_hot(c), return_logvar=True, return_enc=True)
            y_fake = y_fake.float(); logvar = logvar.float()
            bad = nan_rows(y_fake, logvar)
            if bad.any():
                stats.setdefault("nan_F", []).append(bad.float().sum())
                continue
            loss_f = (w / logvar.exp()) * (y_fake - y) ** 2 + logvar
            loss = loss_f.sum()
            if use_gan:
                logit_f = enc_f.float().mean(dim=1, keepdim=True)
                _, _, enc_r = fake(real + n, sig, teacher.one_hot(c), return_logvar=True, return_enc=True)
                logit_r = enc_r.float().mean(dim=1, keepdim=True)
                bce = (F.binary_cross_entropy_with_logits(logit_r, torch.ones_like(logit_r), reduction="none").mean(dim=(1, 2, 3))
                       + F.binary_cross_entropy_with_logits(logit_f, torch.zeros_like(logit_f), reduction="none").mean(dim=(1, 2, 3))) / 2
                loss_d = w.flatten() * bce / logvar.exp().flatten()
                loss = loss + a.lsd * loss_d.sum()
                stats.setdefault("loss_D", []).append(loss_d.detach().mean())
            (loss / B).backward()
            stats.setdefault("loss_F", []).append(loss_f.detach().mean())
        for p in fp:
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
        opt_F.step(); fake.post_step_()

    B, s_min, s_max, rho = a.batch, 0.002, 80.0, 7.0
    next_log, next_snap = cur + a.log_kimg * 1000, (cur // (a.snap_kimg * 1000) + 1) * a.snap_kimg * 1000
    last_t, last_n, stats = time.time(), cur, {}
    while cur < a.total_kimg * 1000:
        train_G = cur >= a.fake_only_kimg * 1000
        use_gan = a.lsg_gan > 0 and cur >= (a.fake_only_kimg + a.sid_only_kimg) * 1000
        fake.requires_grad_(True)
        for _ in range(a.fake_steps):
            fake_step(stats, use_gan)
        if train_G:
            fake.requires_grad_(False); opt_G.zero_grad(set_to_none=True)
            if opt_S is not None:
                opt_S.zero_grad(set_to_none=True)
            for _ in range(B // a.micro_batch):
                c = torch.randint(0, 151, (a.micro_batch,), device="cuda")
                if a.div_w > 0:
                    h = a.micro_batch // 2
                    c = torch.cat([c[:h], c[:h]])
                oh = teacher.one_hot(c)
                y = gen(c, grad=True)
                u = torch.rand(len(c), device="cuda") * a.tmax / 1000
                sig = (s_max ** (1 / rho) + (1 - u) * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
                xt = y + torch.randn_like(y) * sig.view(-1, 1, 1, 1)
                y_real = teacher(xt, sig, oh).float()
                if a.cfg != 1:
                    with torch.no_grad():
                        u = (guide(xt, sig, oh) if guide is not None else teacher(xt, sig, teacher.null_labels(len(c), "cuda"))).float()
                    y_real = u + a.cfg * (y_real - u)
                y_fake, logvar, enc_f = fake(xt, sig, oh, return_logvar=True, return_enc=True)
                y_fake = y_fake.float(); logvar = logvar.float()
                bad = nan_rows(y, y_real, y_fake, logvar)
                if bad.any():
                    stats.setdefault("nan_G", []).append(bad.float().sum())
                    continue
                wf = (y - y_real).abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-5).detach()
                loss_sid = (y_real - y_fake) * ((y_real - y) - a.alpha * (y_real - y_fake)) / wf / logvar.exp()
                loss = a.lsg * loss_sid.sum()
                if use_gan:
                    logit = enc_f.float().mean(dim=1, keepdim=True).clamp(-10, 10)
                    gan = F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit), reduction="none").mean(dim=(1, 2, 3))
                    loss_gan = gan / wf.flatten() / logvar.exp().flatten()
                    loss = 0.5 * loss + 0.5 * a.lsg_gan * loss_gan.sum()
                    stats.setdefault("loss_Ggan", []).append(loss_gan.detach().mean())
                if a.div_w > 0:
                    f = div_feat(y, c)
                    d = (f[:h] - f[h:]).abs().mean(1).mean()
                    loss_div = F.relu(1 - d / d_real) ** 2
                    if div_diag:
                        g1 = torch.autograd.grad(loss / B, y, retain_graph=True)[0]
                        nan_frac = (~torch.isfinite(g1)).float().mean().item()
                        g1 = torch.nan_to_num(g1, nan=0.0, posinf=0.0, neginf=0.0).norm().item()
                        g2 = torch.autograd.grad(a.div_w * loss_div / B, y, retain_graph=True)[0].norm().item() if loss_div > 0 else 0.0
                        print(f"div diag: d/d_real {d.item() / d_real:.3f} | grad at y: main {g1:.4g} (non-finite {nan_frac:.2%}), div {g2:.4g} (ratio {g2 / max(g1, 1e-12):.3g})", flush=True)
                        div_diag = False
                    loss = loss + a.div_w * loss_div
                    stats.setdefault("div_ratio", []).append(d.detach() / d_real)
                    stats.setdefault("loss_div", []).append(loss_div.detach())
                (loss / B).backward()
                stats.setdefault("loss_Gsid", []).append(loss_sid.detach().flatten(1).sum(1).mean())
                stats.setdefault("gap", []).append((y_real - y_fake).detach().abs().mean())
            for p in gp:
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
            stats.setdefault("gnorm", []).append(torch.stack([p.grad.norm() for p in gp if p.grad is not None]).norm())
            opt_G.step()
            if opt_S is not None and log_s.grad is not None:
                torch.nan_to_num(log_s.grad, nan=0.0, posinf=0.0, neginf=0.0, out=log_s.grad)
                opt_S.step()
                with torch.no_grad():
                    sig_ema.lerp_(log_s.detach(), 1 - 0.5 ** (B / (a.ema_kimg * 1000)))
            if hasattr(G, "post_step_"):
                G.post_step_()
            with torch.no_grad():
                beta = 0.5 ** (B / (a.ema_kimg * 1000))
                for pe, p in zip(G_ema.parameters(), G.parameters()):
                    pe.copy_(p.detach().lerp(pe, beta))
        cur += B
        if cur >= next_log:
            now = time.time()
            rec = {k: torch.stack(v).float().mean().item() for k, v in stats.items()}
            rec.update(kimg=cur / 1000, phase="sid" if train_G else "fake", img_per_s=(cur - last_n) / (now - last_t),
                       max_mem_gb=torch.cuda.max_memory_allocated() / 2**30, hours=(now - t0) / 3600)
            if log_s is not None:
                rec.update(sigmas=[round(v, 4) for v in log_s.detach().exp().tolist()],
                           sigmas_ema=[round(v, 4) for v in sig_ema.exp().tolist()])
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            stats, last_t, last_n, next_log = {}, now, cur, next_log + a.log_kimg * 1000
        if cur >= next_snap or cur >= a.total_kimg * 1000:
            if a.one_ckpt:
                sd = {"sigma_g": sigma_g_t, **{"backbone." + k: v.detach().clone() for k, v in G_ema.state_dict().items()}}
                torch.save({**one_meta, "state_dict": sd, "sid_args": vars(a), "nimg": cur}, run / f"ema_{cur // 1000:06d}.ckpt")
            else:
                sd = {"backbone." + k: (v.half() if v.is_floating_point() and k != "mean_image" else v) for k, v in G_ema.state_dict().items()}
                extra = {"sigmas": [sigmas[0]] + sig_ema.exp().tolist()} if log_s is not None else {}
                torch.save({"state_dict": sd, "network": net_cfg, "nimg": cur, "ema_kimg": a.ema_kimg, **extra}, run / f"ema500_{cur // 1000:06d}kimg.ckpt")
            if a.save_state:
                torch.save({"G": G.state_dict(), "fake": fake.state_dict(), "G_ema": G_ema.state_dict(), "opt_G": opt_G.state_dict(),
                            "opt_F": opt_F.state_dict(), "cur_nimg": cur, "args": vars(a)}, state_path)
            print(f"snapshot {cur // 1000} kimg", flush=True)
            next_snap += a.snap_kimg * 1000


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="", help="ECD (or CMT) EMA snapshot (few-step generator)")
    p.add_argument("--one_ckpt", default="", help="one_nfe.ckpt of the D2O one-step generator (instead of --init)")
    p.add_argument("--one_code", default="/root/AIC313-Project-FastGen/cand_one_lhs", help="folder with the d2o package")
    p.add_argument("--freeze_encoder", type=int, default=1, help="one-step mode: train the decoder only, as D2O did")
    p.add_argument("--teacher", default=os.environ.get("ECT_INIT", "/workspace/teacher/handoff/teacher_unguided.pt"),
                   help="frozen true score; also the fake score's initialisation")
    p.add_argument("--g_net", default="", help="network whose architecture the generator takes (default: the teacher's)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--sigmas", default="80,4.0,1.5,0.5")
    p.add_argument("--label_w", default="", help="per-call label guidance of the generator (as the sampler), comma list")
    p.add_argument("--cfg", type=float, default=1.0, help="guidance weight of the real (teacher) score; 1 = unguided")
    p.add_argument("--guide", default="", help="autoguidance guide network (default: CFG with the null class)")
    p.add_argument("--grad_steps", type=int, default=4, help="number of final sampler calls that receive gradients")
    p.add_argument("--uniform_steps", type=int, default=0,
                   help="1 = Uniform-Step Matching: each generator call returns the output after a uniformly drawn number of steps")
    p.add_argument("--learn_sigmas", type=float, default=0.0,
                   help="Adam lr on log sigma_1..sigma_K of the sampler chain (0 = fixed --sigmas); snapshots store the EMA schedule")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--micro_batch", type=int, default=8, help="generator micro-batch (chain with gradients)")
    p.add_argument("--fake_micro", type=int, default=32)
    p.add_argument("--fake_lora", type=int, default=0, help="> 0: fake score = frozen teacher + LoRA of this rank (0 = full copy)")
    p.add_argument("--fake_lora_dec", type=int, default=0, help="1: LoRA on decoder convolutions only")
    p.add_argument("--fake_lora_base", default="teacher", choices=["teacher", "generator"],
                   help="network under the LoRA branches: the frozen teacher, or the generator itself (shared tensors, as D2O)")
    p.add_argument("--fake_steps", type=int, default=1, help="fake-score updates per generator update (two time-scale, as DMD2)")
    p.add_argument("--lr", type=float, default=1e-5, help="fake score")
    p.add_argument("--glr", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tmax", type=float, default=800)
    p.add_argument("--lsg", type=float, default=100.0)
    p.add_argument("--P_mean", type=float, default=-0.5613, help="teacher's sigma distribution (EDM2, sigma_data 0.635)")
    p.add_argument("--P_std", type=float, default=1.6)
    p.add_argument("--fake_only_kimg", type=int, default=20)
    p.add_argument("--sid_only_kimg", type=int, default=0, help="SiD-only generator updates before the GAN term starts")
    p.add_argument("--lsg_gan", type=float, default=0.0, help="SiDA generator GAN weight (sida_train.py: 0.01); 0 = plain SiD")
    p.add_argument("--lsd", type=float, default=1.0, help="SiDA discriminator loss weight")
    p.add_argument("--exclude", default="", help="held-out split (.pt with heldout_indices) removed from the real images")
    p.add_argument("--real_pt", default="", help="real images as a .pt {x_u8, y} (train split minus held-out) instead of the dataset")
    p.add_argument("--div_w", type=float, default=0.0, help="weight of the diversity floor (0 = off)")
    p.add_argument("--div_space", default="enc", choices=["enc", "stats"], help="diversity feature space (see the docstring)")
    p.add_argument("--div_sigma", type=float, default=0.5, help="noise level of the teacher features for the diversity floor")
    p.add_argument("--div_ref_pairs", type=int, default=2048, help="same-class real pairs for d_real")
    p.add_argument("--ema_kimg", type=float, default=20)
    p.add_argument("--total_kimg", type=int, default=1000)
    p.add_argument("--log_kimg", type=int, default=4)
    p.add_argument("--snap_kimg", type=int, default=25)
    p.add_argument("--save_state", type=int, default=1, help="0: no resumable state.pt (2.7 GB), for a nearly full disk")
    p.add_argument("--vram_limit_gb", type=float, default=18.0, help="allocator cap in GiB (18 GiB = 19.3e9 bytes, under the 20 GB rule with the CUDA context)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert bool(args.init) != bool(args.one_ckpt), "give exactly one of --init / --one_ckpt"
    main(args)
