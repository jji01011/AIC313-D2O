import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EDM_TEACHER_CODE = os.environ.get("EDM_TEACHER_CODE", "/workspace/refs/edm_teacher")
sys.path.insert(0, EDM_TEACHER_CODE)
from teacher import official, phema_tools

from dataset import PokemonDataModule


def load_train(exclude=""):
    module = PokemonDataModule(batch_size=256, num_workers=8, return_category=True)
    xs, ys = [], []
    for x, y in DataLoader(module.train_dataset, batch_size=512, num_workers=8):
        xs.append(x); ys.append(y)
    x, y = torch.cat(xs), torch.cat(ys)
    if exclude:
        drop = torch.zeros(len(x), dtype=torch.bool)
        drop[torch.load(exclude)["heldout_indices"]] = True
        x, y = x[~drop], y[~drop]
    return x, y


def main(a):
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.vram_limit_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_limit_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    images, labels = load_train(a.exclude)
    images, labels = images.cuda(), labels.cuda()
    print(f"train images {len(images)}", flush=True)

    net, data = phema_tools.load_network(a.init, "cuda")
    net_cfg = data["network"]
    teacher = None
    if a.distill:
        teacher = copy.deepcopy(net).eval().requires_grad_(False)
        if a.attention != "einsum":
            teacher.set_attention(a.attention)
    if a.init_ckpt:
        sd = torch.load(a.init_ckpt, map_location="cuda", weights_only=False)["state_dict"]
        net.load_state_dict({k[len("backbone."):]: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        print(f"student initialised from {a.init_ckpt}", flush=True)
    if a.dropout >= 0:
        drop = {int(res): (a.dropout if int(res) <= a.dropres else 0.0) for res in net_cfg["kwargs"]["dropout_res"]}
        net.set_dropout(drop)
        net_cfg = copy.deepcopy(net_cfg); net_cfg["kwargs"]["dropout_res"] = {str(k): v for k, v in drop.items()}
        print("dropout", drop, flush=True)
    sigma_d = float(net.sigma_data)
    net.train().requires_grad_(True)
    if a.attention != "einsum":
        net.set_attention(a.attention)
    net.mean_image.requires_grad_(False)
    emas = {h: copy.deepcopy(net).eval().requires_grad_(False) for h in a.ema_kimgs}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-8)
    augment = official.AugmentPipe(p=a.aug_p, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1) if a.aug_p > 0 else None
    print(f"params {sum(p.numel() for p in net.parameters()):,} (trainable {sum(p.numel() for p in params):,})", flush=True)

    cur = 0
    state_path = run / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location="cuda", weights_only=False)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); cur = st["cur_nimg"]
        for h, e in emas.items():
            e.load_state_dict(st["emas"][h])
        print(f"resumed at {cur} images", flush=True)
        del st

    B, MB = a.batch, a.micro_batch
    next_log, next_snap = cur + a.log_kimg * 1000, (cur // (a.snap_kimg * 1000) + 1) * a.snap_kimg * 1000
    last_t, last_n, stats = time.time(), cur, []
    while cur < a.total_kimg * 1000:
        stage = a.start_stage + cur // (a.stage_kimg * 1000)
        decay = 1.0 / a.q ** (stage + 1)
        lr = a.lr / math.sqrt(max(cur / (a.lr_ref_steps * B), 1.0)) if a.lr_ref_steps > 0 else a.lr
        if a.warmup_kimg > 0:
            lr *= min(1.0, (cur + B) / (a.warmup_kimg * 1000))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for k in range(B // MB):
            idx = torch.randint(0, len(images), (MB,), device="cuda")
            y, c = images[idx], labels[idx]
            if augment is not None:
                y, aug_labels = augment(y)
            else:
                aug_labels = None
            c = torch.where(torch.rand(MB, device="cuda") < a.label_dropout, torch.full_like(c, net.num_classes), c)
            onehot = net.one_hot(c)
            t = (torch.randn(MB, device="cuda") * a.P_std + a.P_mean).exp()
            r = (t * (1 - decay * (1 + a.k * torch.sigmoid(-a.b * t)))).clamp(min=0)
            e = torch.randn_like(y)
            tt, rr = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)
            x_t = y + tt * e
            with torch.no_grad():
                if teacher is not None:
                    D_T = teacher(x_t, t, onehot, augment_labels=aug_labels).float()
                    x_r, base = x_t + (rr - tt) * (x_t - D_T) / tt, D_T
                else:
                    x_r, base = y + rr * e, y
            rng = torch.cuda.get_rng_state()
            D_t = net(x_t, t, onehot, augment_labels=aug_labels).float()
            with torch.no_grad():
                torch.cuda.set_rng_state(rng)
                D_r = net(x_r, r.clamp(min=1e-8), onehot, augment_labels=aug_labels).float()
                D_r = torch.nan_to_num(D_r)
                D_r = torch.where((r > 0).view(-1, 1, 1, 1), D_r, base)
            d2 = (D_t - D_r).square().flatten(1).sum(1)
            w = (t ** 2 + sigma_d ** 2) / (t * sigma_d) ** 2 if a.weighting == "snrpk" else 1.0 / (t - r)
            loss = (torch.sqrt(d2 + a.c ** 2) - a.c) * w
            (loss.sum() / B).backward()
            stats.append(loss.detach().mean())
        for p in params:
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
        opt.step()
        cur += B
        with torch.no_grad():
            for h, ema in emas.items():
                beta = 0.5 ** (B / (h * 1000))
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.copy_(p.detach().lerp(pe, beta))
        net.post_step_()
        if cur >= next_log:
            now = time.time()
            rec = dict(kimg=cur / 1000, stage=int(stage), decay=decay, lr=lr, loss=torch.stack(stats).mean().item(),
                       img_per_s=(cur - last_n) / (now - last_t), max_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                       hours=(now - t0) / 3600)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            stats, last_t, last_n, next_log = [], now, cur, next_log + a.log_kimg * 1000
        if cur >= next_snap or cur >= a.total_kimg * 1000:
            for h, ema in emas.items():
                sd = {"backbone." + k: (v.half() if v.is_floating_point() and k != "mean_image" else v) for k, v in ema.state_dict().items()}
                torch.save({"state_dict": sd, "network": net_cfg, "nimg": cur, "ema_kimg": h},
                           run / f"ema{int(h)}_{cur // 1000:06d}kimg.ckpt")
            torch.save({"net": net.state_dict(), "emas": {h: e.state_dict() for h, e in emas.items()}, "opt": opt.state_dict(),
                        "cur_nimg": cur, "args": vars(a)}, state_path)
            print(f"snapshot {cur // 1000} kimg", flush=True)
            next_snap += a.snap_kimg * 1000


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=os.environ.get("ECT_INIT", "/workspace/teacher/handoff/teacher_unguided.pt"))
    p.add_argument("--init_ckpt", default="", help="snapshot to initialise the student from (e.g. cmt_train.py output)")
    p.add_argument("--start_stage", type=int, default=0, help="first ECT stage (decay q^-(stage+1)); 3 = 1/256")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--micro_batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_ref_steps", type=int, default=2000, help="lr / sqrt(max(step / ref, 1)); 0 = constant")
    p.add_argument("--warmup_kimg", type=int, default=0)
    p.add_argument("--total_kimg", type=int, default=12800)
    p.add_argument("--stage_kimg", type=int, default=3200, help="shrink t - r by q every this many kimg (ECT 'double')")
    p.add_argument("--q", type=float, default=4.0)
    p.add_argument("--k", type=float, default=8.0)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--c", type=float, default=0.06, help="pseudo-Huber constant (ECT: 0.06 for 64x64)")
    p.add_argument("--P_mean", type=float, default=-0.8)
    p.add_argument("--P_std", type=float, default=1.6)
    p.add_argument("--weighting", default="snrpk", choices=("snrpk", "inv_dt"))
    p.add_argument("--dropout", type=float, default=0.4, help="< 0 keeps the teacher's dropout table")
    p.add_argument("--dropres", type=int, default=16)
    p.add_argument("--distill", type=int, default=0, help="1 = ECD with the frozen initial network as teacher")
    p.add_argument("--aug_p", type=float, default=0.15)
    p.add_argument("--label_dropout", type=float, default=0.0)
    p.add_argument("--ema_kimgs", type=float, nargs="+", default=[200, 500, 1500])
    p.add_argument("--log_kimg", type=int, default=20)
    p.add_argument("--snap_kimg", type=int, default=400)
    p.add_argument("--vram_limit_gb", type=float, default=18.0, help="allocator cap in GiB (18 GiB = 19.3e9 bytes, under the 20 GB rule with the CUDA context)")
    p.add_argument("--attention", default="sdpa", choices=("einsum", "sdpa"))
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
