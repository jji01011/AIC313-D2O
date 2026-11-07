import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

EDM_TEACHER_CODE = os.environ.get("EDM_TEACHER_CODE", "/workspace/refs/edm_teacher")
sys.path.insert(0, EDM_TEACHER_CODE)
from teacher import phema_tools


def load_trajectories(traj_dir):
    files = sorted(glob.glob(os.path.join(traj_dir, "traj_*.pt")))
    first = torch.load(files[0], weights_only=False)
    n = sum(torch.load(f, weights_only=False, mmap=True)["states"].shape[0] for f in files)
    states = torch.empty((n,) + tuple(first["states"].shape[1:]), dtype=torch.float16)
    labels = torch.empty(n, dtype=torch.long)
    k = 0
    for f in files:
        d = torch.load(f, weights_only=False, mmap=True)
        m = d["states"].shape[0]
        states[k:k + m] = d["states"]; labels[k:k + m] = d["labels"]; k += m
    return states, labels, first["sigmas"].float()


def main(a):
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.vram_limit_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram_limit_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    run = Path(a.run_dir); run.mkdir(parents=True, exist_ok=True)
    log = open(run / "log.jsonl", "a")
    t0 = time.time()
    states, labels, sigmas = load_trajectories(a.traj_dir)
    n_sig = len(sigmas) - 1
    print(f"trajectories {len(states)} | sigmas {[round(float(s), 3) for s in sigmas]}", flush=True)

    net, data = phema_tools.load_network(a.init, "cuda")
    net_cfg = data["network"]
    if a.dropout >= 0:
        drop = {int(res): (a.dropout if int(res) <= a.dropres else 0.0) for res in net_cfg["kwargs"]["dropout_res"]}
        net.set_dropout(drop)
        net_cfg = copy.deepcopy(net_cfg); net_cfg["kwargs"]["dropout_res"] = {str(k): v for k, v in drop.items()}
    print("dropout", net_cfg["kwargs"]["dropout_res"], flush=True)
    net.train().requires_grad_(True); net.set_attention("sdpa")
    net.mean_image.requires_grad_(False)
    emas = {h: copy.deepcopy(net).eval().requires_grad_(False) for h in a.ema_kimgs}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-8)

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
    sig_gpu = sigmas.cuda()
    next_log, next_snap = cur + a.log_kimg * 1000, (cur // (a.snap_kimg * 1000) + 1) * a.snap_kimg * 1000
    last_t, last_n, stats = time.time(), cur, {}
    while cur < a.total_kimg * 1000:
        lr = a.lr * min(1.0, (cur + B) / max(a.warmup_kimg * 1000, 1)) * max(0.0, 1 - cur / (a.total_kimg * 1000)) ** a.decay_pow
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(B // MB):
            n = torch.randint(0, len(states), (MB,))
            i = torch.randint(0, n_sig, (MB,))
            x = states[n, i].cuda(non_blocking=True).float()
            y = states[n, -1].cuda(non_blocking=True).float()
            flip = (torch.rand(MB, device="cuda") < 0.5).view(-1, 1, 1, 1)
            x, y = torch.where(flip, x.flip(3), x), torch.where(flip, y.flip(3), y)
            t = sig_gpu[i.cuda()]
            D = net(x, t, net.one_hot(labels[n].cuda())).float()
            d2 = (D - y).square().flatten(1).sum(1)
            loss = torch.sqrt(d2 + a.c ** 2) - a.c
            (loss.sum() / B).backward()
            stats.setdefault("loss", []).append(loss.detach().mean())
            stats.setdefault("loss_t80", []).append((loss * (i == 0).cuda()).sum().detach() / max(int((i == 0).sum()), 1))
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
            rec = {k: torch.stack(v).float().mean().item() for k, v in stats.items()}
            rec.update(kimg=cur / 1000, lr=lr, img_per_s=(cur - last_n) / (now - last_t),
                       max_mem_gb=torch.cuda.max_memory_allocated() / 2**30, hours=(now - t0) / 3600)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            stats, last_t, last_n, next_log = {}, now, cur, next_log + a.log_kimg * 1000
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
    p.add_argument("--traj_dir", default="/workspace/traj_e112")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--micro_batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=7e-4, help="CMT: 7e-4, decayed linearly to 0")
    p.add_argument("--decay_pow", type=float, default=1.0)
    p.add_argument("--warmup_kimg", type=int, default=50)
    p.add_argument("--total_kimg", type=int, default=3200)
    p.add_argument("--c", type=float, default=0.06)
    p.add_argument("--dropout", type=float, default=-1, help="< 0 keeps the teacher's dropout table")
    p.add_argument("--dropres", type=int, default=16)
    p.add_argument("--ema_kimgs", type=float, nargs="+", default=[200, 500])
    p.add_argument("--log_kimg", type=int, default=20)
    p.add_argument("--snap_kimg", type=int, default=400)
    p.add_argument("--vram_limit_gb", type=float, default=18.0, help="allocator cap in GiB (18 GiB = 19.3e9 bytes, under the 20 GB rule with the CUDA context)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
