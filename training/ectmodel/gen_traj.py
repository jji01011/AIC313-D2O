import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EDM_TEACHER_CODE = os.environ.get("EDM_TEACHER_CODE", "/workspace/refs/edm_teacher")
sys.path.insert(0, EDM_TEACHER_CODE)
from teacher import phema_tools

from dataset import PokemonDataModule


def train_labels():
    module = PokemonDataModule(batch_size=256, num_workers=8, return_category=True)
    return torch.cat([y for _, y in DataLoader(module.train_dataset, batch_size=512, num_workers=8)])


@torch.no_grad()
def heun(net, x, onehot, t_steps, keep):
    states = [x.half().cpu()]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        d = (x - net(x, t_cur.expand(len(x)), onehot).float()) / t_cur
        x_next = x + (t_next - t_cur) * d
        if i < len(t_steps) - 2:
            d2 = (x_next - net(x_next, t_next.expand(len(x)), onehot).float()) / t_next
            x_next = x + (t_next - t_cur) * (0.5 * d + 0.5 * d2)
        x = x_next
        if i + 1 in keep:
            states.append((x.clamp(-1, 1) if i + 1 == len(t_steps) - 1 else x).half().cpu())
    return torch.stack(states, dim=1)


def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    net, _ = phema_tools.load_network(a.net, "cuda")
    net.eval().requires_grad_(False); net.set_attention("sdpa")
    mean = net.mean_image.detach().float()
    labels_all = train_labels()
    i = torch.arange(a.steps, dtype=torch.float64, device="cuda")
    t_steps = (a.sigma_max ** (1 / a.rho) + i / (a.steps - 1) * (a.sigma_min ** (1 / a.rho) - a.sigma_max ** (1 / a.rho))) ** a.rho
    t_steps = torch.cat([t_steps, torch.zeros(1, dtype=torch.float64, device="cuda")]).float()
    keep = set(range(a.every, a.steps + 1, a.every))
    sigmas = torch.stack([t_steps[0]] + [t_steps[k] for k in sorted(keep)]).cpu()
    print("stored sigmas", [round(float(s), 4) for s in sigmas], flush=True)
    for s in range(a.first_shard, a.first_shard + a.shards):
        path = out / f"traj_{s:03d}.pt"
        if path.exists():
            continue
        t0 = time.time()
        gs = torch.Generator(device="cuda").manual_seed(a.seed * 100003 + s)
        g = torch.Generator().manual_seed(a.seed * 100003 + s)
        labels = labels_all[torch.randint(0, len(labels_all), (a.shard,), generator=g)]
        chunks = []
        for b in range(0, a.shard, a.batch):
            c = labels[b:b + a.batch].cuda()
            z = torch.randn(len(c), 3, 64, 64, device="cuda", generator=gs)
            chunks.append(heun(net, mean + t_steps[0] * z, net.one_hot(c), t_steps, keep))
        torch.save({"states": torch.cat(chunks), "sigmas": sigmas, "labels": labels}, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)
        print(f"shard {s} {a.shard / (time.time() - t0):.1f} traj/s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--net", default=os.environ.get("ECT_INIT", "/workspace/teacher/handoff/teacher_unguided.pt"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--shards", type=int, default=16)
    p.add_argument("--first_shard", type=int, default=0)
    p.add_argument("--shard", type=int, default=4096)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--every", type=int, default=2, help="store every k-th grid state")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
