"""Stage 0: class-conditional EDM pre-training on the Pokemon train split, from scratch.

D2O treats diffusion training as generative pre-training. The official code
starts from the public NVlabs EDM checkpoints, which are not allowed here, so
this script trains the EDM that ``train_d2o.py`` later fine-tunes into a
one-step generator. The loss and EMA follow NVlabs/edm (``EDMLoss``).
"""

import argparse
import copy
import os
import time

import torch

from d2o.config import GENERATOR_KWARGS
from d2o.data import InfiniteBatches, load_train_tensors
from d2o.networks import EDMPrecond
from d2o.sampling import edm_sampler, one_hot, save_image_grid
from d2o.training_utils import JsonlLogger, RunningMean, StopSignal, sanitize_gradients, update_ema


def edm_loss(net, images, class_labels, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    noise = torch.randn_like(images) * sigma
    denoised = net(images + noise, sigma, class_labels)
    return (weight * (denoised - images) ** 2).mean()


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.outdir, exist_ok=True)

    images, labels = load_train_tensors(args.data_root, args.split_dir)
    batches = InfiniteBatches(images, labels, args.batch, device, xflip=args.xflip, seed=args.seed)
    print(f"Train split: {len(images)} images, {int(labels.max()) + 1} classes")

    net_kwargs = dict(GENERATOR_KWARGS, model_channels=args.model_channels, num_blocks=args.num_blocks, arch=args.arch)
    net = EDMPrecond(**dict(net_kwargs, dropout=args.dropout), use_fp16=args.fp16).to(device).train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    print(f"EDM parameters: {sum(p.numel() for p in net.parameters()):,}")

    cur_nimg = 0
    state_path = os.path.join(args.outdir, "edm-state-latest.pt")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        cur_nimg = state["cur_nimg"]
        print(f"Resumed from {args.resume} at {cur_nimg / 1e3:.0f} kimg")

    logger = JsonlLogger(os.path.join(args.outdir, "log.jsonl"))
    stats = RunningMean()
    grid_generator = torch.Generator(device=device).manual_seed(0)
    grid_latents = torch.randn([64, 3, 64, 64], device=device, generator=grid_generator)
    grid_labels = one_hot(torch.arange(64, device=device) % net.label_dim)

    total_nimg = int(args.kimg * 1000)
    next_log = (cur_nimg // (args.log_kimg * 1000) + 1) * args.log_kimg * 1000
    next_snap = (cur_nimg // (args.snap_kimg * 1000) + 1) * args.snap_kimg * 1000
    tick_time, tick_nimg = time.time(), cur_nimg
    stop = StopSignal()
    while cur_nimg < total_nimg and not stop.requested:
        real, real_labels = next(batches)
        loss = edm_loss(net, real, one_hot(real_labels, net.label_dim))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(cur_nimg / max(args.lr_rampup_kimg * 1000, 1e-8), 1)
        sanitize_gradients(net.parameters())
        optimizer.step()
        cur_nimg += args.batch
        update_ema(ema, net, cur_nimg, args.batch, args.ema_halflife_kimg, args.ema_rampup_ratio)
        stats.add(loss=loss.detach())

        done = cur_nimg >= total_nimg or stop.requested
        if cur_nimg >= next_log or done:
            now = time.time()
            fields = dict(
                kimg=round(cur_nimg / 1e3, 1),
                **{key: round(value, 5) for key, value in stats.pop().items()},
                img_per_sec=round((cur_nimg - tick_nimg) / (now - tick_time), 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2),
            )
            logger.log(**fields)
            print(" ".join(f"{key} {value}" for key, value in fields.items()), flush=True)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            tick_time, tick_nimg = now, cur_nimg
            next_log += args.log_kimg * 1000

        if cur_nimg >= next_snap or done:
            kimg = cur_nimg // 1000
            torch.save(dict(ema=ema.state_dict(), net_kwargs=net_kwargs, cur_nimg=cur_nimg), os.path.join(args.outdir, f"edm-{kimg:06d}.pt"))
            torch.save(
                dict(net=net.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(), net_kwargs=net_kwargs, cur_nimg=cur_nimg, args=vars(args)),
                state_path,
            )
            save_image_grid(edm_sampler(ema, grid_latents, grid_labels), os.path.join(args.outdir, f"samples-{kimg:06d}.png"))
            next_snap += args.snap_kimg * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDM pre-training (D2O stage 0)")
    parser.add_argument("--outdir", default="./runs/edm")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--resume", default=None, help="edm-state-latest.pt of a previous run")
    parser.add_argument("--arch", default=GENERATOR_KWARGS["arch"], choices=("ddpmpp", "ncsnpp"))
    parser.add_argument("--model_channels", type=int, default=GENERATOR_KWARGS["model_channels"])
    parser.add_argument("--num_blocks", type=int, default=GENERATOR_KWARGS["num_blocks"])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--kimg", type=float, default=5000, help="Training duration in thousands of images")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_rampup_kimg", type=float, default=50)
    parser.add_argument("--ema_halflife_kimg", type=float, default=500)
    parser.add_argument("--ema_rampup_ratio", type=float, default=0.05)
    parser.add_argument("--xflip", type=int, default=1, help="Random horizontal flips of the training images")
    parser.add_argument("--fp16", type=int, default=1, help="Run the U-Net in bfloat16, as the official D2O code does")
    parser.add_argument("--log_kimg", type=float, default=10)
    parser.add_argument("--snap_kimg", type=float, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
