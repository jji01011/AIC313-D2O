"""Sample the teammate's EDM2 teacher offline and store the images as uint8 shards.

Each shard holds 3,020 images (20 per category) from the deterministic Heun sampler. The latents of shard ``s`` are
``torch.randn`` from a CPU generator seeded with ``seed + s``, so (latent, category, image) pairs can be rebuilt later.
Existing shards are skipped, so the script can be stopped and started again at any time.

The GPU is shared with a running D2O training, which leaves about 4.5 GB free (the 20 GB vGPU keeps 2 GB to itself);
this script needs 1.7 GB. The scoring watcher needs all of what is left for a few minutes after every training piece,
and even an idle process here would be in its way. So give --until a time shortly before the running piece ends: no
shard is started that would not finish by then. As a last resort the script exits when free memory runs short.

    PYTHONPATH=. python diagnostics/teammate_edm2/generate_teacher_samples.py --shards 17 --until 09:53
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, "diagnostics/teammate_edm2")
from edm2_teacher import load_teacher  # noqa: E402

from d2o.sampling import edm_sampler  # noqa: E402

PER_CATEGORY, CATEGORIES = 20, 151


def seconds_until(clock):
    """Seconds from now to the next ``HH:MM`` (UTC)."""
    now = time.gmtime()
    hour, minute = (int(part) for part in clock.split(":"))
    return ((hour - now.tm_hour) * 60 + minute - now.tm_min) * 60 % 86400 - now.tm_sec


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    teacher, _ = load_teacher(args.teacher, device=args.device)
    categories = torch.arange(CATEGORIES).repeat_interleave(PER_CATEGORY)
    deadline = time.time() + seconds_until(args.until) if args.until else float("inf")
    shard_seconds = args.shard_seconds
    for shard in range(args.shards):
        path = os.path.join(args.outdir, f"shard-{shard:04d}.pt")
        if os.path.exists(path):
            continue
        if time.time() + shard_seconds > deadline:
            print(f"{time.strftime('%H:%M:%S')} stopping before shard {shard}: it would not finish by {args.until} UTC", flush=True)
            break
        start = time.time()
        latents = torch.randn([len(categories), 3, 64, 64], generator=torch.Generator().manual_seed(args.seed + shard))
        if args.mean_start:
            # The teacher learnt x - mean_image: its prior is mean_image + 80 * noise (the sampler multiplies by 80).
            # Without it the samples come out dark (2026-10-13, results/teacher_handoff_eval.log).
            latents = latents + teacher.mean_image.cpu() / 80.0
        images = []
        for i in range(0, len(categories), args.batch):
            if torch.cuda.mem_get_info(args.device)[0] / 2 ** 30 < args.low_gb:
                raise SystemExit(f"{time.strftime('%H:%M:%S')} less than {args.low_gb} GB of GPU memory free: exiting, shard {shard} is not written")
            with torch.no_grad():
                x = edm_sampler(lambda x, sigma, category: teacher(x.to(torch.float32), sigma, category),
                                latents[i:i + args.batch].to(args.device), categories[i:i + args.batch].to(args.device), num_steps=args.steps)
            images.append(((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())
        tmp = path + ".tmp"
        torch.save(dict(images=torch.cat(images), labels=categories.clone(), seed=args.seed + shard, steps=args.steps, teacher=args.teacher,
                        mean_start=bool(args.mean_start)), tmp)
        os.replace(tmp, path)
        shard_seconds = time.time() - start
        print(f"{time.strftime('%H:%M:%S')} {path}: {len(categories)} images in {time.time() - start:.0f} s, "
              f"peak {torch.cuda.max_memory_allocated(args.device) / 2 ** 30:.2f} GB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline samples of the teammate's EDM2 teacher")
    parser.add_argument("--teacher", default="checkpoints/edm/ema-0016777216-0.100.pt")
    parser.add_argument("--outdir", default="data/cache/teacher_edm2")
    parser.add_argument("--shards", type=int, default=17, help="Number of shards of 3,020 images to have in --outdir")
    parser.add_argument("--steps", type=int, default=18, help="Heun steps (2 * steps - 1 network evaluations)")
    parser.add_argument("--mean_start", type=int, default=0, help="1 = start from mean_image + 80 * noise, the prior the teacher was trained with")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--until", default=None, help="HH:MM (UTC): do not start a shard that would not finish by then")
    parser.add_argument("--shard_seconds", type=float, default=400, help="First guess of the time one shard takes")
    parser.add_argument("--low_gb", type=float, default=0.7, help="Exit when less GPU memory than this is free before a batch")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
