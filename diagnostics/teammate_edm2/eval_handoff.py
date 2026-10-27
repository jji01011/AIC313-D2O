"""The teammate's hand-off teacher files under their sampling conventions and under ours: where does a difference come from?

README_MODELS.md of the hand-off (2026-10-12) reports the fully trained teacher at validation FID 20.50 with Heun 32
steps, the start point mean_image + 80 * noise and the output clamp(x) + 1/255, clamp. Our eval_teacher.py measured the
58.7M snapshot at 24.13 with 18 steps, start 80 * noise and a plain clamp. This script samples any of their network
files (same container as checkpoints/edm/ema-*.pt) with each convention switched on or off and prints both scores.
The official score is for reporting only: nothing may be chosen on it.

    PYTHONPATH=. python diagnostics/teammate_edm2/eval_handoff.py --teacher <file.pt> --steps 32 --mean_start 1 --offset 1 --workdir <folder>
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, "diagnostics/teammate_edm2")
from edm2_teacher import load_teacher  # noqa: E402

from d2o.fid import FIDMonitor  # noqa: E402
from d2o.sampling import edm_sampler  # noqa: E402
from evaluate import compute_fid  # noqa: E402


class Sampler(torch.nn.Module):
    def __init__(self, teacher, steps, mean_start, offset):
        super().__init__()
        self.teacher, self.steps, self.mean_start, self.offset = teacher, steps, mean_start, offset

    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        latents = torch.randn(shape, device=device)
        if self.mean_start:
            latents = latents + self.teacher.mean_image / 80.0  # the sampler multiplies by sigma_max = 80
        x = edm_sampler(lambda x, sigma, c: self.teacher(x.to(torch.float32), sigma, c), latents, category.to(device), num_steps=self.steps).clamp(-1, 1)
        return (x + 1 / 255).clamp(-1, 1) if self.offset else x


def main(args):
    teacher, _ = load_teacher(args.teacher, device=args.device)
    monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=args.device)
    torch.manual_seed(args.seed)
    train = monitor(Sampler(teacher, args.steps, bool(args.mean_start), bool(args.offset)), args.workdir)
    official = float(compute_fid(args.workdir, args.official_reference))
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} {args.teacher}: Heun {args.steps} steps, mean start {args.mean_start}, +1/255 {args.offset}, seed {args.seed}: "
          f"train reference {train:.3f}, official (validation reference, reporting only) {official:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A teacher file under the teammate's sampling conventions and ours")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--mean_start", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workdir", required=True, help="Folder for the samples; it is emptied first")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference", default="./results/reference_train")
    parser.add_argument("--official_reference", default="./results/reference")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
