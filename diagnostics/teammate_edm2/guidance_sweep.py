"""Does classifier-free guidance on part of the noise range lower the teacher's FID? (train-split reference only)

The teammate's EDM2 teacher has a null class, so a guided denoiser costs no parameters. Guidance over the whole range
made it worse (eval_teacher.py: 1.3 gave 12.03 against 11.56). Kynkäänniemi et al., "Applying Guidance in a Limited
Interval Improves Sample and Distribution Quality in Diffusion Models" (NeurIPS 2024), report that guidance helps the
FID only when it is applied in the middle of the noise range. If that holds here, the same guided denoiser can serve as
the real score of the distribution-matching term. Measured like eval_teacher.py: 3,020 images, 20 per category, Heun.

    PYTHONPATH=. python diagnostics/teammate_edm2/guidance_sweep.py --workdir <an empty scratch directory>
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, "diagnostics/teammate_edm2")
from edm2_teacher import load_teacher  # noqa: E402

from d2o.fid import FIDMonitor  # noqa: E402
from d2o.sampling import edm_sampler  # noqa: E402

# (guidance, lowest sigma, highest sigma): guidance is applied for lowest < sigma <= highest. The 18 Heun steps visit
# sigma = 80, 58, 41, 28, 19, 13, 8.4, 5.3, 3.3, 1.9, 1.1, 0.59, 0.30, 0.14, 0.06, 0.02, 0.007, 0.002.
CONFIGS = [
    (1.0, 0.0, 100.0),  # no guidance: has to reproduce the 11.283 of eval_teacher.py
    (1.1, 0.0, 100.0),
    (1.3, 0.25, 4.0),
    (1.6, 0.25, 4.0),
    (1.3, 1.0, 10.0),
    (1.3, 0.1, 1.0),
    (1.3, 4.0, 100.0),
]


class IntervalGuidedSampler(torch.nn.Module):
    """The teacher with the ``sample()`` interface of ``evaluate.generate_samples``, guided inside a sigma interval."""

    def __init__(self, teacher, steps, guidance, lo, hi):
        super().__init__()
        self.teacher, self.steps, self.guidance, self.lo, self.hi = teacher, steps, guidance, lo, hi
        self.evaluations = 0

    def denoise(self, x, sigma, category):
        cond = self.teacher(x.to(torch.float32), sigma, category)
        self.evaluations += 1
        if self.guidance == 1 or not self.lo < float(sigma[0]) <= self.hi:
            return cond
        null = self.teacher(x.to(torch.float32), sigma, None)  # the null class
        self.evaluations += 1
        return null + self.guidance * (cond - null)

    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        latents = torch.randn(shape, device=device)
        self.evaluations = 0
        return edm_sampler(self.denoise, latents, category.to(device), num_steps=self.steps).clamp(-1, 1)


def main(args):
    teacher, _ = load_teacher(args.teacher, device=args.device)
    monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=args.device)
    for guidance, lo, hi in CONFIGS:
        model = IntervalGuidedSampler(teacher, args.steps, guidance, lo, hi)
        torch.manual_seed(0)
        score = monitor(model, args.workdir)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"{stamp} {args.teacher}: {model.evaluations} NFE Heun ({args.steps} steps), guidance {guidance:g} for "
              f"{lo:g} < sigma <= {hi:g}, FID vs train reference {score:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher FID against the train reference with guidance on a sigma interval")
    parser.add_argument("--teacher", default="checkpoints/edm/ema-0020971520-0.100.pt")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference", default="./results/reference_train")
    parser.add_argument("--workdir", required=True, help="Scratch directory for the samples; it is emptied before every measurement")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
