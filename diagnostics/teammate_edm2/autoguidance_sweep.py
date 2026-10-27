"""Does guiding the teacher with a weaker denoiser lower its FID? (train-split reference only)

Karras et al., "Guiding a Diffusion Model with a Bad Version of Itself" (NeurIPS 2024): D = D_weak + w (D_main - D_weak),
where the weak denoiser is a smaller or less trained model of the same data and the same conditioning. Unlike guidance
with the null class (guidance_sweep.py: 11.283 without, 11.075 at best) it does not trade diversity for class fit.
If it lowers the teacher's FID clearly, the guided denoiser can be the real score of the distribution-matching term.

Weak denoisers, all trained from scratch on the train split:
  edm:<path>    our own stage-0 EDM (train_edm.py snapshot)
  edm2:<path>   another snapshot of the teammate's EDM2
  noattn        the teacher itself with its self-attention switched off (no weights besides the teacher's)
  null          the teacher with its null class (classifier-free guidance, as in guidance_sweep.py)
A configuration is weak,w,lowest,highest: guidance acts for lowest < sigma <= highest. Measured like eval_teacher.py:
3,020 images, 20 per category, Heun, seed 0.

    PYTHONPATH=. python diagnostics/teammate_edm2/autoguidance_sweep.py --workdir <an empty scratch directory> \
        --configs edm:runs/edm_c96/edm-005000.pt,2,0,100 noattn,1.5,0.25,4
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, "diagnostics/teammate_edm2")
from edm2_teacher import load_teacher  # noqa: E402

from d2o.fid import FIDMonitor  # noqa: E402
from d2o.networks import EDMPrecond  # noqa: E402
from d2o.sampling import edm_sampler, one_hot  # noqa: E402


class WeakDenoiser:
    """``D_weak(x, sigma, category)`` on images in [-1, 1], built from one of the specifications above."""

    def __init__(self, spec, teacher, device):
        self.kind, _, path = spec.partition(":")
        self.teacher = teacher
        if self.kind == "edm":
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.net = EDMPrecond(**checkpoint["net_kwargs"]).to(device).eval().requires_grad_(False)
            self.net.load_state_dict(checkpoint["ema"])
        elif self.kind == "edm2":
            self.net, _ = load_teacher(path, device=device)
        else:
            assert self.kind in ("noattn", "null"), spec
            self.blocks = [block for block in teacher.net.unet.modules() if getattr(block, "num_heads", 0)]

    def __call__(self, x, sigma, category):
        if self.kind == "edm":
            return self.net(x, sigma, one_hot(category, self.net.label_dim), force_fp32=True)
        if self.kind == "edm2":
            return self.net(x, sigma, category)
        if self.kind == "null":
            return self.teacher(x, sigma, None)
        heads = [block.num_heads for block in self.blocks]
        for block in self.blocks:
            block.num_heads = 0  # Block.forward of the public EDM2 code skips the attention branch
        try:
            return self.teacher(x, sigma, category)
        finally:
            for block, count in zip(self.blocks, heads):
                block.num_heads = count


class GuidedSampler(torch.nn.Module):
    """The teacher with the ``sample()`` interface of ``evaluate.generate_samples``, guided inside a sigma interval."""

    def __init__(self, teacher, weak, steps, guidance, lo, hi):
        super().__init__()
        self.teacher, self.weak, self.steps, self.guidance, self.lo, self.hi = teacher, weak, steps, guidance, lo, hi

    def denoise(self, x, sigma, category):
        x = x.to(torch.float32)
        main = self.teacher(x, sigma, category)
        if self.guidance == 1 or not self.lo < float(sigma[0]) <= self.hi:
            return main
        weak = self.weak(x, sigma.to(torch.float32), category).to(torch.float32)
        return weak + self.guidance * (main - weak)

    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        latents = torch.randn(shape, device=device)
        return edm_sampler(self.denoise, latents, category.to(device), num_steps=self.steps).clamp(-1, 1)


def main(args):
    teacher, _ = load_teacher(args.teacher, device=args.device)
    monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=args.device)
    weak_denoisers = {}
    for config in args.configs:
        spec, guidance, lo, hi = config.split(",")
        if spec not in weak_denoisers:
            weak_denoisers[spec] = WeakDenoiser(spec, teacher, args.device)
        model = GuidedSampler(teacher, weak_denoisers[spec], args.steps, float(guidance), float(lo), float(hi))
        torch.manual_seed(args.seed)
        score = monitor(model, args.workdir)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"{stamp} {args.teacher}: Heun {args.steps} steps, seed {args.seed}, weak {spec}, guidance {float(guidance):g} for "
              f"{float(lo):g} < sigma <= {float(hi):g}, FID vs train reference {score:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teacher FID against the train reference, guided by a weaker denoiser")
    parser.add_argument("--teacher", default="checkpoints/edm/ema-0020971520-0.100.pt")
    parser.add_argument("--configs", nargs="+", required=True, help="weak,w,lowest,highest (see the module docstring)")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference", default="./results/reference_train")
    parser.add_argument("--workdir", required=True, help="Scratch directory for the samples; it is emptied before every measurement")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
