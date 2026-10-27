"""Multi-step FID of the teammate's EDM2 teacher against the train-split reference, measured like our own EDM
(eval_edm.py: 3,020 images, 20 per category, deterministic Heun sampler). Not a submission path.

    PYTHONPATH=. python diagnostics/teammate_edm2/eval_teacher.py --steps 18 --guidance 1
"""

import argparse
import sys

import torch

sys.path.insert(0, "diagnostics/teammate_edm2")
from edm2_teacher import load_teacher  # noqa: E402

from d2o.fid import FIDMonitor  # noqa: E402
from d2o.sampling import edm_sampler, save_image_grid  # noqa: E402


class TeacherSampler(torch.nn.Module):
    """Gives the teacher the ``sample()`` interface expected by ``evaluate.generate_samples``."""

    def __init__(self, teacher, steps, guidance):
        super().__init__()
        self.teacher, self.steps, self.guidance = teacher, steps, guidance

    def denoise(self, x, sigma, category):
        cond = self.teacher(x.to(torch.float32), sigma, category)
        if self.guidance == 1:
            return cond
        null = self.teacher(x.to(torch.float32), sigma, None)  # classifier-free guidance with the null class
        return null + self.guidance * (cond - null)

    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        latents = torch.randn(shape, device=device)
        return edm_sampler(self.denoise, latents, category.to(device), num_steps=self.steps).clamp(-1, 1)


def main(args):
    teacher, _ = load_teacher(args.teacher, device=args.device)
    model = TeacherSampler(teacher, args.steps, args.guidance)
    if args.grid:
        torch.manual_seed(0)
        categories = torch.tensor([103, 13, 9, 129, 30, 40, 127, 82] * 2)  # Pikachu ... Mewtwo, two rows
        save_image_grid(model.sample((16, 3, 64, 64), device=args.device, category=categories), args.grid, columns=8)
    monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=args.device)
    torch.manual_seed(0)
    score = monitor(model, args.workdir)
    nfe = (2 * args.steps - 1) * (1 if args.guidance == 1 else 2)
    print(f"{args.teacher}: {nfe} NFE Heun ({args.steps} steps), guidance {args.guidance:g}, FID vs train reference {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-step FID of the teammate's EDM2 teacher")
    parser.add_argument("--teacher", default="checkpoints/edm/ema-0016777216-0.100.pt")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--grid", default=None, help="Also save a small sample grid to this PNG")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference", default="./results/reference_train")
    parser.add_argument("--workdir", default="./results/teacher_edm2_fid_samples")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
