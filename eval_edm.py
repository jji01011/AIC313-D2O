"""Diagnostic: multi-step FID of a stage-0 EDM snapshot (Heun sampler), against the train-split reference.

This is not a submission path (it uses 2 * steps - 1 network evaluations per
image). It tells how good the pre-trained model is, which bounds what the
one-step D2O generator can be expected to reach.
"""

import argparse

import torch

from d2o.fid import FIDMonitor
from d2o.networks import EDMPrecond
from d2o.sampling import edm_sampler, one_hot


class EDMSampler(torch.nn.Module):
    """Gives an EDM the ``sample()`` interface expected by ``evaluate.generate_samples``."""

    def __init__(self, net, steps):
        super().__init__()
        self.net = net
        self.steps = steps

    def sample(self, shape, *, device="cuda", category=None, **kwargs):
        latents = torch.randn(shape, device=device)
        return edm_sampler(self.net, latents, one_hot(category.to(device), self.net.label_dim), num_steps=self.steps).clamp(-1, 1)


def main(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.edm, map_location="cpu", weights_only=False)
    net = EDMPrecond(**checkpoint["net_kwargs"], use_fp16=True)
    net.load_state_dict(checkpoint["ema"])
    model = EDMSampler(net, args.steps).to(device).eval()
    monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=device)
    torch.manual_seed(0)
    score = monitor(model, args.workdir)
    print(f"{args.edm}: {2 * args.steps - 1} NFE Heun, FID vs train reference {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-step FID of an EDM snapshot")
    parser.add_argument("--edm", required=True)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference", default="./results/reference_train")
    parser.add_argument("--workdir", default="./results/edm_fid_samples")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
