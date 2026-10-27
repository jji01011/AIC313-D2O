"""Sanity checks for the D2O models: parameter limits, NFE count, checkpoint round trip, conditioning.

Usage: python check_d2o.py [--mode one_nfe|few_nfe] [--checkpoint path/to/<mode>.ckpt] [--edm path/to/edm-*.pt]
                           [--d_teacher path/to/edm2-ema.pt [--d_teacher_min_res 16|8]]
"""

import argparse
import os
import tempfile

import torch

from d2o.config import GENERATOR_KWARGS
from d2o.discriminator import EDM2FeatureNetwork, ProjectedDiscriminator
from d2o.edm2 import EDM2Encoder
from d2o.networks import EDMPrecond, UNetBlock
from d2o.training_utils import apply_freeze
from evaluate import MAX_MODEL_PARAMETERS
from model import Model, ModelFewNFE, ModelOneNFE
from src.utils import count_parameters


def check(condition, message):
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        raise SystemExit(1)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_class, max_nfe = {"one_nfe": (ModelOneNFE, 1), "few_nfe": (ModelFewNFE, 4)}[args.mode]

    # Checkpoint save / load through the evaluation entry point.
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = os.path.join(tmp, f"{args.mode}.ckpt")
            torch.save({"state_dict": model_class().state_dict()}, checkpoint)
        model = Model.load_checkpoint(checkpoint, evaluate_mode=args.mode, device=device)
        resaved = os.path.join(tmp, "resaved.ckpt")
        torch.save({"state_dict": model.state_dict()}, resaved)
        reloaded = Model.load_checkpoint(resaved, evaluate_mode=args.mode, device=device)
    same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values()))
    check(same and not model.training, "checkpoint round trip through Model.load_checkpoint (strict, eval mode)")

    # Parameter limits: every learned model individually, and the generator together with its EMA copy.
    generator = model.count_parameters()
    edm = EDMPrecond(**GENERATOR_KWARGS)
    if args.edm is not None:
        edm.load_state_dict(torch.load(args.edm, map_location="cpu", weights_only=False)["ema"])
    # The discriminator as train_d2o.py builds it: with --d_teacher its features come from an EDM2 encoder instead.
    teacher_features = EDM2FeatureNetwork(EDM2Encoder.from_checkpoint(args.d_teacher, min_resolution=args.d_teacher_min_res)) if args.d_teacher else None
    discriminator = ProjectedDiscriminator(edm, c_dim=model.backbone.label_dim, feature_network=teacher_features)
    d_count = count_parameters(discriminator, trainable_only=False)
    d_name = f"frozen EDM2 encoder down to {args.d_teacher_min_res}x{args.d_teacher_min_res}" if args.d_teacher else "frozen EDM encoder"
    trainable = sum(param.numel() for param in apply_freeze(model.backbone, 1))
    model.requires_grad_(False)
    counts = {
        f"{model_class.__name__} (submitted generator)": generator,
        "generator + EMA copy during training": 2 * generator,
        "EDM teacher / initialisation": count_parameters(edm, trainable_only=False),
        f"discriminator ({d_name} + heads)": d_count,
        # The models of one training framework are summed, the discriminator included (TA on Slack, 2026-10-09).
        "D2O training framework, generator + discriminator": generator + d_count,
    }
    for name, value in counts.items():
        check(value < MAX_MODEL_PARAMETERS, f"{name}: {value:,} parameters")
    total = 2 * generator + d_count
    print(f"      with the generator's EMA copy counted as one more model: {total:,} parameters "
          f"({'inside' if total < MAX_MODEL_PARAMETERS else 'over'} the limit of {MAX_MODEL_PARAMETERS:,})")
    print(f"      D2O-F (freeze=1): {trainable:,} trainable generator parameters ({100 * (1 - trainable / generator):.1f}% frozen)")
    print(f"      discriminator heads: {count_parameters(discriminator.heads, trainable_only=False):,} trainable parameters")

    # Backbone evaluations per sample() call: exactly one for one_nfe, at most four for few_nfe.
    calls = {"backbone": 0, "unet": 0, "blocks": 0}
    num_blocks = sum(isinstance(module, UNetBlock) for module in model.backbone.modules())
    hooks = [
        model.backbone.register_forward_hook(lambda *_: calls.__setitem__("backbone", calls["backbone"] + 1)),
        model.backbone.model.register_forward_hook(lambda *_: calls.__setitem__("unet", calls["unet"] + 1)),
    ]
    hooks += [
        module.register_forward_hook(lambda *_: calls.__setitem__("blocks", calls["blocks"] + 1))
        for module in model.backbone.modules() if isinstance(module, UNetBlock)
    ]
    category = torch.arange(8, device=device) * 19
    torch.manual_seed(0)
    samples = model.sample((8, 3, 64, 64), device=device, category=category)
    for hook in hooks:
        hook.remove()
    nfe = calls["backbone"]
    expected = 1 if args.mode == "one_nfe" else int((model.sigmas > 0).sum())
    check(nfe == expected <= max_nfe and calls == {"backbone": nfe, "unet": nfe, "blocks": nfe * num_blocks},
          f"sample() runs the backbone {nfe} time(s), limit {max_nfe} (calls: {calls}, {num_blocks} blocks)")
    check(samples.shape == (8, 3, 64, 64) and samples.dtype == torch.float32, "sample() returns float32 [N, 3, 64, 64]")
    check(bool(torch.isfinite(samples).all()) and samples.min() >= -1 and samples.max() <= 1, "samples are finite and in [-1, 1]")

    # Class conditioning: same latents, different categories -> different images; same category -> identical.
    torch.manual_seed(0)
    same_category = model.sample((8, 3, 64, 64), device=device, category=category)
    torch.manual_seed(0)
    other_category = model.sample((8, 3, 64, 64), device=device, category=(category + 1) % model.backbone.label_dim)
    check(torch.equal(samples, same_category), "sampling is deterministic given the seed and categories")
    difference = (samples - other_category).abs().mean().item()
    check(difference > 0, f"changing only the category changes the output (mean abs difference {difference:.4f})")
    print("All checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks for ModelOneNFE / ModelFewNFE")
    parser.add_argument("--mode", default="one_nfe", choices=("one_nfe", "few_nfe"))
    parser.add_argument("--checkpoint", default=None, help="checkpoint of the selected mode; a randomly initialised model is used if omitted")
    parser.add_argument("--edm", default=None, help="stage-0 EDM snapshot for the discriminator feature network")
    parser.add_argument("--d_teacher", default=None, help="EDM2 checkpoint, if the run was trained with train_d2o.py --d_teacher")
    parser.add_argument("--d_teacher_min_res", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
