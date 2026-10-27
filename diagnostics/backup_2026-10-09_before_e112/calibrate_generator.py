"""Post-hoc calibration of a trained one-step generator, selected on the train split.

Three corrections, all written into the weights so that ``ModelOneNFE`` and
``evaluate.py`` stay unchanged and sampling is still one network evaluation:

- ``--gains``: output contrast. The generator leaves backgrounds slightly off
  pure white / black (9.5% of real pixels are exactly 255 against 2.4% of the
  generated ones at 1000 kimg of ``runs/d2o_one_final``); the discriminator
  cannot see this because it looks at noised images. Scaling the last
  convolution by g scales the image by g before ``sample()`` clamps it to
  [-1, 1]. g = 1.35 lowered the train-reference FID from 24.4 to 21.7.
- ``--smooths``: pixel-level noise. The samples carry slightly more
  high-frequency content than the real images, for the same reason. The last
  3x3 convolution W is replaced by the centre 3x3 of K * W, where
  K = (1 - a) * identity + a * binomial 3x3 blur. a = 0.2 gave another 0.7
  (21.36 to 20.66 at 1075 kimg, three seeds).
- ``--alphas`` (needs ``--state``): theta = theta_ema + alpha * (theta_raw - theta_ema).
  alpha = -0.25 gave 0.5 without the gain but nothing on top of it, so it is
  off by default.

- ``--label_scales``: strength of the class conditioning. The class embedding
  ``map_label`` is linear in the one-hot label, so scaling its weight by w is the
  same as feeding w times the label. Tested at 1300 kimg of ``runs/d2o_one_dn30``:
  w = 1 is best (19.5; 0.8 gives 21.0, 1.2 gives 20.4), so it stays at 1.

- ``--chromas``: colour saturation. The three output channels are mixed so that
  their mean stays and the deviations from it are scaled by c; with the gain g
  the image is g * (mean + c * (rgb - mean)) before the clamp.

- ``--offsets``: a constant added to the image, in grey levels (1 = 1/255 of the range). ``evaluate.py`` saves the
  samples with ``to_pil_image``, which truncates instead of rounding, so every pixel that is not clamped lands half a
  grey level low on average; an offset of 0.5 turns the truncation into rounding. It is added to the bias of the last
  convolution after the gain.

Every combination is scored against the train reference
like the training monitor; ``--out`` writes the best one in the format of
``best.ckpt``. No network is trained and the validation split is not read.

    python calibrate_generator.py --ckpt runs/d2o_one_final/best.ckpt --gains=1.2,1.35,1.5 --smooths=0,0.2 --seeds 0,1,2 \
        --out runs/d2o_one_final/best_calibrated.ckpt
"""

import argparse

import torch
import torch.nn.functional as F

from d2o.fid import FIDMonitor
from model import ModelOneNFE

# Last convolution of the U-Net (weight and bias). The image is c_skip * x + c_out * F(x) with c_skip = 4e-5 at sigma 80,
# so scaling this layer scales the image.
OUTPUT_LAYER = ("model.dec.64x64_aux_conv.weight", "model.dec.64x64_aux_conv.bias")
LABEL_LAYER = "model.map_label.weight"


def main(args):
    device = torch.device(args.device)
    assert bool(args.ckpt) != bool(args.state), "give either --ckpt or --state"
    source = torch.load(args.ckpt or args.state, map_location="cpu", weights_only=False)
    model = ModelOneNFE(device=device, sigma_g=source["args"]["sigma_g"], **source["net_kwargs"]).to(device)
    model.load_state_dict(source["state_dict"] if args.ckpt else source["ema_model"])
    model.eval().requires_grad_(False)
    ema = {key: value.clone() for key, value in model.backbone.state_dict().items()}
    raw = {key: value.to(device) for key, value in source["net"].items()} if args.state else None
    assert raw is not None or args.alphas == [0.0], "--alphas needs the raw generator of a --state file"
    print(f"{args.ckpt or args.state}: {source['cur_nimg'] / 1e3:.0f} kimg")

    binomial = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64, device=device) / 4
    binomial = binomial[:, None] * binomial[None, :]

    c_out = float(model.backbone.coefficients(model.sigma_g.to(device))[1])  # image = c_skip * x + c_out * F(x)

    def load(alpha, gain, smooth, label_scale, chroma=1.0, offset=0.0):
        weights = dict(ema)
        if alpha:
            weights = {key: (ema[key].double() + alpha * (raw[key].double() - ema[key].double())).to(ema[key].dtype)
                       if ema[key].is_floating_point() else ema[key] for key in ema}
        if smooth:
            weight = weights[OUTPUT_LAYER[0]]
            kernel = smooth * binomial
            kernel[1, 1] += 1 - smooth
            pairs = weight.shape[0] * weight.shape[1]
            full = F.conv2d(weight.double().reshape(1, pairs, 3, 3), kernel[None, None].repeat(pairs, 1, 1, 1), padding=2, groups=pairs)  # 5x5
            weights[OUTPUT_LAYER[0]] = full[0, :, 1:4, 1:4].reshape(weight.shape).to(weight.dtype)
        if chroma != 1:
            mix = chroma * torch.eye(3, dtype=torch.float64, device=device) + (1 - chroma) / 3
            for key in OUTPUT_LAYER:
                weights[key] = torch.tensordot(mix, weights[key].double(), dims=1).to(weights[key].dtype)
        for key in OUTPUT_LAYER:
            weights[key] = weights[key] * gain
        if offset:
            weights[OUTPUT_LAYER[1]] = weights[OUTPUT_LAYER[1]] + offset / 127.5 / c_out
        weights[LABEL_LAYER] = weights[LABEL_LAYER] * label_scale
        model.backbone.load_state_dict(weights)

    monitor = None
    if args.fid:
        monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=device)
        grid_categories = torch.arange(64, device=device) % model.backbone.label_dim
        probe_categories = torch.arange(8, device=device).repeat_interleave(8)
    scores = {}
    combinations = [(alpha, gain, smooth, label_scale, chroma, offset) for alpha in args.alphas for gain in args.gains for smooth in args.smooths
                    for label_scale in args.label_scales for chroma in args.chromas for offset in args.offsets]
    with_offset = args.offsets != [0.0]  # the offset is only named in the output when it is in use
    describe = lambda alpha, gain, smooth, label_scale, chroma, offset: (
        f"alpha {alpha:+.2f} gain {gain:.2f} smooth {smooth:.2f} label {label_scale:.2f} chroma {chroma:.2f}" + (f" offset {offset:.2f}" if with_offset else ""))
    for alpha, gain, smooth, label_scale, chroma, offset in combinations:
        load(alpha, gain, smooth, label_scale, chroma, offset)
        if monitor is None:
            continue
        fids = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            # Seed 0 then draws the same latents as evaluate_and_save in train_d2o.py, so (0, 1) reproduces the logged value.
            model.sample((64, 3, 64, 64), device=device, category=grid_categories)
            model.sample((64, 3, 64, 64), device=device, category=probe_categories)
            fids.append(monitor(model, args.workdir))
        scores[alpha, gain, smooth, label_scale, chroma, offset] = sum(fids) / len(fids)
        print(f"{describe(alpha, gain, smooth, label_scale, chroma, offset)}  train-reference FID {scores[alpha, gain, smooth, label_scale, chroma, offset]:.3f}"
              f"  ({', '.join(f'{value:.3f}' for value in fids)}; seeds {args.seeds})", flush=True)

    if args.out:
        alpha, gain, smooth, label_scale, chroma, offset = min(scores, key=scores.get) if scores else combinations[0]
        load(alpha, gain, smooth, label_scale, chroma, offset)
        torch.save(dict(state_dict=model.state_dict(), net_kwargs=source["net_kwargs"], cur_nimg=source["cur_nimg"], args=source["args"],
                        ema_extrapolation_alpha=alpha, output_gain=gain, output_smooth=smooth, label_scale=label_scale, output_chroma=chroma,
                        output_offset=offset), args.out)
        print(f"wrote {args.out} with {describe(alpha, gain, smooth, label_scale, chroma, offset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score and export calibrated one-step generators")
    parser.add_argument("--ckpt", default=None, help="Snapshot in the format of best.ckpt (EMA generator only)")
    parser.add_argument("--state", default=None, help="d2o-state-latest.pt (EMA and raw generator), needed for --alphas")
    parser.add_argument("--gains", type=lambda text: [float(value) for value in text.split(",")], default=[1.0],
                        help="Comma-separated output gains folded into the last convolution; 1 = unchanged")
    parser.add_argument("--smooths", type=lambda text: [float(value) for value in text.split(",")], default=[0.0],
                        help="Comma-separated output smoothing strengths folded into the last convolution; 0 = unchanged")
    parser.add_argument("--label_scales", type=lambda text: [float(value) for value in text.split(",")], default=[1.0],
                        help="Comma-separated scales of the class embedding weight; 1 = unchanged")
    parser.add_argument("--chromas", type=lambda text: [float(value) for value in text.split(",")], default=[1.0],
                        help="Comma-separated colour saturation factors folded into the last convolution; 1 = unchanged")
    parser.add_argument("--offsets", type=lambda text: [float(value) for value in text.split(",")], default=[0.0],
                        help="Comma-separated constants added to the image, in grey levels; 0.5 makes the truncating PNG export round")
    parser.add_argument("--alphas", type=lambda text: [float(value) for value in text.split(",")], default=[0.0],
                        help="Comma-separated; 0 = the EMA generator, 1 = the raw generator, negative = away from the raw generator")
    parser.add_argument("--seeds", type=lambda text: [int(value) for value in text.split(",")], default=[0])
    parser.add_argument("--fid", type=int, default=1, help="0 = only export the first combination")
    parser.add_argument("--out", default=None, help="Write the best-scoring combination as a checkpoint in the format of best.ckpt")
    parser.add_argument("--workdir", default="./results/calibrate_fid_samples", help="Scratch folder for the generated PNGs (emptied on every evaluation)")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference", default="./results/reference_train")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
