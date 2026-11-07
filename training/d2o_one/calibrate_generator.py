import argparse

import torch
import torch.nn.functional as F

from d2o.fid import FIDMonitor
from model import ModelOneNFE

OUTPUT_LAYER = ("model.dec.64x64_aux_conv.weight", "model.dec.64x64_aux_conv.bias")
LABEL_LAYER = "model.map_label.weight"
EDM2_OUTPUT = ("unet.out_conv.weight", "unet.out_gain", "mean_image")


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

    edm2 = model.family == "edm2"
    if edm2:
        assert args.label_scales == [1.0] and args.chromas == [1.0], "--label_scales and --chromas are not available for the EDM2 generator"
    else:
        c_out = float(model.backbone.coefficients(model.sigma_g.to(device))[1])

    def blurred(weight, smooth):
        kernel = smooth * binomial
        kernel[1, 1] += 1 - smooth
        pairs = weight.shape[0] * weight.shape[1]
        full = F.conv2d(weight.double().reshape(1, pairs, 3, 3), kernel[None, None].repeat(pairs, 1, 1, 1), padding=2, groups=pairs)
        return full[0, :, 1:4, 1:4].reshape(weight.shape)

    def load(alpha, gain, smooth, label_scale, chroma=1.0, offset=0.0):
        weights = dict(ema)
        if alpha:
            weights = {key: (ema[key].double() + alpha * (raw[key].double() - ema[key].double())).to(ema[key].dtype)
                       if ema[key].is_floating_point() else ema[key] for key in ema}
        if edm2:
            conv, out_gain, mean_image = EDM2_OUTPUT
            scale = gain
            if smooth:
                smoothed = blurred(weights[conv], smooth)
                scale = scale * float((smoothed.flatten(1).norm(dim=1) / weights[conv].double().flatten(1).norm(dim=1)).mean())
                weights[conv] = smoothed.to(weights[conv].dtype)
            weights[out_gain] = weights[out_gain] * scale
            weights[mean_image] = weights[mean_image] * gain + offset / 127.5
            model.backbone.load_state_dict(weights)
            return
        if smooth:
            weights[OUTPUT_LAYER[0]] = blurred(weights[OUTPUT_LAYER[0]], smooth).to(weights[OUTPUT_LAYER[0]].dtype)
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
    with_offset = args.offsets != [0.0]
    describe = lambda alpha, gain, smooth, label_scale, chroma, offset: (
        f"alpha {alpha:+.2f} gain {gain:.2f} smooth {smooth:.2f} label {label_scale:.2f} chroma {chroma:.2f}" + (f" offset {offset:.2f}" if with_offset else ""))
    for alpha, gain, smooth, label_scale, chroma, offset in combinations:
        load(alpha, gain, smooth, label_scale, chroma, offset)
        if monitor is None:
            continue
        fids = []
        for seed in args.seeds:
            torch.manual_seed(seed)
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
