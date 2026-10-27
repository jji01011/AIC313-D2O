"""Where does the one-step generator stand against its teacher and the real images? (train split only)

Three sets of 3,020 images (20 per category), all scored against the train reference like d2o/fid.py does:
real train images, the teacher's stored samples (results/teacher_edm2_fid_samples, 35 NFE, no guidance) and fresh
samples of a one-step checkpoint. For each: the FID split into its mean term and its covariance term, the total
variance of the Inception features, the same variance inside the categories, and the pixel distance between images of
one category (the measure train_d2o.py logs as intra_class_distance, here over all categories).
--teacher_gains repeats the teacher's row with its images multiplied by a gain and clamped, which is what
calibrate_generator.py does to the generator: does the teacher gain from it too?

    PYTHONPATH=. python diagnostics/gap_probe.py --ckpt runs/<run>/calib-<kimg>.ckpt --workdir <an empty scratch directory>
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid
from PIL import Image

from d2o.data import load_train_tensors
from evaluate import SAMPLES_PER_CATEGORY, generate_samples
from model import ModelOneNFE


def read_folder(folder):
    """uint8 [N, 64, 64, 3] and the category of every image, in the order clean-fid reads the folder."""
    paths = sorted(Path(folder).glob("*.png"))
    return np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths]), np.array([int(path.name[:4]) for path in paths])


def write_folder(folder, images, categories):
    shutil.rmtree(folder, ignore_errors=True)
    Path(folder).mkdir(parents=True)
    counts = {}
    for image, category in zip(images, categories):
        index = counts.get(int(category), 0)
        counts[int(category)] = index + 1
        Image.fromarray(image).save(Path(folder) / f"{int(category):04d}_{index:02d}.png")


def describe(name, folder, feat_model, reference, device):
    feats = fid.get_folder_features(str(folder), feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False)
    images, categories = read_folder(folder)
    mu, sigma = feats.mean(axis=0), np.cov(feats, rowvar=False)
    total = float(fid.frechet_distance(mu, sigma, reference["mu"], reference["sigma"]))
    mean_term = float(np.sum((mu - reference["mu"]) ** 2))
    inside = np.mean([feats[categories == c].var(axis=0, ddof=1).sum() for c in np.unique(categories)])
    pixels = torch.from_numpy(images).to(torch.float32).flatten(1) / 127.5 - 1
    distance = np.mean([torch.cdist(pixels[categories == c], pixels[categories == c]).mean().item() for c in np.unique(categories)]) / pixels.shape[1] ** 0.5
    white = float((images == 255).all(axis=-1).mean())
    print(f"{name:34s} FID {total:7.3f} = mean {mean_term:6.3f} + covariance {total - mean_term:6.3f} | feature variance {np.trace(sigma):7.2f}, "
          f"inside categories {inside:7.2f} | pixel distance inside categories {distance:.4f} | pure white pixels {100 * white:5.2f}%", flush=True)


def main(args):
    device = torch.device(args.device)
    workdir = Path(args.workdir)
    feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    reference = np.load(args.fid_reference_stats)
    print(f"reference: feature variance {np.trace(reference['sigma']):.2f}")

    images, labels = load_train_tensors(args.data_root, args.split_dir)
    if not args.only_ckpt:
        picks = torch.cat([torch.nonzero(labels == c).flatten()[:SAMPLES_PER_CATEGORY] for c in range(int(labels.max()) + 1)])
        write_folder(workdir / "real", images[picks].permute(0, 2, 3, 1).numpy(), labels[picks].numpy())
        describe("real train images", workdir / "real", feat_model, reference, device)
        describe("teacher, 35 NFE", args.teacher_samples, feat_model, reference, device)
    teacher_images, teacher_categories = read_folder(args.teacher_samples) if args.teacher_gains else (None, None)
    for gain in args.teacher_gains:
        scaled = np.clip((teacher_images.astype(np.float32) / 127.5 - 1) * gain, -1, 1)
        write_folder(workdir / "teacher_gain", np.round((scaled + 1) * 127.5).astype(np.uint8), teacher_categories)
        describe(f"teacher, 35 NFE, gain {gain:g}", workdir / "teacher_gain", feat_model, reference, device)

    for path in args.ckpt:
        source = torch.load(path, map_location="cpu", weights_only=False)
        model = ModelOneNFE(device=device, sigma_g=source["args"]["sigma_g"], **source["net_kwargs"]).to(device)
        model.load_state_dict(source["state_dict"])
        categories = torch.arange(int(labels.max()) + 1).repeat_interleave(SAMPLES_PER_CATEGORY)
        for seed in args.seeds:
            torch.manual_seed(seed)
            shutil.rmtree(workdir / "student", ignore_errors=True)
            generate_samples(model, workdir / "student", categories, batch_size=64, device=device)
            describe(f"{Path(path).parent.name}/{Path(path).name} seed {seed}"[-34:], workdir / "student", feat_model, reference, device)
        del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID terms and diversity of real images, the teacher's samples and a one-step checkpoint")
    parser.add_argument("--ckpt", nargs="*", default=[], help="Checkpoints in the format of best.ckpt")
    parser.add_argument("--seeds", type=lambda text: [int(value) for value in text.split(",")], default=[0])
    parser.add_argument("--teacher_samples", default="./results/teacher_edm2_fid_samples")
    parser.add_argument("--only_ckpt", type=int, default=0, help="1 = only the rows of --ckpt (runs/nearest_check.sh)")
    parser.add_argument("--teacher_gains", type=lambda text: [float(value) for value in text.split(",") if value], default=[])
    parser.add_argument("--workdir", required=True, help="Scratch directory for the PNGs")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--fid_reference_stats", default="./results/reference_train_stats.npz")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
