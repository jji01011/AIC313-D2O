"""Class-balanced train reference statistics for the FID monitor.

``evaluate.py`` generates 20 images for each of the 151 categories, but the train split holds 46 to 183 images per
category, so the plain train reference (``results/reference_train``) weights frequent categories up to four times more
than the generated set does. This script reads the Inception features of the same 17,079 train PNGs and weights every
image by 1 / (151 * images of its category), so that each category counts once, as in the generated set.

It writes ``<out>_stats.npz`` (mu, sigma) next to ``results/reference_train``; ``FIDMonitor`` and
``calibrate_generator.py --fid_reference <out>`` read it as they read the plain one. The per-image features and
categories go to ``<out>_features.npz``. Only the train split is read.

    PYTHONPATH=. python diagnostics/balanced_reference.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid

from dataset import PokemonDataModule


def main(args):
    module = PokemonDataModule(data_root=args.data_root, split_dir=args.split_dir, batch_size=64)
    categories = np.array(module.train_dataset.categories)
    reference = Path(args.reference)
    files = [str(reference / f"{index:04d}.png") for index in range(len(categories))]
    assert all(Path(file).exists() for file in files), "the train reference PNGs are missing"
    assert not (reference / f"{len(categories):04d}.png").exists(), "more PNGs than train images"

    feat_model = fid.build_feature_extractor("clean", torch.device(args.device), use_dataparallel=False)
    feats = fid.get_files_features(files, feat_model, num_workers=4, batch_size=64, device=torch.device(args.device), mode="clean")

    names, inverse, counts = np.unique(categories, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(names) * counts[inverse])
    mu = np.average(feats, axis=0, weights=weights)
    sigma = np.cov(feats, rowvar=False, aweights=weights)

    out = Path(args.out)
    np.savez(out.with_name(out.name + "_stats.npz"), mu=mu, sigma=sigma)
    np.savez(out.with_name(out.name + "_features.npz"), features=feats, categories=categories)

    plain = np.load(reference.with_name(reference.name + "_stats.npz"))
    print(f"{len(feats)} images, {len(names)} categories ({counts.min()} to {counts.max()} each)")
    print(f"FID between the plain and the balanced train statistics: {fid.frechet_distance(mu, sigma, plain['mu'], plain['sigma']):.3f}")
    print(f"recomputed plain statistics against the stored ones: "
          f"{fid.frechet_distance(feats.mean(0), np.cov(feats, rowvar=False), plain['mu'], plain['sigma']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-balanced train reference statistics")
    parser.add_argument("--reference", default="./results/reference_train")
    parser.add_argument("--out", default="./results/reference_train_balanced")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
