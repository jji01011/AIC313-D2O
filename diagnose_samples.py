"""Where does a generator lose FID? A breakdown of saved sample folders against the *train* split.

For every folder of 3,020 samples (20 per category, as written by ``d2o/fid.py``
or ``eval_edm.py``) this prints, in the clean-fid Inception feature space:

- the train-reference FID and its two terms (mean shift, covariance);
- precision / recall (Kynkaanniemi et al., NeurIPS 2019) and density / coverage
  (Naeem et al., ICML 2020) against a class-balanced real subset of the same size;
- class consistency: how often the nearest real image of a sample has its class.

A second, disjoint real subset is scored the same way and gives the ceiling of
every column. Analysis only: nothing here enters training, and the validation
split is not read.

    python diagnose_samples.py teacher=results/edm_fid_samples onestep=runs/d2o_one_final/fid_samples
"""

import argparse
import glob
import os

import numpy as np
import torch
from cleanfid import fid

SAMPLES_PER_CATEGORY = 20


def folder_features(folder, feat_model, device):
    """Features in clean-fid's order, i.e. sorted by file name."""
    return fid.get_folder_features(folder, feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False)


def reference_features(reference_dir, feat_model, device):
    """Features of the train reference PNGs in dataset order (row i = ``<i>.png``), cached next to the folder."""
    cache_path = reference_dir.rstrip("/") + "_feats.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path)
    names = sorted(os.path.basename(path) for path in glob.glob(os.path.join(reference_dir, "*.png")))
    feats = folder_features(reference_dir, feat_model, device)
    feats = feats[np.argsort([int(name.split(".")[0]) for name in names])]
    np.save(cache_path, feats)
    return feats


def prdc(real, fake, k):
    """Precision, recall, density, coverage with k-nearest-neighbour balls."""
    real, fake = torch.from_numpy(real).cuda().double(), torch.from_numpy(fake).cuda().double()
    radius_real = torch.cdist(real, real).kthvalue(k + 1, dim=1).values  # +1: the point itself
    radius_fake = torch.cdist(fake, fake).kthvalue(k + 1, dim=1).values
    dist = torch.cdist(real, fake)
    inside_real = dist < radius_real[:, None]
    precision = inside_real.any(0).double().mean().item()
    recall = (dist < radius_fake[None, :]).any(1).double().mean().item()
    density = inside_real.double().sum(0).mean().item() / k
    coverage = (dist.min(1).values < radius_real).double().mean().item()
    same_class = (dist.argmin(0) // SAMPLES_PER_CATEGORY == torch.arange(len(fake), device=dist.device) // SAMPLES_PER_CATEGORY)
    return precision, recall, density, coverage, same_class.double().mean().item()


def main(args):
    device = torch.device(args.device)
    feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    ref_feats = reference_features(args.fid_reference, feat_model, device)
    ref_mu, ref_sigma = ref_feats.mean(0), np.cov(ref_feats, rowvar=False)

    # Two disjoint class-balanced real subsets, class by class like the sample folders.
    labels = torch.load(args.train_cache, weights_only=True)["labels"].numpy()
    assert len(labels) == len(ref_feats)
    rng = np.random.default_rng(args.seed)
    rows = [rng.permutation(np.flatnonzero(labels == c))[: 2 * SAMPLES_PER_CATEGORY] for c in range(labels.max() + 1)]
    real_a = ref_feats[np.concatenate([r[:SAMPLES_PER_CATEGORY] for r in rows])]
    real_b = ref_feats[np.concatenate([r[SAMPLES_PER_CATEGORY:] for r in rows])]

    sets = {"real train subset (ceiling)": real_b}
    for item in args.folders:
        name, folder = item.split("=", 1)
        sets[name] = folder_features(folder, feat_model, device)
        assert len(sets[name]) == len(real_a), f"{folder}: expected {len(real_a)} samples"

    print(f"{'samples':30s} {'FID':>6s} {'mean':>6s} {'cov':>6s} | {'prec':>5s} {'rec':>5s} {'dens':>5s} {'cover':>5s} | {'class':>5s}")
    for name, feats in sets.items():
        mu, sigma = feats.mean(0), np.cov(feats, rowvar=False)
        total = float(fid.frechet_distance(mu, sigma, ref_mu, ref_sigma))
        mean_term = float(((mu - ref_mu) ** 2).sum())
        values = prdc(real_a, feats, args.k)
        print(f"{name:30s} {total:6.2f} {mean_term:6.2f} {total - mean_term:6.2f} | " + " ".join(f"{v:5.3f}" for v in values[:4]) + f" | {values[4]:5.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID breakdown of saved sample folders against the train split")
    parser.add_argument("folders", nargs="+", help="name=folder, each folder holding 20 samples per category named <category>_<index>.png")
    parser.add_argument("--fid_reference", default="./results/reference_train", help="Train reference PNG folder (built by d2o/fid.py)")
    parser.add_argument("--train_cache", default="./data/cache/train_64x64.pt", help="Train tensors with the labels in dataset order (built by d2o/data.py)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
