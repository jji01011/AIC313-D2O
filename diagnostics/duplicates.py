"""Near-duplicates inside the train split, and between the held-out H and the rest of it (train split only).

Real train images score 14.9 against the held-out H (diagnostics/make_holdout.py) but 21.0 against the validation
split (results/official_fid.log, 2026-10-08). On 2026-10-13 39% of the train images had another train image within
0.10 pixel RMS. If the validation split was cut without such near-copies of train images and H was not (H is a random
draw), the gap is the copies: then a held-out set built from images without a near-copy in the rest of the train split
should score against it as the validation split does, and selection on it would see what the validation split sees.

For every train image: the nearest other train image in pixels (RMS on [-1, 1], horizontal flips included) and in
Inception features (results/reference_train_balanced_features.npz, rows in train order). Near-copies (pixel RMS below
--threshold, flips included) are grouped into connected components. Prints the counts and writes
results/duplicates.json (nearest distances, component of every image). Nothing of the validation split is read.

    PYTHONPATH=. python diagnostics/duplicates.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from d2o.data import load_train_tensors


def pairs_below(bank, threshold, chunk=128):
    """Nearest other row of every row (distance, index) and all pairs (i < j) closer than threshold, flips included."""
    n, dim = bank.shape[0], bank[0].numel()
    flat = bank.flatten(1)
    flipped = bank.flip(-1).flatten(1)
    best = torch.full([n], float("inf"), device=bank.device)
    best_index = torch.zeros(n, dtype=torch.long, device=bank.device)
    found = []
    for start in range(0, n, chunk):
        rows = flat[start:start + chunk]
        d = torch.minimum(torch.cdist(rows, flat), torch.cdist(rows, flipped)) / dim ** 0.5
        d[torch.arange(len(rows)), torch.arange(start, start + len(rows))] = float("inf")
        value, index = d.min(dim=1)
        best[start:start + len(rows)], best_index[start:start + len(rows)] = value, index
        i, j = torch.nonzero(d < threshold, as_tuple=True)
        keep = (i + start) < j
        found.append(torch.stack([i[keep] + start, j[keep]], dim=1).cpu())
    return best.cpu(), best_index.cpu(), torch.cat(found).numpy()


def components(n, pairs):
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([root(a) for a in range(n)])


def main(args):
    device = torch.device(args.device)
    images, labels = load_train_tensors(args.data_root, args.split_dir)
    labels = labels.numpy()
    pixels = images.to(device, torch.float32) / 127.5 - 1
    best, best_index, pairs = pairs_below(pixels, args.threshold)
    comp = components(len(labels), pairs)
    sizes = np.bincount(comp, minlength=len(labels))[comp]
    feats = torch.from_numpy(np.load(args.features)["features"]).to(device)
    fbest = []
    for s in range(0, len(feats), 512):
        d = torch.cdist(feats[s:s + 512], feats)
        d[torch.arange(len(d)), torch.arange(s, s + len(d))] = float("inf")
        fbest.append(d.min(dim=1).values.cpu())
    fbest = torch.cat(fbest)

    same_class = labels[best_index.numpy()] == labels
    print(f"{len(labels):,} train images; near-copy = another train image within {args.threshold} pixel RMS (flips included)")
    print(f"  with a near-copy: {100 * (best < args.threshold).float().mean():.1f}%  (nearest of the same category: {100 * same_class.mean():.1f}%)")
    print(f"  pairs below the threshold: {len(pairs):,}; components with more than one image: {len(np.unique(comp[sizes > 1])):,}, "
          f"largest {sizes.max()}; images in such components {100 * (sizes > 1).mean():.1f}%")
    print(f"  nearest other image, pixels: median {best.median():.4f}, 10% {best.quantile(0.1):.4f}; "
          f"features: median {fbest.median():.3f}, 10% {fbest.quantile(0.1):.3f}")

    held = np.array(json.load(open(args.holdout))["holdout"])
    rest = np.setdiff1d(np.arange(len(labels)), held)
    rest_pixels = pixels[rest]
    h_pix = []
    for s in range(0, len(held), 128):
        rows = pixels[held[s:s + 128]].flatten(1)
        d = torch.minimum(torch.cdist(rows, rest_pixels.flatten(1)), torch.cdist(rows, rest_pixels.flip(-1).flatten(1))) / rows.shape[1] ** 0.5
        h_pix.append(d.min(dim=1).values.cpu())
    h_pix = torch.cat(h_pix)
    h_feat = torch.cat([torch.cdist(feats[held[s:s + 512]], feats[rest]).min(dim=1).values.cpu() for s in range(0, len(held), 512)])
    print(f"held-out H ({len(held)}) against the rest of the train split: with a near-copy there {100 * (h_pix < args.threshold).float().mean():.1f}%; "
          f"pixels median {h_pix.median():.4f}, 10% {h_pix.quantile(0.1):.4f}; features median {h_feat.median():.3f}, 10% {h_feat.quantile(0.1):.3f}")
    for cut in (0.05, 0.10, 0.15, 0.20):
        print(f"  H images with a rest image within {cut:.2f} pixel RMS: {100 * (h_pix < cut).float().mean():.1f}%")

    Path(args.out).write_text(json.dumps(dict(
        threshold=args.threshold, nearest_pixel=[round(float(v), 5) for v in best], nearest_index=[int(v) for v in best_index],
        nearest_feature=[round(float(v), 4) for v in fbest], component=[int(v) for v in comp])))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Near-duplicates inside the train split")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--features", default="./results/reference_train_balanced_features.npz")
    parser.add_argument("--holdout", default="./results/holdout_h20.json")
    parser.add_argument("--out", default="./results/duplicates.json")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
