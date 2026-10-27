"""How close do generated images come to single training images? (train split only)

For every image of a folder: the distance to its nearest training image, in Inception features (the space the FID is
computed in) and in pixels. Real train images (20 per category) are measured against the other training images. A model whose samples sit
closer to training images than real images sit to each other is reproducing them; a lower FID against the train
reference can come from exactly that, and then it does not carry over to images the model has not seen.

    PYTHONPATH=. python diagnostics/nearest_train.py name=folder [name=folder ...]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid
from PIL import Image

from d2o.data import load_train_tensors


def nearest(queries, bank, own=None, chunk=256):
    """Distance of every query row to its nearest bank row; ``own``: the bank index of each query, left out."""
    found = []
    for start in range(0, len(queries), chunk):
        distances = torch.cdist(queries[start:start + chunk].double(), bank.double())
        if own is not None:
            distances[torch.arange(len(distances)), own[start:start + chunk]] = float("inf")
        found.append(distances.min(dim=1).values)
    return torch.cat(found).float()


def report(name, f, p):
    print(f"{name:28s} {len(f):5d} images | features: median {f.median():6.3f}, 10% {f.quantile(0.1):6.3f} | "
          f"pixels (RMS): median {p.median():.4f}, 10% {p.quantile(0.1):.4f}, below 0.10: {100 * (p < 0.10).float().mean():.2f}%", flush=True)


def main(args):
    device = torch.device(args.device)
    feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    train_feats = torch.from_numpy(np.load(args.train_feats)).to(device, torch.float32)
    images, labels = load_train_tensors(args.data_root, args.split_dir)
    train_pixels = images.flatten(1).to(device, torch.float32) / 127.5 - 1
    scale = train_pixels.shape[1] ** 0.5
    print(f"{len(train_feats):,} training images. Columns: median and 10th percentile of the distance to the nearest training image")
    # Real images against the other training images: 20 per category, as in the generated sets (the rows of the feature
    # file are in the order of the train tensors: both follow the train dataset).
    picks = torch.cat([torch.nonzero(labels == c).flatten()[:20] for c in range(int(labels.max()) + 1)]).to(device)
    report("real train images", nearest(train_feats[picks], train_feats, picks), nearest(train_pixels[picks], train_pixels, picks) / scale)
    for pair in args.folders:
        name, folder = pair.split("=", 1)
        feats = torch.from_numpy(fid.get_folder_features(folder, feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False)).to(device, torch.float32)
        pixels = torch.stack([torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))) for path in sorted(Path(folder).glob("*.png"))])
        pixels = pixels.permute(0, 3, 1, 2).flatten(1).to(device, torch.float32) / 127.5 - 1
        report(name, nearest(feats, train_feats), nearest(pixels, train_pixels) / scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance of generated images to their nearest training image")
    parser.add_argument("folders", nargs="+", help="name=folder of PNGs")
    parser.add_argument("--train_feats", default="./results/reference_train_feats.npy", help="Inception features of the train reference")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
