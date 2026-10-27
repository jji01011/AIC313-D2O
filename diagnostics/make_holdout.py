"""A held-out part of the train split for model selection: images the one-step generator's discriminator never sees.

Since 3,700 kimg the train-reference FID has kept falling (11.55 to 11.38) while the official score rose by about one
(22.08 to 23.33, measured for reporting only). The train reference is the very set the discriminator learns from, so it
rewards fitting those images. The README allows train images for hyper-parameter selection, so selection moves to
images that training does not use.

The split is the teammate's (FastGen-teammate/teacher_code_2026-10-13/teacher/data.py, draw_splits, seed 0, fixed by
them before any training): per category a random permutation of its images, the first 20 form the holdout H (3,020
images, 20 per category like the validation split) and the first 5 of those the canary C (755). Their teacher E112 was
trained on everything but C, so C is unseen by the teacher too; H is unseen by the generator's training from now on.

Writes results/holdout_h20.json (indices into load_train_tensors' order, which is train_split.txt's) and the reference
PNGs and statistics results/reference_holdout_h20{,_stats.npz} and results/reference_holdout_c5{,_stats.npz}, exported
as evaluate.py exports the validation images. Also prints a floor: 20 real images per category from the rest of the
train split against H. Only the train split is read.

    PYTHONPATH=. python diagnostics/make_holdout.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid
from torchvision.transforms.functional import to_pil_image

from d2o.data import load_train_tensors
from d2o.fid import folder_statistics


def draw_splits(labels, num_classes, seed=0, canary_per_class=5, holdout_per_class=20):
    """The teammate's draw_splits, canary and holdout part (same random calls in the same order)."""
    rng = np.random.RandomState(seed)
    canary, holdout = [], []
    for c in range(num_classes):
        perm = rng.permutation(np.flatnonzero(labels == c))
        holdout.append(perm[:holdout_per_class])
        canary.append(perm[:canary_per_class])
    return np.sort(np.concatenate(canary)), np.sort(np.concatenate(holdout))


def export(images, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    for count, image in enumerate(images):
        # evaluate.prepare_reference_set on the dataset tensor x = u8 / 127.5 - 1
        x = image.to(torch.float32) / 127.5 - 1
        to_pil_image(((x + 1) / 2).clamp(0, 1)).convert("RGB").save(folder / f"{count:04d}.png")


def main():
    images, labels = load_train_tensors("./data/pokemon-generation-one-22k", "./data/pokemon-generation-one-22k")
    labels_np = labels.numpy()
    num_classes = int(labels_np.max()) + 1
    canary, holdout = draw_splits(labels_np, num_classes)
    assert len(canary) == 5 * num_classes and len(holdout) == 20 * num_classes and set(canary) <= set(holdout)
    Path("results/holdout_h20.json").write_text(json.dumps(dict(
        source="teammate draw_splits, seed 0 (FastGen-teammate/teacher_code_2026-10-13/teacher/data.py)",
        holdout=[int(i) for i in holdout], canary=[int(i) for i in canary])))

    device = torch.device("cuda")
    feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    stats = {}
    for name, index in [("reference_holdout_h20", holdout), ("reference_holdout_c5", canary)]:
        export(images[index], f"results/{name}")
        mu, sigma = folder_statistics(f"results/{name}", feat_model, device)
        np.savez(f"results/{name}_stats.npz", mu=mu, sigma=sigma)
        stats[name] = (mu, sigma)
        print(f"{name}: {len(index)} images")

    # Floor: real images of the rest of the train split, 20 per category, against H (three draws).
    rest = np.setdiff1d(np.arange(len(labels_np)), holdout)
    for seed in range(3):
        rs = np.random.RandomState(1000 + seed)
        pick = np.sort(np.concatenate([rs.choice(rest[labels_np[rest] == c], 20, replace=False) for c in range(num_classes)]))
        export(images[pick], "results/holdout_floor_samples")
        mu, sigma = folder_statistics("results/holdout_floor_samples", feat_model, device)
        scores = [fid.frechet_distance(mu, sigma, *stats[name]) for name in stats]
        print(f"floor, real train images outside H (20 per category, draw {seed}): against H {scores[0]:.3f}, against C {scores[1]:.3f}")


if __name__ == "__main__":
    main()
