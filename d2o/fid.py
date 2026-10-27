"""FID monitoring during training.

Samples are generated and scored exactly as in ``evaluate.py`` (20 samples per
category through ``model.sample()``, PNG export, clean-fid), but the reference
set is built from the *train* split: the project rules reserve the validation
split for the final FID and forbid using it for model selection.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from cleanfid import fid
from torch.utils.data import DataLoader

from dataset import PokemonDataModule
from evaluate import SAMPLES_PER_CATEGORY, generate_samples, prepare_reference_set


def folder_statistics(folder, feat_model, device):
    feats = fid.get_folder_features(str(folder), feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


class FIDMonitor:
    def __init__(self, data_root, split_dir, reference_dir="./results/reference_train", device="cuda"):
        self.device = torch.device(device)
        module = PokemonDataModule(data_root=data_root, split_dir=split_dir, batch_size=64)
        self.categories = torch.arange(len(module.category_to_id), dtype=torch.long).repeat_interleave(SAMPLES_PER_CATEGORY)
        self.feat_model = fid.build_feature_extractor("clean", self.device, use_dataparallel=False)

        reference_dir = Path(reference_dir)
        stats_path = reference_dir.with_name(reference_dir.name + "_stats.npz")
        if not stats_path.exists():
            if not any(reference_dir.glob("*.png")):
                # Same export routine as evaluate.py, pointed at the train split.
                train_loader = DataLoader(module.train_dataset, batch_size=64, shuffle=False, num_workers=4)
                prepare_reference_set(SimpleNamespace(val_dataloader=lambda: train_loader), reference_dir)
            mu, sigma = folder_statistics(reference_dir, self.feat_model, self.device)
            np.savez(stats_path, mu=mu, sigma=sigma)
        stats = np.load(stats_path)
        self.mu, self.sigma = stats["mu"], stats["sigma"]

    def __call__(self, model, workdir):
        """Return the FID of ``model.sample()`` against the reference statistics."""
        workdir = Path(workdir)
        shutil.rmtree(workdir, ignore_errors=True)
        generate_samples(model, workdir, self.categories, batch_size=64, device=self.device)
        mu, sigma = folder_statistics(workdir, self.feat_model, self.device)
        return float(fid.frechet_distance(mu, sigma, self.mu, self.sigma))
