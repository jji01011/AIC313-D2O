from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PokemonDataModule


def load_train_tensors(data_root, split_dir, cache_path="./data/cache/train_64x64.pt", num_workers=8):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=True)
        return cache["images"], cache["labels"]

    module = PokemonDataModule(data_root=data_root, split_dir=split_dir, return_category=True)
    loader = DataLoader(module.train_dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    images, labels = [], []
    for batch_images, batch_labels in loader:
        images.append(((batch_images + 1) * 127.5).round().clamp(0, 255).to(torch.uint8))
        labels.append(batch_labels)
    images, labels = torch.cat(images), torch.cat(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, cache_path)
    return images, labels


class InfiniteBatches:

    def __init__(self, images, labels, batch_size, device, xflip=False, seed=0, balanced=False):
        self.images = images.to(device)
        self.labels = labels.to(device)
        self.batch_size = batch_size
        self.xflip = xflip
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long, device=device)
        self.balanced = balanced
        if balanced:
            self.by_class = torch.argsort(self.labels, stable=True)
            self.class_counts = torch.bincount(self.labels)
            assert bool((self.class_counts > 0).all()), "a category without images"
            self.class_starts = torch.cumsum(self.class_counts, 0) - self.class_counts

    def __iter__(self):
        return self

    def __next__(self):
        if self.balanced:
            device = self.images.device
            classes = torch.randint(len(self.class_counts), [self.batch_size], device=device, generator=self.generator)
            offset = (torch.rand(self.batch_size, device=device, generator=self.generator) * self.class_counts[classes]).long()
            index = self.by_class[self.class_starts[classes] + offset.clamp(max=self.class_counts[classes] - 1)]
        else:
            while len(self.order) < self.batch_size:
                perm = torch.randperm(len(self.images), device=self.images.device, generator=self.generator)
                self.order = torch.cat([self.order, perm])
            index, self.order = self.order[: self.batch_size], self.order[self.batch_size :]
        images = self.images[index].to(torch.float32) / 127.5 - 1
        if self.xflip:
            flip = torch.rand(self.batch_size, device=images.device, generator=self.generator) < 0.5
            images = torch.where(flip[:, None, None, None], images.flip(3), images)
        return images, self.labels[index]
