import json
import os
import sys

import numpy as np
import torch

from . import official

if official.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, official.PROJECT_ROOT)

SAMPLES_PER_CATEGORY = 20
PREP_FORMAT = 'fastgen-teacher-data-v1'


def u8_to_float(images_u8):
    return images_u8.to(torch.float32).div(255).sub(0.5).div(0.5)

def float_to_u8(images):
    return torch.round((images.to(torch.float32) + 1) * 127.5).clamp(0, 255).to(torch.uint8)

def export_png(image, path):
    from torchvision.transforms.functional import to_pil_image
    image = ((image.float().cpu() + 1) / 2).clamp(0, 1)
    to_pil_image(image).convert('RGB').save(path)

def category_order(num_classes, per_class=SAMPLES_PER_CATEGORY):
    return torch.arange(num_classes, dtype=torch.long).repeat_interleave(per_class)


def load_course_dataset(data_root, split_dir):
    from dataset import PokemonDataset, load_or_save_category_mapping
    category_to_id = load_or_save_category_mapping(split_dir, 'train_split.txt', 'val_split.txt')
    ds = PokemonDataset(data_root, os.path.join(split_dir, 'train_split.txt'), return_category=True, category_to_id=category_to_id)
    return ds, category_to_id

def build_cache(data_root, split_dir, num_workers=4, batch_size=256, verbose=True):
    ds, category_to_id = load_course_dataset(data_root, split_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    images, labels = [], []
    max_err = 0.0
    for x, y in loader:
        u8 = float_to_u8(x)
        max_err = max(max_err, float((u8_to_float(u8) - x).abs().max()))
        images.append(u8)
        labels.append(y.to(torch.int64))
        if verbose and (len(images) % 20 == 0 or sum(len(i) for i in images) == len(ds)):
            print(f'  cached {sum(len(i) for i in images)}/{len(ds)} images', flush=True)
    if max_err != 0.0:
        raise RuntimeError(f'uint8 round trip is not exact (max error {max_err}); the course transform changed?')
    return dict(images=torch.cat(images), labels=torch.cat(labels), paths=list(ds.paths), category_to_id=category_to_id)


def draw_splits(labels, num_classes, seed=0, canary_per_class=5, holdout_per_class=20, probe_per_class=5,
                num_ref_subsets=3, ref_per_class=20, num_floor_pairs=5):
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    by_class = [np.flatnonzero(labels == c) for c in range(num_classes)]
    canary, holdout = [], []
    for idx in by_class:
        if len(idx) < holdout_per_class + max(probe_per_class, 1):
            raise ValueError(f'class with {len(idx)} images is too small for the holdout')
        perm = rng.permutation(idx)
        holdout.append(perm[:holdout_per_class])
        canary.append(perm[:canary_per_class])
    canary = np.sort(np.concatenate(canary))
    holdout = np.sort(np.concatenate(holdout))
    everything = np.arange(len(labels))
    pilot_train = np.setdiff1d(everything, holdout)
    main_train = np.setdiff1d(everything, canary)

    def balanced(pool, per_class, rs, exclude=()):
        out = []
        for c in range(num_classes):
            idx = np.setdiff1d(pool[labels[pool] == c], exclude)
            if len(idx) < per_class:
                raise ValueError(f'class {c} has {len(idx)} images, fewer than {per_class}')
            out.append(rs.choice(idx, per_class, replace=False))
        return np.sort(np.concatenate(out))

    probe = balanced(pilot_train, probe_per_class, np.random.RandomState(seed + 1))
    ref_subsets = [balanced(main_train, ref_per_class, np.random.RandomState(seed + 100 + k)) for k in range(num_ref_subsets)]
    floor_pairs = []
    for k in range(num_floor_pairs):
        rs = np.random.RandomState(seed + 200 + k)
        a = balanced(main_train, ref_per_class, rs)
        b = balanced(main_train, ref_per_class, rs, exclude=a)
        floor_pairs.append([a, b])
    as_list = lambda a: [int(i) for i in a]
    return dict(seed=int(seed), num_images=int(len(labels)),
                canary=as_list(canary), holdout=as_list(holdout), pilot_train=as_list(pilot_train), main_train=as_list(main_train),
                probe=as_list(probe), ref_subsets=[as_list(r) for r in ref_subsets],
                floor_pairs=[[as_list(a), as_list(b)] for a, b in floor_pairs])


def mean_and_sigma_data(images_u8, chunk=2048):
    n = images_u8.shape[0]
    total = torch.zeros(images_u8.shape[1:], dtype=torch.float64)
    for s in range(0, n, chunk):
        total += u8_to_float(images_u8[s : s + chunk]).to(torch.float64).sum(dim=0)
    mean = total / n
    mean = (mean + mean.flip(-1)) / 2
    sq = 0.0
    for s in range(0, n, chunk):
        d = u8_to_float(images_u8[s : s + chunk]).to(torch.float64) - mean
        sq += float((d ** 2).sum())
    sigma_data = (sq / (n * mean.numel())) ** 0.5
    return mean.to(torch.float32), float(sigma_data)


def prepared_paths(prep_dir):
    return dict(cache=os.path.join(prep_dir, 'train_cache.pt'), splits=os.path.join(prep_dir, 'splits.json'),
                stats=os.path.join(prep_dir, 'stats.json'), mean=os.path.join(prep_dir, 'mean_image.pt'),
                refs=os.path.join(prep_dir, 'refs'), features=os.path.join(prep_dir, 'ref_features'))

def load_prepared(prep_dir):
    p = prepared_paths(prep_dir)
    cache = torch.load(p['cache'], map_location='cpu', weights_only=True)
    with open(p['splits']) as f:
        splits = json.load(f)
    with open(p['stats']) as f:
        stats = json.load(f)
    mean_image = torch.load(p['mean'], map_location='cpu', weights_only=True)
    assert stats['format'] == PREP_FORMAT
    return dict(images=cache['images'], labels=cache['labels'], paths=cache['paths'], splits=splits, stats=stats,
                mean_image=mean_image, dirs=p, prep_dir=prep_dir)

def reference_dir(prep_dir, name):
    return os.path.join(prepared_paths(prep_dir)['refs'], name)

def reference_indices(splits, name):
    if name == 'holdout':
        return splits['holdout']
    if name.startswith('train'):
        return splits['ref_subsets'][int(name[len('train'):])]
    if name.startswith('floor'):
        k, side = int(name[len('floor'):-1]), name[-1]
        return splits['floor_pairs'][k][0 if side == 'a' else 1]
    raise ValueError(name)

def export_reference(images_u8, indices, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for count, i in enumerate(indices):
        export_png(u8_to_float(images_u8[i]), os.path.join(out_dir, f'{count:04d}.png'))
    return out_dir


class TrainStream:

    def __init__(self, images_u8, labels, seed, device):
        self.images = images_u8.to(device)
        self.labels = labels.to(device)
        self.n = int(images_u8.shape[0])
        self.seed = int(seed)
        self.device = device
        self._iter = None
        self._pos = None

    def _indices(self, pos, count):
        if self._iter is None or self._pos != pos:
            self._iter = iter(official.edm2_misc.InfiniteSampler(range(self.n), seed=self.seed, start_idx=pos))
            self._pos = pos
        idx = [next(self._iter) for _ in range(count)]
        self._pos += count
        return torch.as_tensor(idx, dtype=torch.long)

    def get(self, pos, count):
        idx = self._indices(pos, count).to(self.device)
        return u8_to_float(self.images[idx]), self.labels[idx]
