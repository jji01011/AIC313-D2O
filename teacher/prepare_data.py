"""Day-0 data preparation (TEACHER_DESIGN.md sections 5.1 and 10.3).

    python -m teacher.prepare_data --out data/teacher_prep [--floors]

Downloads the dataset through the course code if needed, caches the training split as uint8,
draws the canary C, pilot holdout H, guard probe, FID reference subsets and floor pairs, computes
the mean image and sigma_data on the pilot training set, exports the reference PNGs, and with
--floors computes the train-versus-train FID floors. Only train_split.txt is read.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from . import data as data_mod
from . import presets
from .utils import device_info, now_iso, write_json

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='./data/pokemon-generation-one-22k')
    p.add_argument('--split-dir', default='./data/pokemon-generation-one-22k')
    p.add_argument('--out', default='./data/teacher_prep')
    p.add_argument('--seed', type=int, default=0, help='Seed for the splits (fixed once, never tuned)')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--floors', action='store_true', help='Also compute train-vs-train FID floors (needs the Inception weights)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-export', action='store_true', help='Do not export reference PNGs now (they are exported on first use)')
    args = p.parse_args(argv)

    t0 = time.time()
    paths = data_mod.prepared_paths(args.out)
    os.makedirs(args.out, exist_ok=True)
    if os.path.isfile(paths['stats']):
        raise SystemExit(f'{args.out} is already prepared; use a new --out to re-prepare')

    print('Caching the training split through dataset.PokemonDataset ...')
    cache = data_mod.build_cache(args.data_root, args.split_dir, num_workers=args.num_workers)
    images, labels = cache['images'], cache['labels']
    num_classes = len(cache['category_to_id'])
    counts = np.bincount(labels.numpy(), minlength=num_classes)
    print(f'  {len(images)} images, {num_classes} classes, per class min {counts.min()} / median {int(np.median(counts))} / max {counts.max()}')

    splits = data_mod.draw_splits(labels.numpy(), num_classes, seed=args.seed)
    pilot = torch.as_tensor(splits['pilot_train'])
    mean_image, sigma_data = data_mod.mean_and_sigma_data(images[pilot])
    stats = dict(format=data_mod.PREP_FORMAT, created=now_iso(), seed=args.seed, num_classes=num_classes, num_images=int(len(images)),
                 n_canary=len(splits['canary']), n_holdout=len(splits['holdout']), n_pilot_train=len(splits['pilot_train']),
                 n_main_train=len(splits['main_train']), n_probe=len(splits['probe']),
                 sigma_data=sigma_data, P_mean_edm2=presets.p_mean_for('edm2', sigma_data), P_mean_edm=presets.p_mean_for('edm', sigma_data),
                 mean_image_file='mean_image.pt', mean_image_pixel_mean=float(mean_image.mean()),
                 mean_image_channel_mean=[float(v) for v in mean_image.mean(dim=(1, 2))],
                 class_counts=[int(c) for c in counts], data_root=os.path.abspath(args.data_root), split_dir=os.path.abspath(args.split_dir),
                 category_to_id=cache['category_to_id'], note='mean_image and sigma_data are computed on the pilot training set (all train images minus H) and frozen.')
    torch.save(dict(images=images, labels=labels, paths=cache['paths']), paths['cache'])
    torch.save(mean_image, paths['mean'])
    write_json(paths['splits'], splits)
    write_json(paths['stats'], stats)
    print(f"  canary {stats['n_canary']}, holdout {stats['n_holdout']}, pilot train {stats['n_pilot_train']}, main train {stats['n_main_train']}")
    print(f'  sigma_data {sigma_data:.5f}; P_mean (EDM2) {stats["P_mean_edm2"]:.4f}; mean pixel {stats["mean_image_pixel_mean"]:+.4f}')

    if not args.skip_export:
        print('Exporting reference PNGs (holdout, train0..2) ...')
        for name in ['holdout', 'train0', 'train1', 'train2']:
            data_mod.export_reference(images, data_mod.reference_indices(splits, name), data_mod.reference_dir(args.out, name))

    if args.floors:
        from . import fid as fid_mod
        device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
        extractor = fid_mod.FeatureExtractor(device, num_workers=args.num_workers)
        prep = data_mod.load_prepared(args.out)
        print('Computing train-versus-train FID floors (5 disjoint pairs) ...')
        floors = fid_mod.floors(prep, extractor)
        for name in ['holdout', 'train0', 'train1', 'train2']:
            fid_mod.ProxyFID(prep, device, extractor=extractor).reference_features(name)
        floors.update(device=device_info(device), created=now_iso())
        write_json(os.path.join(args.out, 'floors.json'), floors)
        print(f"  floor FID {floors['fid_mean']:.3f} +- {floors['fid_sd']:.3f} over {len(floors['pairs'])} pairs; "
              f"a subset against itself {floors['self_fid']:.2e}")
    print(f'Done in {time.time() - t0:.0f} s -> {args.out}')
    print(json.dumps({k: stats[k] for k in ('num_images', 'n_pilot_train', 'n_main_train', 'sigma_data')}, indent=2))

if __name__ == '__main__':
    main()
