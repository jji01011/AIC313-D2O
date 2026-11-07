import os
import shutil
import tempfile
import time

import numpy as np
import torch

from . import data as data_mod
from . import sampler as sampler_mod

def _cleanfid():
    from cleanfid import fid as cfid
    return cfid

class FeatureExtractor:
    def __init__(self, device, num_workers=4, batch_size=64, model=None):
        self.cfid = _cleanfid()
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model = model if model is not None else self.cfid.build_feature_extractor('clean', self.device)

    def folder(self, folder, cache=None):
        if cache is not None and os.path.isfile(cache):
            return np.load(cache)['feats']
        feats = self.cfid.get_folder_features(folder, self.model, num_workers=self.num_workers, batch_size=self.batch_size,
                                              device=self.device, mode='clean', verbose=False)
        if cache is not None:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache + '.tmp.npz', feats=feats)
            os.replace(cache + '.tmp.npz', cache)
        return feats

def fid_from_feats(feats1, feats2):
    cfid = _cleanfid()
    mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return float(cfid.frechet_distance(mu1, sigma1, mu2, sigma2))

def kid_from_feats(feats_gen, feats_ref, seed=0):
    cfid = _cleanfid()
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return float(cfid.kernel_distance(feats_gen, feats_ref))
    finally:
        np.random.set_state(state)

def brightness_summary(images, bins=20):
    m = images.to(torch.float32).mean(dim=(1, 2, 3))
    hist = torch.histc(m, bins=bins, min=-1, max=1)
    return dict(mean=float(m.mean()), std=float(m.std()), hist=[int(h) for h in hist])

class ProxyFID:

    def __init__(self, prep, device, extractor=None, num_workers=4, feature_batch=64, work_dir=None):
        self.prep = prep
        self.device = torch.device(device)
        self.extractor = extractor or FeatureExtractor(device, num_workers=num_workers, batch_size=feature_batch)
        self.work_dir = work_dir
        self._ref_cache = {}
        self.num_classes = int(prep['stats']['num_classes'])
        self.train_brightness = brightness_summary(data_mod.u8_to_float(prep['images'][prep['splits']['main_train']]))

    def reference_features(self, name):
        if name not in self._ref_cache:
            prep_dir = self.prep['prep_dir']
            folder = data_mod.reference_dir(prep_dir, name)
            expected = len(data_mod.reference_indices(self.prep['splits'], name))
            if not os.path.isdir(folder) or len([f for f in os.listdir(folder) if f.endswith('.png')]) != expected:
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
                data_mod.export_reference(self.prep['images'], data_mod.reference_indices(self.prep['splits'], name), folder)
            cache = os.path.join(data_mod.prepared_paths(prep_dir)['features'], f'{name}.npz')
            self._ref_cache[name] = self.extractor.folder(folder, cache=cache)
        return self._ref_cache[name]

    def generate(self, net, out_dir, seed, per_class=data_mod.SAMPLES_PER_CATEGORY, guidance=1.0, guide='none', gnet=None,
                 batch_size=256, sampler_kwargs=None):
        classes = data_mod.category_order(self.num_classes, per_class)
        seeds = sampler_mod.seeds_for(seed, len(classes))
        images = sampler_mod.sample(net, classes, seeds, guidance=guidance, guide=guide, gnet=gnet, batch_size=batch_size,
                                    device=self.device, sampler_kwargs=sampler_kwargs)
        os.makedirs(out_dir, exist_ok=True)
        counts = {}
        for image, category in zip(images, classes.tolist()):
            idx = counts.get(category, 0)
            counts[category] = idx + 1
            data_mod.export_png(image, os.path.join(out_dir, f'{category:0>4}_{idx:0>2}.png'))
        return images, classes

    def evaluate(self, net, refs, seed=0, per_class=data_mod.SAMPLES_PER_CATEGORY, guidance=1.0, guide='none', gnet=None,
                 batch_size=256, sampler_kwargs=None, keep_dir=None, grid_path=None):
        t0 = time.time()
        tmp = keep_dir or tempfile.mkdtemp(prefix='fid-', dir=self.work_dir)
        try:
            images, classes = self.generate(net, tmp, seed, per_class=per_class, guidance=guidance, guide=guide, gnet=gnet,
                                            batch_size=batch_size, sampler_kwargs=sampler_kwargs)
            t1 = time.time()
            feats = self.extractor.folder(tmp)
        finally:
            if keep_dir is None:
                shutil.rmtree(tmp, ignore_errors=True)
        if grid_path is not None:
            first = torch.stack([images[(classes == c).nonzero()[0, 0]] for c in range(self.num_classes)])
            sampler_mod.save_grid(first, grid_path, nrow=16)
        per_ref = {}
        for name in refs:
            ref = self.reference_features(name)
            per_ref[name] = dict(fid=fid_from_feats(feats, ref), kid=kid_from_feats(feats, ref, seed=seed))
        return dict(fid=float(np.mean([v['fid'] for v in per_ref.values()])), kid=float(np.mean([v['kid'] for v in per_ref.values()])),
                    per_ref=per_ref, seed=int(seed), n=int(len(images)), brightness=brightness_summary(images),
                    train_brightness=self.train_brightness, sample_sec=t1 - t0, total_sec=time.time() - t0)

def floors(prep, extractor, pairs=None):
    prep_dir = prep['prep_dir']
    out = []
    pairs = range(len(prep['splits']['floor_pairs'])) if pairs is None else pairs
    for k in pairs:
        feats = []
        for side in 'ab':
            name = f'floor{k}{side}'
            folder = data_mod.reference_dir(prep_dir, name)
            if not os.path.isdir(folder):
                data_mod.export_reference(prep['images'], data_mod.reference_indices(prep['splits'], name), folder)
            feats.append(extractor.folder(folder, cache=os.path.join(data_mod.prepared_paths(prep_dir)['features'], f'{name}.npz')))
        out.append(dict(pair=k, fid=fid_from_feats(feats[0], feats[1]), kid=kid_from_feats(feats[0], feats[1])))
        if k == 0:
            self_fid = fid_from_feats(feats[0], feats[0])
    fids = [o['fid'] for o in out]
    return dict(pairs=out, fid_mean=float(np.mean(fids)), fid_sd=float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0,
                self_fid=float(self_fid))
