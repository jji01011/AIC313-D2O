import glob
import os
import re

import torch

from . import networks, official
from .utils import to_builtin

SNAPSHOT_FORMAT = 'fastgen-teacher-ema-v1'
_NAME_RE = re.compile(r'(.*)-(\d+)-(\d+\.\d+)\.pt')

def snapshot_name(nimg, std, prefix='ema'):
    return f'{prefix}-{int(nimg):010d}-{float(std):.3f}.pt'

def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)

@torch.no_grad()
def network_state(net, trainable_dtype=torch.float16):
    trainable = {name for name, p in net.named_parameters() if p.requires_grad}
    sd = {}
    for k, v in net.state_dict().items():
        v = v.detach().to('cpu')
        if v.is_floating_point():
            v = v.to(trainable_dtype if k in trainable else torch.float32)
        sd[k] = v.contiguous().clone()
    return sd, sorted(trainable)

def save_snapshot(path, net, nimg, std, network_cfg, meta=None, trainable_dtype=torch.float16):
    sd, trainable = network_state(net, trainable_dtype)
    _atomic_save(dict(format=SNAPSHOT_FORMAT, nimg=int(nimg), std=float(std), network=to_builtin(network_cfg),
                      trainable=trainable, meta=to_builtin(meta or {}), state_dict=sd), path)
    return path

def read_header(path):
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    m = _NAME_RE.fullmatch(os.path.basename(path))
    return dict(path=path, nimg=int(data['nimg']), std=float(data['std']), prefix=m.group(1) if m else '', network=data['network'])

def list_snapshots(dirs, prefix='ema', stds=None, max_nimg=None):
    if isinstance(dirs, str):
        dirs = [dirs]
    found = {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, f'{prefix}-*-*.pt'))):
            if not _NAME_RE.fullmatch(os.path.basename(path)):
                continue
            h = read_header(path)
            if stds is not None and not any(abs(h['std'] - s) < 1e-9 for s in stds):
                continue
            if max_nimg is not None and h['nimg'] > max_nimg:
                continue
            found.setdefault((h['nimg'], round(h['std'], 6)), h)
    return [found[k] for k in sorted(found)]

def load_network(path, device='cpu'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    net = networks.build_network(data['network'])
    sd = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in data['state_dict'].items()}
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval().requires_grad_(False), data

@torch.no_grad()
def reconstruct(snapshots, out_std, out_nimg=None, verbose=True):
    if not snapshots:
        raise ValueError('No input snapshots')
    if out_nimg is None:
        out_nimg = max(s['nimg'] for s in snapshots)
    elif not any(out_nimg == s['nimg'] for s in snapshots):
        raise ValueError('Reconstruction time must match one of the input snapshots')
    ins = [s for s in snapshots if 0 < s['nimg'] <= out_nimg]
    out_std = [float(out_std)] if isinstance(out_std, (int, float)) else sorted(set(float(s) for s in out_std))
    coefs = official.solve_posthoc_coefficients([s['nimg'] for s in ins], [s['std'] for s in ins], out_nimg, out_std)
    outs = [None] * len(out_std)
    network_cfg, trainable = None, set()
    for i, s in enumerate(ins):
        data = torch.load(s['path'], map_location='cpu', weights_only=True)
        network_cfg = data['network']
        trainable = set(data['trainable'])
        sd = data['state_dict']
        for j in range(len(out_std)):
            if outs[j] is None:
                outs[j] = {k: (torch.zeros_like(v, dtype=torch.float32) if k in trainable else v.clone()) for k, v in sd.items()}
            for k in trainable:
                outs[j][k] += sd[k].to(torch.float32) * float(coefs[i, j])
            for k, v in sd.items():
                if k not in trainable:
                    outs[j][k].copy_(v)
        if verbose:
            print(f'\r  reconstruction: {i + 1}/{len(ins)} snapshots', end='', flush=True)
        del data, sd
    if verbose:
        print()
    return [dict(state_dict=outs[j], nimg=int(out_nimg), std=out_std[j], network=network_cfg, trainable=sorted(trainable),
                 coefs=[float(c) for c in coefs[:, j]], inputs=[(s['nimg'], s['std']) for s in ins]) for j in range(len(out_std))]

def save_reconstruction(rec, out_dir, prefix='phema'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, snapshot_name(rec['nimg'], rec['std'], prefix))
    _atomic_save(dict(format=SNAPSHOT_FORMAT, nimg=int(rec['nimg']), std=float(rec['std']), network=to_builtin(rec['network']),
                      trainable=list(rec['trainable']), meta=to_builtin(dict(kind='posthoc-reconstruction', inputs=rec['inputs'], coefs=rec['coefs'])),
                      state_dict=rec['state_dict']), path)
    return path

def load_reconstruction_into(net, rec):
    sd = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in rec['state_dict'].items()}
    net.load_state_dict(sd, strict=True)
    return net
