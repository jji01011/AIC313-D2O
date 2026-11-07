"""Train-split FID proxy for one network (and optional guide), over several seeds.

    python -m teacher.eval_fid --prep-dir data/teacher_prep --net runs/main_E112/phema/phema-...-0.100.pt --seeds 0,1,2
    python -m teacher.eval_fid ... --guide-net GUIDE.pt --guidance 1.7            # autoguidance
    python -m teacher.eval_fid ... --cfg 1.2                                       # CFG

References default to the three class-balanced training subsets (train0, train1, train2);
use --refs holdout for pilot runs. The validation split is never used.
"""

import argparse
import json
import os

import numpy as np
import torch

from . import data as data_mod
from . import fid as fid_mod
from .sample import load_guided
from .utils import now_iso

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True)
    p.add_argument('--net', required=True)
    p.add_argument('--guide-net', default=None)
    p.add_argument('--guidance', type=float, default=1.0)
    p.add_argument('--cfg', type=float, default=None)
    p.add_argument('--refs', default='train0,train1,train2')
    p.add_argument('--seeds', default='0')
    p.add_argument('--per-class', type=int, default=20)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--S-churn', type=float, default=0.0, help='EDM stochastic sampling (diagnostic only)')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--out', default=None, help='Append the result as a JSON line to this file')
    p.add_argument('--keep-images', default=None, help='Keep the seed-0 PNGs in this folder')
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    prep = data_mod.load_prepared(args.prep_dir)
    net, gnet, guide, w = load_guided(args, device)
    proxy = fid_mod.ProxyFID(prep, device, num_workers=args.num_workers)
    refs = [r.strip() for r in args.refs.split(',') if r.strip()]
    runs = []
    for i, seed in enumerate(int(s) for s in args.seeds.split(',')):
        r = proxy.evaluate(net, refs, seed=seed, per_class=args.per_class, guidance=w, guide=guide, gnet=gnet, batch_size=args.batch,
                           sampler_kwargs=dict(num_steps=args.steps, S_churn=args.S_churn), keep_dir=args.keep_images if i == 0 else None)
        per_ref = ', '.join('%s %.3f' % (k, v['fid']) for k, v in r['per_ref'].items())
        print(f"seed {seed}: FID {r['fid']:.3f} (per reference: {per_ref}), KID {r['kid']:.5f}")
        runs.append(r)
    fids = [r['fid'] for r in runs]
    res = dict(created=now_iso(), net=os.path.abspath(args.net), guide_net=args.guide_net and os.path.abspath(args.guide_net), guide=guide,
               guidance=w, steps=args.steps, S_churn=args.S_churn, refs=refs, fids=fids, mean=float(np.mean(fids)),
               sd=float(np.std(fids, ddof=1)) if len(fids) > 1 else None, runs=runs)
    print(f"FID {res['mean']:.3f}" + (f" +- {res['sd']:.3f}" if res['sd'] is not None else '') + f' over {len(fids)} seed(s)')
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps(res) + '\n')

if __name__ == '__main__':
    main()
