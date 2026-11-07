"""Sample images from a teacher snapshot or reconstruction.

    python -m teacher.sample --net runs/main_E112/phema/phema-0153600000-0.100.pt --out grid.png
    python -m teacher.sample --net MAIN.pt --guide-net GUIDE.pt --guidance 1.7 --out ag.png     # autoguidance
    python -m teacher.sample --net MAIN.pt --cfg 1.2 --out cfg.png                               # CFG with the null class
    python -m teacher.sample --net MAIN.pt --per-class 20 --out-dir samples/                     # evaluate.py-style PNGs
"""

import argparse
import os

import torch

from . import data as data_mod
from . import phema_tools
from . import sampler as sampler_mod

def load_guided(args, device):
    net, _ = phema_tools.load_network(args.net, device)
    gnet, guide, w = None, 'none', 1.0
    if args.guide_net:
        gnet, _ = phema_tools.load_network(args.guide_net, device)
        guide, w = 'autoguidance', args.guidance
    elif args.cfg is not None:
        guide, w = 'cfg', args.cfg
    return net, gnet, guide, w

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--net', required=True)
    p.add_argument('--guide-net', default=None, help='Autoguidance: the guiding network (e.g. an early reconstruction)')
    p.add_argument('--guidance', type=float, default=1.0, help='Autoguidance weight w')
    p.add_argument('--cfg', type=float, default=None, help='CFG weight w with the learned null class')
    p.add_argument('--classes', default='0-150', help="e.g. '0-150' or '3,25,150'")
    p.add_argument('--per-class', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--out', default=None, help='Grid PNG')
    p.add_argument('--out-dir', default=None, help='Also write evaluate.py-style PNGs here')
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    net, gnet, guide, w = load_guided(args, device)
    if '-' in args.classes:
        a, b = args.classes.split('-')
        classes = list(range(int(a), int(b) + 1))
    else:
        classes = [int(c) for c in args.classes.split(',')]
    classes = torch.tensor(classes).repeat_interleave(args.per_class)
    seeds = sampler_mod.seeds_for(args.seed, len(classes))
    x = sampler_mod.sample(net, classes, seeds, guidance=w, guide=guide, gnet=gnet, batch_size=args.batch, device=device,
                           sampler_kwargs=dict(num_steps=args.steps))
    if args.out:
        sampler_mod.save_grid(x, args.out, nrow=16)
        print(f'wrote {args.out}')
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        counts = {}
        for img, c in zip(x, classes.tolist()):
            i = counts.get(c, 0)
            counts[c] = i + 1
            data_mod.export_png(img, os.path.join(args.out_dir, f'{c:0>4}_{i:0>2}.png'))
        print(f'wrote {len(x)} PNGs to {args.out_dir}')

if __name__ == '__main__':
    main()
