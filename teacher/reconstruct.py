"""Post-hoc EMA reconstruction from a run's snapshots (EDM2 reconstruct_phema.py, TEACHER_DESIGN.md 5.7).

    python -m teacher.reconstruct --run-dir runs/main_E112 --std 0.05,0.07,0.10 [--nimg latest|<images>] [--out-dir DIR]

Writes phema-<nimg:010d>-<std>.pt files (fp32 state dicts that load with phema_tools.load_network).
"""

import argparse
import os

from . import phema_tools
from .utils import parse_nimg

def parse_stds(s):
    return sorted({float(x) for x in str(s).split(',') if x.strip()})

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, action='append', help='Run directory with ema-*.pt snapshots (repeatable)')
    p.add_argument('--std', required=True, help='Comma-separated sigma_rel values, e.g. 0.05,0.10')
    p.add_argument('--nimg', default='latest', help="Training time to reconstruct at: 'latest' or an image count matching a snapshot")
    p.add_argument('--in-std', default=None, help='Only use input snapshots with these sigma_rel values')
    p.add_argument('--out-dir', default=None, help='Default: <first run dir>/phema')
    args = p.parse_args(argv)

    snaps = phema_tools.list_snapshots(args.run_dir, stds=parse_stds(args.in_std) if args.in_std else None)
    if not snaps:
        raise SystemExit('No ema-*.pt snapshots found')
    out_nimg = None if args.nimg == 'latest' else parse_nimg(args.nimg)
    if out_nimg is not None and not any(s['nimg'] == out_nimg for s in snaps):
        times = sorted({s['nimg'] for s in snaps})
        raise SystemExit(f'No snapshot at {out_nimg} images. Available: {times}')
    out_dir = args.out_dir or os.path.join(args.run_dir[0], 'phema')
    recs = phema_tools.reconstruct(snaps, parse_stds(args.std), out_nimg=out_nimg)
    for rec in recs:
        path = phema_tools.save_reconstruction(rec, out_dir)
        print(f"{path}: sigma_rel {rec['std']:.3f} at {rec['nimg']} images from {len(rec['inputs'])} snapshots")

if __name__ == '__main__':
    main()
