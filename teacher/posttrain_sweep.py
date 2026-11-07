"""Post-training sweeps on the train-split FID proxy (TEACHER_DESIGN.md section 8, steps 1-3) and hand-off.

    python -m teacher.posttrain_sweep --prep-dir data/teacher_prep --run-dir runs/main_E112 --stage all

Stages (each reads earlier results from <out>/sweep.jsonl, so they can be run separately):
  ema    sigma_rel in {0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20} at the final time, unguided, seed 0;
         the top 2 at seeds 1 and 2.                                   -> teacher_unguided
  cfg    w in {1.05, 1.1, 1.2, 1.3, 1.4} x main sigma_rel {0.03, 0.05}, seed 0; top 2 at 3 seeds.
  ag     snapshot autoguidance: guide end time {T/32, T/16, T/8, T/4} (the latest snapshot at or
         before each) x guide sigma_rel {0.10, 0.175} x w {1.3, 1.6, 1.9, 2.2} at main sigma_rel 0.05,
         seed 0; then main sigma_rel {0.03, 0.045, 0.06, 0.075} x w +-0.1 around the best with its
         guide; the top 3 at 3 seeds.                                  -> teacher_ag_main, guide, w
  export the hand-off files of design 5.11 into <out>/handoff.
Ties (differences under 2 seed SDs) go to the simpler or more diverse option: the longer EMA,
the weaker guidance. Neon and DDO (design steps 4 and 6) are not implemented here.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch

from . import data as data_mod
from . import fid as fid_mod
from . import phema_tools, presets
from .utils import code_fingerprint, now_iso, read_jsonl, write_json

EMA_GRID = [0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20]
CFG_W = [1.05, 1.1, 1.2, 1.3, 1.4]
CFG_STD = [0.03, 0.05]
AG_FRACTIONS = [1 / 32, 1 / 16, 1 / 8, 1 / 4]
AG_GUIDE_STD = [0.10, 0.175]
AG_W = [1.3, 1.6, 1.9, 2.2]
AG_MAIN_STD = 0.05
AG_REFINE_STD = [0.03, 0.045, 0.06, 0.075]

class Sweeper:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
        self.prep = data_mod.load_prepared(args.prep_dir)
        self.proxy = fid_mod.ProxyFID(self.prep, self.device, num_workers=args.num_workers)
        self.out = args.out or os.path.join(args.run_dir, 'posttrain')
        self.rec_dir = os.path.join(self.out, 'phema')
        os.makedirs(self.rec_dir, exist_ok=True)
        self.snaps = phema_tools.list_snapshots(args.run_dir)
        if not self.snaps:
            raise SystemExit(f'No snapshots in {args.run_dir}')
        self.times = sorted({s['nimg'] for s in self.snaps})
        self.T = int(args.nimg) if args.nimg else self.times[-1]
        if self.T not in self.times:
            raise SystemExit(f'--nimg must match a snapshot time: {self.times}')
        self.refs = [r.strip() for r in args.refs.split(',') if r.strip()]
        self.log_path = os.path.join(self.out, 'sweep.jsonl')
        self._nets = {}

    def net(self, nimg, std):
        key = (int(nimg), round(float(std), 6))
        if key in self._nets:
            return self._nets[key]
        path = os.path.join(self.rec_dir, phema_tools.snapshot_name(nimg, std, 'phema'))
        if not os.path.isfile(path):
            rec = phema_tools.reconstruct(self.snaps, [std], out_nimg=nimg)[0]
            phema_tools.save_reconstruction(rec, self.rec_dir)
        net, _ = phema_tools.load_network(path, self.device)
        if len(self._nets) >= 6:
            self._nets.pop(next(iter(self._nets)))
        self._nets[key] = net
        return net

    def guide_time(self, frac):
        target = self.T * frac
        cands = [t for t in self.times if t <= target]
        return cands[-1] if cands else self.times[0]

    def done(self):
        return read_jsonl(self.log_path)

    def score(self, stage, cfg, seed):
        key = dict(stage=stage, seed=seed, nimg=self.T, refs=self.refs, steps=self.args.steps, per_class=self.args.per_class, **cfg)
        for r in self.done():
            if all(r.get(k) == v for k, v in key.items()):
                return r
        net = self.net(self.T, cfg['main_std'])
        gnet = self.net(cfg['guide_nimg'], cfg['guide_std']) if cfg.get('guide') == 'autoguidance' else None
        res = self.proxy.evaluate(net, self.refs, seed=seed, per_class=self.args.per_class, guidance=cfg.get('w', 1.0),
                                  guide=cfg.get('guide', 'none'), gnet=gnet, batch_size=self.args.batch, sampler_kwargs=dict(num_steps=self.args.steps))
        row = dict(key, fid=res['fid'], kid=res['kid'], per_ref={k: v['fid'] for k, v in res['per_ref'].items()},
                   brightness=res['brightness']['mean'], sec=res['total_sec'], created=now_iso())
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        desc = ', '.join(f'{k}={v}' for k, v in cfg.items())
        print(f'  [{stage}] {desc} seed {seed}: FID {row["fid"]:.3f}')
        return row

    def confirm(self, stage, cfgs, top):
        first = [(c, self.score(stage, c, 0)['fid']) for c in cfgs]
        first.sort(key=lambda t: t[1])
        ranked = []
        for c, _ in first[:top]:
            fids = [self.score(stage, c, s)['fid'] for s in (0, 1, 2)]
            ranked.append(dict(cfg=c, fids=fids, mean=float(np.mean(fids)), sd=float(np.std(fids, ddof=1))))
        ranked.sort(key=lambda r: r['mean'])
        return ranked

    @staticmethod
    def tie_break(ranked, simpler_key):
        best = ranked[0]
        ties = [r for r in ranked if r['mean'] - best['mean'] < 2 * max(best['sd'], r['sd'])]
        return max(ties, key=simpler_key)

    def stage_ema(self):
        cfgs = [dict(main_std=s, guide='none', w=1.0, guide_nimg=None, guide_std=None) for s in self.args_list('ema_grid', EMA_GRID)]
        ranked = self.confirm('ema', cfgs, top=2)
        choice = self.tie_break(ranked, simpler_key=lambda r: r['cfg']['main_std'])
        print(f"EMA: best sigma_rel {choice['cfg']['main_std']} FID {choice['mean']:.3f} +- {choice['sd']:.3f}")
        return dict(ranked=ranked, choice=choice)

    def stage_cfg(self):
        cfgs = [dict(main_std=s, guide='cfg', w=w, guide_nimg=None, guide_std=None) for s in CFG_STD for w in CFG_W]
        ranked = self.confirm('cfg', cfgs, top=2)
        choice = self.tie_break(ranked, simpler_key=lambda r: -r['cfg']['w'])
        print(f"CFG: best w {choice['cfg']['w']} at sigma_rel {choice['cfg']['main_std']}: FID {choice['mean']:.3f} +- {choice['sd']:.3f}")
        return dict(ranked=ranked, choice=choice)

    def stage_ag(self):
        coarse = [dict(main_std=AG_MAIN_STD, guide='autoguidance', w=w, guide_nimg=self.guide_time(f), guide_std=gs)
                  for f in AG_FRACTIONS for gs in AG_GUIDE_STD for w in AG_W]
        first = sorted(((c, self.score('ag', c, 0)['fid']) for c in coarse), key=lambda t: t[1])
        best = first[0][0]
        refine = [dict(best, main_std=ms, w=round(best['w'] + dw, 4)) for ms in AG_REFINE_STD for dw in (-0.1, 0.0, 0.1)]
        cand = {json.dumps(c, sort_keys=True): c for c in [c for c, _ in first] + refine}
        ranked = self.confirm('ag', list(cand.values()), top=3)
        choice = self.tie_break(ranked, simpler_key=lambda r: -r['cfg']['w'])
        c = choice['cfg']
        print(f"Autoguidance: main sigma_rel {c['main_std']}, guide at {c['guide_nimg']} images sigma_rel {c['guide_std']}, w {c['w']}: "
              f"FID {choice['mean']:.3f} +- {choice['sd']:.3f}")
        return dict(ranked=ranked, choice=choice)

    def args_list(self, name, default):
        v = getattr(self.args, name, None)
        return [float(x) for x in v.split(',')] if v else default

    def export(self, summary):
        h = os.path.join(self.out, 'handoff')
        os.makedirs(h, exist_ok=True)
        def copy_rec(nimg, std, name):
            self.net(nimg, std)
            src = os.path.join(self.rec_dir, phema_tools.snapshot_name(nimg, std, 'phema'))
            shutil.copy2(src, os.path.join(h, name))
        torch.save(self.prep['mean_image'], os.path.join(h, 'mean_image.pt'))
        info = dict(created=now_iso(), run_dir=os.path.abspath(self.args.run_dir), T_nimg=self.T, refs=self.refs, steps=self.args.steps)
        if 'ema' in summary:
            c = summary['ema']['choice']
            copy_rec(self.T, c['cfg']['main_std'], 'teacher_unguided.pt')
            info['teacher_unguided'] = dict(std=c['cfg']['main_std'], fid=c['mean'], sd=c['sd'])
        if 'ag' in summary:
            c = summary['ag']['choice']
            copy_rec(self.T, c['cfg']['main_std'], 'teacher_ag_main.pt')
            copy_rec(c['cfg']['guide_nimg'], c['cfg']['guide_std'], 'guide.pt')
            info['autoguidance'] = dict(main_std=c['cfg']['main_std'], guide_nimg=c['cfg']['guide_nimg'], guide_std=c['cfg']['guide_std'],
                                        w=c['cfg']['w'], fid=c['mean'], sd=c['sd'])
        if 'cfg' in summary:
            c = summary['cfg']['choice']
            info['cfg'] = dict(main_std=c['cfg']['main_std'], w=c['cfg']['w'], fid=c['mean'], sd=c['sd'])
        net = self.net(self.T, summary['ema']['choice']['cfg']['main_std']) if 'ema' in summary else self.net(self.T, 0.10)
        stats = self.prep['stats']
        network = self.snaps[-1]['network']
        family = network['family']
        cfg = dict(network=network, sigma_data=stats['sigma_data'], P_mean=presets.p_mean_for(family, float(stats['sigma_data'])),
                   P_std=presets.RECIPES[family]['P_std'], null_class_index=int(net.num_classes),
                   augment_dim=net.unet.augment_dim if hasattr(net, 'unet') else 9, c_noise='log(sigma) / 4', mean_image='mean_image.pt',
                   label_dim_with_null=int(net.label_dim), sampler=dict(type='edm_heun', num_steps=32, sigma_min=0.002, sigma_max=80, rho=7),
                   output='clamp(x, -1, 1) + 1/255, clamp', results=info, code_sha256=code_fingerprint())
        write_json(os.path.join(h, 'teacher_config.json'), cfg)
        print(f'wrote hand-off files to {h}')

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--stage', default='all', choices=['ema', 'cfg', 'ag', 'export', 'all'])
    p.add_argument('--nimg', default=None, help='Final time T (images); default the latest snapshot')
    p.add_argument('--refs', default='train0,train1,train2')
    p.add_argument('--ema-grid', default=None, help='Override the sigma_rel grid of the ema stage')
    p.add_argument('--per-class', type=int, default=20)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(argv)

    sw = Sweeper(args)
    summary_path = os.path.join(sw.out, 'summary.json')
    summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
    stages = ['ema', 'cfg', 'ag', 'export'] if args.stage == 'all' else [args.stage]
    for st in stages:
        if st == 'export':
            sw.export(summary)
            continue
        summary[st] = getattr(sw, f'stage_{st}')()
        write_json(summary_path, summary)
    print(f'summary: {summary_path}')

if __name__ == '__main__':
    main()
