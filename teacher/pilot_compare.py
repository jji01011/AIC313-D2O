"""Score the pilot arms and apply the decision rules (TEACHER_DESIGN.md section 10.4).

    python -m teacher.pilot_compare --prep-dir data/teacher_prep --runs runs/pilot_E112_lr1x,runs/pilot_E112_lr2x,runs/pilot_DDPMpp128

For each arm, at its final snapshot and at the slowest arm's image count: proxy FID of the tracked
sigma_rel 0.10 EMA against the pilot holdout H (3 seeds), and the lambda-weighted denoising loss
on H over the sigma grid of design 5.10. From the training logs: img/s, loss spikes and NaN/Inf
zeroings. Then:
  1. learning rate: per width, the EDM2 arm with the lower held-out denoising loss, unless it shows
     loss spikes or repeated NaN zeroing; within 1%, the lower learning rate;
  2. family: EDM2 stays unless DDPM++128 beats the best EDM2 arm on FID by more than 2 SD. Then one
     more EDM2 arm runs at 0.5 x alpha_ref (run_pilot.sh with HALF_LR_ARM=1) and this script runs
     again with it; if DDPM++128 still wins by more than 2 SD, it becomes the teacher;
  3. layout: another EDM2 layout arm (E128-1233, E144-1223) replaces E112 only if better by more
     than 2 SD; if several are, the one with the lowest FID.
Only training images (H) are used.
"""

import argparse
import json
import os

import numpy as np
import torch

from . import data as data_mod
from . import fid as fid_mod
from . import losses, phema_tools, presets
from .utils import now_iso, read_jsonl, write_json

def count_spikes(values, window=3, ratio=1.1):
    spikes = 0
    for i in range(window, len(values)):
        ref = min(values[i - window:i])
        if ref > 0 and values[i] > ratio * ref:
            spikes += 1
    return spikes

def run_info(run_dir):
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    stats = read_jsonl(os.path.join(run_dir, 'stats.jsonl'))
    ips = [s['img_per_sec'] for s in stats[1:]] or [s['img_per_sec'] for s in stats]
    spikes = count_spikes([s['wmse'] for s in stats])
    nonfinite_steps = int(sum(s.get('nonfinite_steps', 0) for s in stats))
    steps = stats[-1]['step'] if stats else 0
    return dict(run_dir=run_dir, preset=cfg['network']['preset'], family=cfg['network']['family'], ref_lr=cfg['recipe']['ref_lr'],
                img_per_sec=float(np.median(ips)) if ips else None, loss_spikes=spikes, nonfinite_steps=nonfinite_steps, steps=steps,
                unstable=bool(spikes > 0 or nonfinite_steps > max(3, 0.001 * max(steps, 1))))

def evaluate_at(prep, proxy, run_dir, nimg, seeds, device, per_class, sampler_kwargs, batch_size, heldout_max=None, std=0.100, cache=None):
    snaps = [s for s in phema_tools.list_snapshots(run_dir, stds=[std]) if s['nimg'] <= nimg]
    if not snaps:
        return None
    snap = snaps[-1]
    if cache is not None and snap['path'] in cache:
        return cache[snap['path']]
    net, _ = phema_tools.load_network(snap['path'], device)
    fids = [proxy.evaluate(net, ['holdout'], seed=s, per_class=per_class, sampler_kwargs=sampler_kwargs, batch_size=batch_size)['fid'] for s in seeds]
    h = prep['splits']['holdout'][:heldout_max]
    x = data_mod.u8_to_float(prep['images'][h]).to(device)
    y = net.one_hot(prep['labels'][h].to(device))
    held = losses.fixed_sigma_wmse(net, x, y, float(prep['stats']['sigma_data']), seed=7)
    res = dict(nimg=snap['nimg'], std=std, fids=fids, fid=float(np.mean(fids)), sd=float(np.std(fids, ddof=1)) if len(fids) > 1 else 0.0,
               heldout_wmse=held, heldout_wmse_mean=float(np.mean(list(held.values()))))
    if cache is not None:
        cache[snap['path']] = res
    return res

def pooled_sd(a, b):
    return float(np.sqrt((a['sd'] ** 2 + b['sd'] ** 2) / 2))

def decide(arms):
    out = dict(notes=[])
    edm2 = [a for a in arms if a['family'] == 'edm2']
    best_by_preset = {}
    for preset in sorted({a['preset'] for a in edm2}):
        cands = sorted([a for a in edm2 if a['preset'] == preset], key=lambda a: a['ref_lr'])
        stable = [a for a in cands if not a['unstable']] or cands
        best = min(stable, key=lambda a: a['final']['heldout_wmse_mean'])
        low = stable[0]
        if best is not low and (low['final']['heldout_wmse_mean'] - best['final']['heldout_wmse_mean']) / best['final']['heldout_wmse_mean'] < 0.01:
            out['notes'].append(f'{preset}: held-out losses within 1%; keeping the lower learning rate {low["ref_lr"]}')
            best = low
        best_by_preset[preset] = best
    out['lr_choice'] = {p: dict(run_dir=a['run_dir'], ref_lr=a['ref_lr']) for p, a in best_by_preset.items()}
    winner = best_by_preset.get('E112') or (min(best_by_preset.values(), key=lambda a: a['final']['fid']) if best_by_preset else None)
    if 'E112' in best_by_preset:
        base = best_by_preset['E112']
        better = [b for p, b in best_by_preset.items() if p != 'E112'
                  and base['final']['fid'] - b['final']['fid'] > 2 * pooled_sd(base['final'], b['final'])]
        for b in better:
            out['notes'].append(f"{b['preset']} beats E112 by more than 2 SD ({b['final']['fid']:.3f} vs {base['final']['fid']:.3f})")
        if better:
            winner = min(better, key=lambda a: a['final']['fid'])
            out['notes'].append(f"{winner['preset']} replaces E112")
    ddpm = [a for a in arms if a['family'] == 'edm']
    out['rerun_half_lr'] = False
    if ddpm and winner is not None:
        d = min(ddpm, key=lambda a: a['final']['fid'])
        e = min([a for a in edm2 if not a['unstable']] or edm2, key=lambda a: a['final']['fid'])
        margin = e['final']['fid'] - d['final']['fid']
        if margin > 2 * pooled_sd(e['final'], d['final']):
            half = 0.5 * presets.NETWORK_PRESETS[winner['preset']]['ref_lr']
            if any(a['preset'] == winner['preset'] and a['ref_lr'] <= half * (1 + 1e-6) for a in edm2):
                out['notes'].append(f"DDPM++128 still beats the best EDM2 arm by {margin:.3f} FID (> 2 SD) with the 0.5 x alpha_ref arm "
                                    f"included: DDPM++128 becomes the teacher (design 10.4)")
                winner = d
            else:
                out['notes'].append(f"DDPM++128 beats the best EDM2 arm by {margin:.3f} FID (> 2 SD). Design 10.4: run one more {winner['preset']} "
                                    f"arm at 0.5 x alpha_ref ({half:g}) for the same time (HALF_LR_ARM=1 HALF_LR_PRESET={winner['preset']} "
                                    f"bash teacher/scripts/run_pilot.sh), which re-scores all arms; if DDPM++128 still wins, it becomes the teacher.")
                out['rerun_half_lr'] = True
                out['ddpmpp_candidate'] = d['run_dir']
    elif winner is None and ddpm:
        winner = min(ddpm, key=lambda a: a['final']['fid'])
    out['winner'] = winner and dict(run_dir=winner['run_dir'], preset=winner['preset'], ref_lr=winner['ref_lr'])
    return out

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True)
    p.add_argument('--runs', required=True, help='Comma-separated pilot run directories')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--per-class', type=int, default=20)
    p.add_argument('--steps', type=int, default=32, help='Heun steps of the FID sampler')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--heldout-max', type=int, default=None, help='Use only the first N holdout images for the denoising loss (tests)')
    p.add_argument('--out', default=None, help='Default: <first run>/../pilot_decision.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    prep = data_mod.load_prepared(args.prep_dir)
    proxy = fid_mod.ProxyFID(prep, device, num_workers=args.num_workers)
    seeds = [int(s) for s in args.seeds.split(',')]
    runs = [r.strip() for r in args.runs.split(',') if r.strip()]
    arms = []
    for r in runs:
        info = run_info(r)
        snaps = phema_tools.list_snapshots(r, stds=[0.100])
        info['final_nimg'] = snaps[-1]['nimg'] if snaps else 0
        arms.append(info)
    common = min(a['final_nimg'] for a in arms)
    kw = dict(seeds=seeds, device=device, per_class=args.per_class, sampler_kwargs=dict(num_steps=args.steps), batch_size=args.batch,
              heldout_max=args.heldout_max)
    for a in arms:
        print(f"Scoring {a['run_dir']} ({a['preset']}, ref_lr {a['ref_lr']}) ...")
        cache = {}
        a['final'] = evaluate_at(prep, proxy, a['run_dir'], a['final_nimg'], cache=cache, **kw)
        a['at_common'] = evaluate_at(prep, proxy, a['run_dir'], common, cache=cache, **kw)
        f = a['final']
        print(f"  final {f['nimg'] / 1e6:.2f} Mimg: FID {f['fid']:.3f} +- {f['sd']:.3f}; held-out wmse {f['heldout_wmse_mean']:.5f}; "
              f"{a['img_per_sec']:.1f} img/s; spikes {a['loss_spikes']}, nonfinite steps {a['nonfinite_steps']}")
    decision = decide(arms)
    result = dict(created=now_iso(), common_nimg=common, arms=arms, decision=decision)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(runs[0])), 'pilot_decision.json')
    write_json(out, result)
    print('\nDecision:')
    for n in decision['notes']:
        print('  - ' + n)
    if decision['winner']:
        w = decision['winner']
        from .training import find_latest_state
        state = find_latest_state(w['run_dir'])
        decision['winner']['state'] = state
        write_json(out, result)
        if decision['rerun_half_lr']:
            print(f"  Provisional winner: {w['preset']} at ref_lr {w['ref_lr']} ({w['run_dir']}); the family decision waits for the "
                  f"0.5 x alpha_ref arm.")
        else:
            print(f"  Winner: {w['preset']} at ref_lr {w['ref_lr']} ({w['run_dir']}).")
            print(f"  Resume it on the main set: python -m teacher.train --prep-dir {args.prep_dir} --run-dir runs/main_{w['preset']} "
                  f"--preset {w['preset']} --subset main --allow-data-switch --resume {state} --total-steps <N from the benchmark>")
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
