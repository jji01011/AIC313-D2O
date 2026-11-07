"""Hour-one GPU benchmark (TEACHER_DESIGN.md sections 3.5 and 10.2).

    python -m teacher.benchmark --prep-dir data/teacher_prep --out runs/benchmark

Measures what the GPU is (name, SM count, memory, an 8192^2 bf16 matmul in TFLOPS), then times
real training steps (the trainer's own `train_step` and `finish_step`: augmentation, loss,
backward, Adam, EMA update, weight projection) for E112 in several modes (eager, torch.compile,
channels_last, SDPA attention and combinations, joined with '+') and for the other presets in
E112's fastest mode, recording img/s and `max_memory_allocated`.
It also times forward-only sampling at batch 256, which sets the cost of FID checks.
From R (the best E112 img/s) it applies the design's size rule (teacher width and pilot arms)
and converts days into --total-steps, and the planned images into the dropout rate.
Results are appended to <out>/benchmark_history.jsonl so runs at different times of day
(time-sliced GPUs) can be compared.
"""

import argparse
import gc
import json
import os
import time

import numpy as np
import torch

from . import presets
from .utils import device_info, now_iso, write_json

def gpu_identity(device):
    info = device_info(device)
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        info.update(mem_free=free, mem_total=total)
        a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
        b = torch.randn_like(a)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize(device)
        t = time.time()
        for _ in range(50):
            a @ b
        torch.cuda.synchronize(device)
        info['bf16_matmul_tflops'] = 50 * 2 * 8192 ** 3 / (time.time() - t) / 1e12
        del a, b
        torch.cuda.empty_cache()
        sm = info.get('sm_count')
        info['reading'] = {108: 'full A100 or time-sliced vGPU', 56: 'MIG 4g (40/80 GB card)', 42: 'MIG 3g', 28: 'MIG 2g (80 GB card)', 14: 'MIG 1g (80 GB card)'}.get(sm, 'unknown')
    return info

def _trainer(prep_dir, run_dir, preset, mode, micro_batch, device, extra=()):
    from . import train
    from .training import TeacherTraining
    argv = ['--prep-dir', prep_dir, '--run-dir', run_dir, '--preset', preset, '--subset', 'main', '--resume', 'none', '--force',
            '--micro-batch', str(micro_batch), '--device', str(device), '--no-fid', '--no-guard', '--total-steps', '0']
    if 'compile' in mode:
        argv.append('--compile')
    if 'channels_last' in mode:
        argv.append('--channels-last')
    if 'sdpa' in mode and not preset.startswith('DDPM'):
        argv += ['--attention', 'sdpa']
    return TeacherTraining(train.build_parser().parse_args(argv + list(extra)))

def mode_flags(mode):
    parts = mode.split('+')
    return ' '.join(f for f, on in [('--compile', 'compile' in parts), ('--channels-last', 'channels_last' in parts),
                                     ('--attention sdpa', 'sdpa' in parts)] if on)

def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()
    except Exception:
        pass

def time_training(prep_dir, out_dir, preset, mode, micro_batch, device, steps, warmup):
    run_dir = os.path.join(out_dir, 'runs', f'{preset}-{mode}-mb{micro_batch}')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    tr = _trainer(prep_dir, run_dir, preset, mode, micro_batch, device)
    batch = tr.batch_size
    cur = 0
    losses_seen = []
    try:
        t_compile = time.time()
        for _ in range(warmup):
            tr.train_step(cur, None)
            cur += batch
            tr.finish_step(cur)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        warm_sec = time.time() - t_compile
        t0 = time.time()
        for _ in range(steps):
            g, _nf, _lr = tr.train_step(cur, None)
            losses_seen.append(g)
            cur += batch
            tr.finish_step(cur)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        dt = time.time() - t0
        res = dict(preset=preset, mode=mode, micro_batch=tr.state['micro_batch'], steps=steps, warmup=warmup, warmup_sec=warm_sec,
                   img_per_sec=steps * batch / dt, sec_per_step=dt / steps, ok=True,
                   params=sum(p.numel() for p in tr.net.parameters()))
        if device.type == 'cuda':
            res['max_memory_allocated'] = torch.cuda.max_memory_allocated(device)
            res['max_memory_reserved'] = torch.cuda.max_memory_reserved(device)
        net = tr.ema.emas[-1]
        x = torch.randn([256, 3, 64, 64], device=device)
        sigma = torch.full([256], 2.0, device=device)
        y = net.one_hot(torch.arange(256, device=device) % net.num_classes)
        with torch.no_grad():
            for _ in range(3):
                net(x, sigma, y)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t1 = time.time()
            n_fwd = 10
            for _ in range(n_fwd):
                net(x, sigma, y)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        res['forward_img_per_sec'] = n_fwd * 256 / (time.time() - t1)
        res['fid_check_minutes_per_seed'] = 3020 * 63 / res['forward_img_per_sec'] / 60
        if device.type == 'cuda':
            res['max_memory_allocated_incl_sampling'] = torch.cuda.max_memory_allocated(device)
    except torch.cuda.OutOfMemoryError as e:
        res = dict(preset=preset, mode=mode, micro_batch=micro_batch, ok=False, error=f'OOM: {str(e)[:200]}')
    except Exception as e:
        res = dict(preset=preset, mode=mode, micro_batch=micro_batch, ok=False, error=f'{type(e).__name__}: {str(e)[:300]}')
    _free(tr)
    return res

def check_mode_matches(prep_dir, out_dir, preset, micro_batch, device, steps, mode):
    traces = {}
    for mode in ['eager', mode]:
        tr = _trainer(prep_dir, os.path.join(out_dir, 'runs', f'check-{preset}-{mode}'), preset, mode, micro_batch, device)
        from . import losses
        cur, trace = 0, []
        for _ in range(steps):
            b = losses.BucketStats(device)
            tr.train_step(cur, b)
            cur += tr.batch_size
            tr.finish_step(cur)
            trace.append(b.summary()['loss'])
        traces[mode] = trace
        _free(tr)
    e, c = np.array(traces['eager']), np.array(traces[mode])
    w = max(1, steps // 10)
    ew = np.convolve(e, np.ones(w) / w, mode='valid')
    cw = np.convolve(c, np.ones(w) / w, mode='valid')
    return dict(mode=mode, steps=steps, window=w, max_abs_diff_windowed=float(np.max(np.abs(ew - cw))), eager_last=float(ew[-1]), mode_last=float(cw[-1]),
                eager_trace=traces['eager'], mode_trace=traces[mode])

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True)
    p.add_argument('--out', default='runs/benchmark')
    p.add_argument('--device', default='cuda')
    p.add_argument('--presets', default='E112,E96,E80,E128-1233,E144-1223,DDPMpp128')
    p.add_argument('--modes', default='eager,sdpa,compile,compile+sdpa,compile+channels_last,compile+channels_last+sdpa', help='Modes tried for E112 (or the first preset)')
    p.add_argument('--all-modes-for-all', action='store_true', help='Try every mode for every preset')
    p.add_argument('--micro-batch', type=int, default=64)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--check-steps', type=int, default=0, help="Also compare the losses of eager and the best mode over this many steps (design: 2000)")
    p.add_argument('--days', default='3,4,5,6,7', help='Days to convert into --total-steps')
    p.add_argument('--overhead', type=float, default=0.05, help='Share of wall-clock spent on FID checks, the guard and checkpoints')
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    os.makedirs(args.out, exist_ok=True)
    ident = gpu_identity(device)
    print(json.dumps({k: v for k, v in ident.items()}, indent=2, default=str))
    names = [s.strip() for s in args.presets.split(',') if s.strip()]
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    results = []
    first = names[0]

    def timed(name, mode):
        mb = args.micro_batch
        while True:
            r = time_training(args.prep_dir, args.out, name, mode, mb, device, args.steps, args.warmup)
            if r['ok'] or not r.get('error', '').startswith('OOM') or mb <= 16:
                return r
            mb //= 2
            print(f'{name} {mode}: out of memory, retrying at micro-batch {mb}')

    for mode in modes:
        r = timed(first, mode)
        print(json.dumps(r, default=str))
        results.append(r)
    ok = [r for r in results if r['ok']]
    best_mode = max(ok, key=lambda r: r['img_per_sec'])['mode'] if ok else 'eager'
    for name in names[1:]:
        for mode in (modes if args.all_modes_for_all else [best_mode]):
            r = timed(name, mode)
            print(json.dumps(r, default=str))
            results.append(r)

    summary = dict(created=now_iso(), gpu=ident, results=results, best_mode=best_mode, best_mode_flags=mode_flags(best_mode))
    e112 = [r for r in results if r['ok'] and r['preset'] == 'E112']
    if e112:
        R = max(r['img_per_sec'] for r in e112)
        rule = presets.size_rule(R)
        summary.update(R_e112=R, size_rule=rule)
        speeds = {r['preset']: max(x['img_per_sec'] for x in results if x['ok'] and x['preset'] == r['preset']) for r in results if r['ok']}
        teacher_speed = speeds.get(rule['teacher'])
        plan = []
        for d in [float(x) for x in args.days.split(',')]:
            if teacher_speed:
                steps = presets.steps_for_days(teacher_speed * (1 - args.overhead), d)
                plan.append(dict(days=d, preset=rule['teacher'], img_per_sec=teacher_speed, total_steps=steps,
                                 mimg=steps * presets.BATCH_SIZE / 1e6, dropout=presets.dropout_for_planned_images(steps * presets.BATCH_SIZE)))
        summary['plan'] = plan
        summary['speeds'] = speeds
        print(f"\nR (E112, best mode '{best_mode}') = {R:.1f} img/s -> teacher {rule['teacher']}, pilot arms {rule['arms']}")
        print(f"Training flags for that mode: {mode_flags(best_mode) or '(none: eager)'}")
        print(f"Steps below assume {args.overhead:.0%} of wall-clock goes to evaluation and checkpoints.")
        for row in plan:
            print(f"  {row['days']:.0f} days of {row['preset']} at {row['img_per_sec']:.0f} img/s: --total-steps {row['total_steps']} "
                  f"({row['mimg']:.1f} Mimg), dropout {row['dropout']:.2f}")
    if args.check_steps > 0 and best_mode != 'eager':
        summary['mode_check'] = check_mode_matches(args.prep_dir, args.out, first, args.micro_batch, device, args.check_steps, best_mode)
        c = summary['mode_check']
        print(f"'{best_mode}' vs eager over {c['steps']} steps: max windowed loss difference {c['max_abs_diff_windowed']:.4f} "
              f"(final {c['mode_last']:.4f} vs {c['eager_last']:.4f})")
    write_json(os.path.join(args.out, 'benchmark.json'), summary)
    with open(os.path.join(args.out, 'benchmark_history.jsonl'), 'a') as f:
        f.write(json.dumps(summary, default=str) + '\n')
    print(f"wrote {os.path.join(args.out, 'benchmark.json')}")

if __name__ == '__main__':
    main()
