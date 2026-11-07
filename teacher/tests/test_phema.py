import os
import tempfile

import numpy as np
import torch

from teacher import networks, official, phema_tools, train
from teacher.training import count_snapshots, snapshot_interval
from teacher.utils import MI, crossed, parse_nimg

CFG = dict(family='edm2', kwargs=dict(img_resolution=64, img_channels=3, label_dim=151, model_channels=16, channel_mult=[1, 2, 2],
                                      num_blocks=1, attn_resolutions=[16], channels_per_head=16, augment_dim=9))

def test_reconstruct_tracked_profile():
    torch.manual_seed(0)
    net = networks.build_network(CFG)
    with torch.no_grad():
        net.mean_image.copy_(torch.randn(3, 64, 64) * 0.2)
    ema = official.PowerFunctionEMA(net, stds=[0.05, 0.10])
    with tempfile.TemporaryDirectory() as d:
        nimg = 0
        for step in range(1, 41):
            with torch.no_grad():
                for p in net.parameters():
                    if p.requires_grad:
                        p.add_(torch.randn_like(p) * 0.01)
            net.post_step_()
            nimg += 256
            ema.update(cur_nimg=nimg, batch_size=256)
            if step % 8 == 0:
                for e, suffix in ema.get():
                    phema_tools.save_snapshot(os.path.join(d, phema_tools.snapshot_name(nimg, float(suffix[1:]))), e, nimg, float(suffix[1:]), CFG)
        snaps = phema_tools.list_snapshots(d)
        assert len(snaps) == 10 and snaps[-1]['nimg'] == nimg
        rec = phema_tools.reconstruct(snaps, [0.10, 0.07], out_nimg=nimg, verbose=False)
        tracked = dict(ema.emas[1].state_dict())
        r10 = [r for r in rec if abs(r['std'] - 0.10) < 1e-9][0]
        for k, v in tracked.items():
            tol = 2e-3 * float(v.abs().max()) + 1e-6
            assert torch.allclose(r10['state_dict'][k].float(), v.float(), atol=tol), k
        assert torch.equal(r10['state_dict']['mean_image'], net.mean_image.detach())
        path = phema_tools.save_reconstruction(rec[0], os.path.join(d, 'out'))
        loaded, data = phema_tools.load_network(path)
        assert data['nimg'] == nimg and networks.count_parameters(loaded)['total'] == networks.count_parameters(net)['total']

def test_reconstruction_time_must_match():
    with tempfile.TemporaryDirectory() as d:
        net = networks.build_network(CFG)
        for nimg in (512, 1024):
            phema_tools.save_snapshot(os.path.join(d, phema_tools.snapshot_name(nimg, 0.1)), net, nimg, 0.1, CFG)
        try:
            phema_tools.reconstruct(phema_tools.list_snapshots(d), [0.1], out_nimg=700, verbose=False)
        except ValueError:
            return
        raise AssertionError('expected a ValueError')

def _profile_error(times, t_out, s_out, stds=(0.05, 0.10)):
    corr = official.phema.power_function_correlation
    ins = [(t, s) for t in times if 0 < t <= t_out for s in stds]
    a = np.float64([t for t, _ in ins]).reshape(-1, 1)
    b = np.float64([s for _, s in ins]).reshape(-1, 1)
    A = corr(a, b, a.T, b.T)
    B = corr(a, b, np.float64([[t_out]]), np.float64([[s_out]]))[:, 0]
    x = official.solve_posthoc_coefficients(a[:, 0], b[:, 0], t_out, s_out)[:, 0]
    c = float(corr(np.float64(t_out), np.float64(s_out), np.float64(t_out), np.float64(s_out)))
    return float(np.sqrt(max(c - 2 * x @ B + x @ A @ x, 0.0) / c))

def _shift_distance(t, s, f=1.1):
    corr = official.phema.power_function_correlation
    t, s = np.float64(t), np.float64(s)
    return float(np.sqrt((corr(t, s, t, s) + corr(t, f * s, t, f * s) - 2 * corr(t, s, t, f * s)) / corr(t, s, t, s)))

def test_default_snapshot_schedule():
    a = train.build_parser().parse_args(['--prep-dir', 'x', '--run-dir', 'y', '--total-steps', '1'])
    sched = dict(dense=parse_nimg(a.snapshot_dense), dense_until=parse_nimg(a.snapshot_dense_until), mid=parse_nimg(a.snapshot_mid),
                 mid_until=parse_nimg(a.snapshot_mid_until), sparse=parse_nimg(a.snapshot_every))
    T = 150_000_000 // 256 * 256
    times, prev = [], 0
    for cur in range(256, T + 1, 256):
        if crossed(prev, cur, snapshot_interval(sched, cur)):
            times.append(cur)
        prev = cur
    assert count_snapshots(sched, 0, T) == len(times)
    assert count_snapshots(sched, 5 * MI, T) == len([t for t in times if t > 5 * MI])
    assert times[:8] == [k * MI // 2 for k in range(1, 9)] and 6 * MI in times and 48 * MI in times and 54 * MI in times
    times.append(T)
    assert len(times) <= 47
    for s in (0.03, 0.05, 0.075, 0.10, 0.13, 0.16, 0.20):
        assert _profile_error(times, T, s) < _shift_distance(T, s) / 4, s
    for f in (0.5, 0.7, 0.85):
        t = max(u for u in times if u <= f * T)
        for s in (0.05, 0.10, 0.20):
            assert _profile_error(times, t, s) < _shift_distance(t, s) / 4, (f, s)
    old = sorted(set([k * MI // 2 for k in range(1, 9)] + list(range(6 * MI, T + 1, 2 * MI)) + [T]))
    for f in (1 / 32, 1 / 16, 1 / 8, 1 / 4):
        t_new, t_old = max(u for u in times if u <= f * T), max(u for u in old if u <= f * T)
        for s in (0.10, 0.175):
            assert _profile_error(times, t_new, s) <= _profile_error(old, t_old, s) + 1e-3, (f, s)
