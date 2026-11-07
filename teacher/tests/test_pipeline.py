import glob
import json
import os
import shutil
import tempfile

import torch

from teacher import fid as fid_mod
from teacher import official

SPLIT_DIR = os.path.join(official.PROJECT_ROOT, 'data', 'pokemon-generation-one-22k')
_STATE = {}

class FakeInception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(12, 3, 3, 3, generator=g) / 10

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x.float() / 255, 8)
        x = torch.nn.functional.conv2d(x, self.w)
        return torch.nn.functional.adaptive_avg_pool2d(x.relu(), 4).flatten(1)

class FakeExtractor(fid_mod.FeatureExtractor):
    def __init__(self, device, num_workers=0, batch_size=64, model=None):
        super().__init__(device, num_workers=0, batch_size=batch_size, model=FakeInception())

def _setup():
    if _STATE:
        return _STATE
    fid_mod.FeatureExtractor = FakeExtractor
    base = os.environ.get('FASTGEN_TEST_DIR') or tempfile.mkdtemp(prefix='fastgen-teacher-test-')
    os.makedirs(base, exist_ok=True)
    root, prep = os.path.join(base, 'dataset'), os.path.join(base, 'prep')
    from teacher.tests import make_synthetic_dataset
    if not os.path.isdir(root):
        make_synthetic_dataset.main(['--root', root, '--split-dir', SPLIT_DIR, '--limit-per-class', '50'])
    if not os.path.isdir(prep):
        from teacher import prepare_data
        prepare_data.main(['--data-root', root, '--split-dir', root, '--out', prep, '--device', 'cpu', '--num-workers', '0', '--floors'])
    _STATE.update(base=base, root=root, prep=prep)
    return _STATE

def _train(run_dir, *extra, preset='tiny-edm2', subset='pilot', fid=False, guard=False):
    from teacher import train
    s = _setup()
    argv = ['--prep-dir', s['prep'], '--run-dir', run_dir, '--preset', preset, '--subset', subset, '--device', 'cpu',
            '--batch-size', '16', '--micro-batch', '8', '--status-every', '64', '--snapshot-dense', '64', '--snapshot-dense-until', '256',
            '--snapshot-mid-until', '0', '--snapshot-every', '128', '--state-every', '64', '--early-samples', '64',
            '--fid-per-class', '1', '--fid-steps', '2', '--fid-batch', '64', '--fid-workers', '0'] + list(extra)
    argv += [] if fid else ['--no-fid']
    argv += [] if guard else ['--no-guard']
    train.main(argv)

def _state(run_dir):
    from teacher.training import find_latest_state
    return torch.load(find_latest_state(run_dir), map_location='cpu', weights_only=True)

def test_prepare_outputs():
    s = _setup()
    stats = json.load(open(os.path.join(s['prep'], 'stats.json')))
    assert stats['n_canary'] == 755 and stats['n_holdout'] == 3020 and stats['num_classes'] == 151
    assert 0.1 < stats['sigma_data'] < 1.0
    floors = json.load(open(os.path.join(s['prep'], 'floors.json')))
    assert len(floors['pairs']) == 5 and floors['fid_mean'] >= 0
    assert len(glob.glob(os.path.join(s['prep'], 'refs', 'holdout', '*.png'))) == 3020

def test_resume_is_exact():
    s = _setup()
    a, b = os.path.join(s['base'], 'resume_a'), os.path.join(s['base'], 'resume_b')
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    _train(a, '--total-steps', '16')
    _train(b, '--total-steps', '8')
    _train(b, '--total-steps', '16')
    sa, sb = _state(a), _state(b)
    assert sa['state']['cur_nimg'] == sb['state']['cur_nimg'] == 256
    for k in sa['net']:
        assert torch.equal(sa['net'][k], sb['net'][k]), k
    for ea, eb in zip(sa['ema']['emas'], sb['ema']['emas']):
        for k in ea:
            assert torch.equal(ea[k], eb[k]), k
    la = [json.loads(l)['loss'] for l in open(os.path.join(a, 'stats.jsonl'))]
    lb = [json.loads(l)['loss'] for l in open(os.path.join(b, 'stats.jsonl'))]
    assert la == lb
    t = json.load(open(os.path.join(b, 'training_time.json')))
    assert len(t['segments']) == 2 and t['total_wall_hours'] > 0 and t['stop_reason'] == 'total_steps'

def test_in_run_fid_guard_and_data_switch():
    s = _setup()
    pilot, main = os.path.join(s['base'], 'pilot_arm'), os.path.join(s['base'], 'main_run')
    for d in (pilot, main):
        shutil.rmtree(d, ignore_errors=True)
    _train(pilot, '--total-steps', '16', '--quick-fid-every', '128', '--decision-fid-every', '256', '--guard-every', '128',
           '--guard-max-images', '16', '--converge-min', '0', fid=True, guard=True)
    pilot_fid = [json.loads(l) for l in open(os.path.join(pilot, 'fid.jsonl'))]
    assert any(r['kind'] == 'decision' and len(r['fids']) == 3 for r in pilot_fid)
    assert all(list(r.get('per_ref', {'holdout': 0})) == ['holdout'] for r in pilot_fid if r['kind'] == 'quick')
    assert len(open(os.path.join(pilot, 'guard.jsonl')).readlines()) == 2
    assert glob.glob(os.path.join(pilot, 'samples', 'early-*.png')) and glob.glob(os.path.join(pilot, 'samples', 'fid-*.png'))
    from teacher.training import find_latest_state
    _train(main, '--total-steps', '24', '--resume', find_latest_state(pilot), '--allow-data-switch',
           '--quick-fid-every', '128', '--decision-fid-every', '0', subset='main', fid=True)
    st = _state(main)['state']
    assert st['subset'] == 'main' and st['data_offset_nimg'] == 256 and st['cur_nimg'] == 384
    assert st['lineage'][0]['subset'] == 'pilot'
    assert os.path.isfile(os.path.join(main, 'lineage', 'pilot_arm', 'stats.jsonl'))
    pilot_snaps = {os.path.basename(p) for p in glob.glob(os.path.join(pilot, 'ema-*.pt'))}
    main_snaps = {os.path.basename(p) for p in glob.glob(os.path.join(main, 'ema-*.pt'))}
    assert pilot_snaps < main_snaps
    q = [json.loads(l) for l in open(os.path.join(main, 'fid.jsonl'))]
    assert q and list(q[-1]['per_ref']) == ['train0', 'train1', 'train2']
    t = json.load(open(os.path.join(main, 'training_time.json')))
    assert len(t['segments']) == 2 and t['lineage'][0]['subset'] == 'pilot'
    _STATE['main'] = main

def test_convergence_stop():
    s = _setup()
    d = os.path.join(s['base'], 'converge')
    shutil.rmtree(d, ignore_errors=True)
    _train(d, '--total-steps', '40', '--decision-fid-every', '64', '--quick-fid-every', '0', '--converge-min', '0',
           '--converge-patience', '1', '--converge-sd', '1e9', fid=True)
    st = _state(d)['state']
    assert st['stop_reason'] == 'converged' and st['cur_nimg'] == 128, st['cur_nimg']

def _names(run_dir, pattern):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, pattern)))

def _nimgs(run_dir, log='stats.jsonl'):
    return [json.loads(l)['nimg'] for l in open(os.path.join(run_dir, log))]

def _raises(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False

def test_forced_restart_sets_later_files_aside():
    s = _setup()
    d = os.path.join(s['base'], 'restart')
    shutil.rmtree(d, ignore_errors=True)
    _train(d, '--total-steps', '16')
    older = os.path.join(d, 'training-state-0000000192.pt')
    assert _names(d, 'training-state-*.pt') == ['training-state-0000000192.pt', 'training-state-0000000256.pt']
    assert _raises(_train, d, '--total-steps', '24', '--resume', older)
    _train(d, '--total-steps', '24', '--resume', older, '--force')
    aside = glob.glob(os.path.join(d, 'abandoned', '*'))
    assert len(aside) == 1
    assert _names(aside[0], '*.pt') == ['ema-0000000256-0.050.pt', 'ema-0000000256-0.100.pt', 'training-state-0000000256.pt']
    assert max(_nimgs(aside[0])) == 256
    old = torch.load(os.path.join(aside[0], 'ema-0000000256-0.100.pt'), weights_only=True)['state_dict']
    new = torch.load(os.path.join(d, 'ema-0000000256-0.100.pt'), weights_only=True)['state_dict']
    assert all(torch.equal(old[k], new[k]) for k in old)
    assert _names(d, 'training-state-*.pt') == ['training-state-0000000320.pt', 'training-state-0000000384.pt']
    assert _nimgs(d) == [64, 128, 192, 256, 320, 384]
    st = _state(d)['state']
    assert st['cur_nimg'] == 384 and [e['kind'] for e in st['events']].count('set_aside') == 1
    assert _raises(_train, d, '--total-steps', '4', '--resume', 'none')
    _train(d, '--total-steps', '4', '--resume', 'none', '--force')
    assert len(glob.glob(os.path.join(d, 'abandoned', '*'))) == 2
    assert _names(d, 'training-state-*.pt') == ['training-state-0000000064.pt'] and _nimgs(d) == [64]
    assert _names(d, 'ema-*.pt') == ['ema-0000000064-0.050.pt', 'ema-0000000064-0.100.pt']

def test_override_ref_lr_in_another_directory():
    s = _setup()
    a, b = os.path.join(s['base'], 'lr_src'), os.path.join(s['base'], 'lr_dst')
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    _train(a, '--total-steps', '8')
    from teacher.training import find_latest_state
    src = find_latest_state(a)
    assert _raises(_train, b, '--total-steps', '12', '--resume', src, '--ref-lr', '0.005')
    assert not os.path.exists(os.path.join(b, 'lineage'))
    _train(b, '--total-steps', '12', '--resume', src, '--override-ref-lr', '0.005')
    sb = _state(b)
    r = sb['recipe']
    assert r['ref_lr'] == 0.005 and [(e['old'], e['new']) for e in sb['state']['events'] if e['kind'] == 'ref_lr_override'] == [(0.01, 0.005)]
    cfg = json.load(open(os.path.join(b, 'config.json')))
    assert cfg['recipe']['ref_lr'] == 0.005 and len(cfg['ref_lr_overrides']) == 1
    last = [json.loads(l) for l in open(os.path.join(b, 'stats.jsonl'))][-1]
    lr = official.learning_rate_schedule(cur_nimg=last['nimg'] - 16, batch_size=16, ref_lr=0.005, ref_batches=r['ref_batches'],
                                         rampup_Mimg=r['rampup_steps'] * 16 / 1e6)
    assert abs(last['lr'] - float(lr)) < 1e-15, (last['lr'], lr)
    assert set(_names(a, 'ema-*.pt')) < set(_names(b, 'ema-*.pt'))
    assert os.path.isfile(os.path.join(b, 'lineage', 'lr_src', 'stats.jsonl'))
    assert sb['state']['lineage'][-1]['run_dir'] == os.path.abspath(a)
    for p in glob.glob(os.path.join(b, 'training-state-*.pt')):
        os.remove(p)
    _train(b, '--total-steps', '12', '--resume', src, '--override-ref-lr', '0.005')
    assert _nimgs(b) == [192] and len(glob.glob(os.path.join(b, 'abandoned', '*'))) == 1
    assert len(json.load(open(os.path.join(b, 'config.json')))['ref_lr_overrides']) == 1
    c = os.path.join(s['base'], 'lr_other')
    shutil.rmtree(c, ignore_errors=True)
    _train(c, '--total-steps', '4')
    os.remove(os.path.join(c, 'training-state-0000000064.pt'))
    assert _raises(_train, c, '--total-steps', '12', '--resume', src)

def test_disk_space_stop():
    s = _setup()
    d = os.path.join(s['base'], 'disk')
    shutil.rmtree(d, ignore_errors=True)
    _train(d, '--total-steps', '16', '--min-free-gb', '1e12')
    st = _state(d)['state']
    assert st['stop_reason'] == 'disk_space' and st['cur_nimg'] == 64, (st['stop_reason'], st['cur_nimg'])
    assert 'disk_free_gb' in json.loads(open(os.path.join(d, 'stats.jsonl')).readline())

def test_reconstruct_sample_eval():
    s = _setup()
    main = _STATE.get('main')
    if main is None:
        test_in_run_fid_guard_and_data_switch()
        main = _STATE['main']
    from teacher import eval_fid, reconstruct, sample
    reconstruct.main(['--run-dir', main, '--std', '0.05,0.1'])
    recs = sorted(glob.glob(os.path.join(main, 'phema', 'phema-*.pt')))
    assert len(recs) == 2
    out = os.path.join(s['base'], 'grid.png')
    sample.main(['--net', recs[0], '--classes', '0-7', '--steps', '2', '--out', out, '--device', 'cpu'])
    sample.main(['--net', recs[1], '--guide-net', recs[0], '--guidance', '1.5', '--classes', '3,9', '--steps', '2', '--out', out, '--device', 'cpu'])
    sample.main(['--net', recs[1], '--cfg', '1.2', '--classes', '3,9', '--steps', '2', '--out', out, '--device', 'cpu'])
    res = os.path.join(s['base'], 'eval.jsonl')
    eval_fid.main(['--prep-dir', s['prep'], '--net', recs[1], '--seeds', '0,1', '--per-class', '1', '--steps', '2', '--device', 'cpu',
                   '--num-workers', '0', '--out', res])
    r = json.loads(open(res).readlines()[-1])
    assert len(r['fids']) == 2 and r['sd'] is not None

def test_posttrain_sweep():
    s = _setup()
    main = _STATE.get('main')
    if main is None:
        test_in_run_fid_guard_and_data_switch()
        main = _STATE['main']
    from teacher import posttrain_sweep as pt
    pt.CFG_W, pt.CFG_STD = [1.1, 1.3], [0.05]
    pt.AG_FRACTIONS, pt.AG_GUIDE_STD, pt.AG_W, pt.AG_REFINE_STD = [1 / 4, 1 / 2], [0.10], [1.3], [0.05]
    out = os.path.join(s['base'], 'posttrain')
    shutil.rmtree(out, ignore_errors=True)
    pt.main(['--prep-dir', s['prep'], '--run-dir', main, '--stage', 'all', '--ema-grid', '0.05,0.1', '--per-class', '1', '--steps', '2',
             '--device', 'cpu', '--num-workers', '0', '--out', out])
    summary = json.load(open(os.path.join(out, 'summary.json')))
    assert {'ema', 'cfg', 'ag'} <= set(summary)
    for f in ['teacher_unguided.pt', 'teacher_ag_main.pt', 'guide.pt', 'mean_image.pt', 'teacher_config.json']:
        assert os.path.isfile(os.path.join(out, 'handoff', f)), f
    from teacher import phema_tools
    net, _ = phema_tools.load_network(os.path.join(out, 'handoff', 'teacher_unguided.pt'))
    assert net.num_classes == 151

def test_pilot_compare():
    s = _setup()
    arms = []
    for name, extra, preset in [('arm_lr1', [], 'tiny-edm2'), ('arm_lr2', ['--ref-lr', '0.02'], 'tiny-edm2'), ('arm_ddpm', [], 'tiny-ddpmpp')]:
        d = os.path.join(s['base'], name)
        shutil.rmtree(d, ignore_errors=True)
        _train(d, '--total-steps', '8', *extra, preset=preset)
        arms.append(d)
    from teacher import pilot_compare
    out = os.path.join(s['base'], 'pilot_decision.json')
    pilot_compare.main(['--prep-dir', s['prep'], '--runs', ','.join(arms), '--seeds', '0,1', '--per-class', '1', '--steps', '2',
                        '--heldout-max', '64', '--device', 'cpu', '--num-workers', '0', '--out', out])
    dec = json.load(open(out))
    assert dec['decision']['winner'] is not None and len(dec['arms']) == 3

def test_benchmark_cpu():
    s = _setup()
    from teacher import benchmark
    out = os.path.join(s['base'], 'bench')
    benchmark.main(['--prep-dir', s['prep'], '--out', out, '--device', 'cpu', '--presets', 'tiny-edm2,tiny-ddpmpp', '--modes', 'eager',
                    '--steps', '1', '--warmup', '1', '--micro-batch', '64'])
    b = json.load(open(os.path.join(out, 'benchmark.json')))
    assert all(r['ok'] for r in b['results']), b['results']
