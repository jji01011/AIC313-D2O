import json
import os
import tempfile

from teacher import pilot_compare as pc

def _arm(preset, ref_lr, fid, sd=0.1, wmse=1.0, family=None, unstable=False):
    family = family or ('edm' if preset.startswith('DDPM') else 'edm2')
    return dict(run_dir=f'runs/{preset}_{ref_lr}', preset=preset, family=family, ref_lr=ref_lr, unstable=unstable,
                final=dict(fid=fid, sd=sd, heldout_wmse_mean=wmse))

def test_spike_rule_on_logs():
    d = tempfile.mkdtemp(prefix='fastgen-pilot-')
    with open(os.path.join(d, 'config.json'), 'w') as f:
        json.dump(dict(network=dict(preset='E112', family='edm2'), recipe=dict(ref_lr=0.0045)), f)
    loss = [0.30, 0.05, -0.10, -0.20, -0.28, -0.33, -0.36, -0.38]
    wmse = [1.00, 0.80, 0.70, 0.65, 0.62, 0.60, 0.59, 0.58]
    with open(os.path.join(d, 'stats.jsonl'), 'w') as f:
        for i, (l, w) in enumerate(zip(loss, wmse)):
            f.write(json.dumps(dict(nimg=(i + 1) * 1024, step=(i + 1) * 4, loss=l, wmse=w, img_per_sec=100.0, nonfinite_steps=0)) + '\n')
    info = pc.run_info(d)
    assert info['loss_spikes'] == 0 and not info['unstable'], info
    assert pc.count_spikes(wmse[:5] + [0.75] + wmse[5:]) == 1
    assert pc.count_spikes([1.0, 0.9, 0.8, 0.85, 0.7]) == 0

def test_learning_rate_rule():
    lo, hi = _arm('E112', 0.0045, fid=5.0, wmse=1.000), _arm('E112', 0.009, fid=4.9, wmse=0.995)
    d = pc.decide([lo, hi])
    assert d['winner']['ref_lr'] == 0.0045
    hi['final']['heldout_wmse_mean'] = 0.95
    assert pc.decide([lo, hi])['winner']['ref_lr'] == 0.009
    hi['unstable'] = True
    assert pc.decide([lo, hi])['winner']['ref_lr'] == 0.0045

def test_family_rule_needs_the_half_rate_arm():
    edm2 = [_arm('E112', 0.0045, fid=5.0), _arm('E112', 0.009, fid=5.2, wmse=1.1)]
    ddpm = _arm('DDPMpp128', 2e-4, fid=4.0)
    d = pc.decide(edm2 + [ddpm])
    assert d['rerun_half_lr'] and d['winner']['preset'] == 'E112'
    d = pc.decide(edm2 + [_arm('E112', 0.00225, fid=4.95, wmse=1.05), ddpm])
    assert not d['rerun_half_lr'] and d['winner']['preset'] == 'DDPMpp128'
    d = pc.decide(edm2 + [_arm('E112', 0.00225, fid=4.1, wmse=1.05), ddpm])
    assert not d['rerun_half_lr'] and d['winner']['preset'] == 'E112'
    d = pc.decide(edm2 + [_arm('DDPMpp128', 2e-4, fid=4.9)])
    assert not d['rerun_half_lr'] and d['winner']['preset'] == 'E112'

def test_layout_rule():
    e112, e128 = _arm('E112', 0.0045, fid=5.0), _arm('E128-1233', 0.0042, fid=4.5)
    assert pc.decide([e112, e128])['winner']['preset'] == 'E128-1233'
    e128['final']['fid'] = 4.85
    assert pc.decide([e112, e128])['winner']['preset'] == 'E112'
    e144 = _arm('E144-1223', 0.0040, fid=4.6)
    assert pc.decide([e112, e128, e144])['winner']['preset'] == 'E144-1223'
    e128['final']['fid'] = 4.4
    d = pc.decide([e112, e128, e144])
    assert d['winner']['preset'] == 'E128-1233' and d['winner']['ref_lr'] == 0.0042
