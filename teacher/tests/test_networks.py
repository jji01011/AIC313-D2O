import copy

import torch

from teacher import networks, official, presets

def _small_edm2(**kw):
    args = dict(img_resolution=64, img_channels=3, label_dim=151, model_channels=32, channel_mult=[1, 2, 2], num_blocks=1,
                attn_resolutions=[16], channels_per_head=32)
    args.update(kw)
    return args

def _nonzero_gains_(net):
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.ndim == 0 and 'emb_extra_gain' not in name:
                p.fill_(0.7)

def test_param_counts():
    for name in ['E112', 'E128-1233', 'E144-1223', 'E96', 'E80', 'E64']:
        net = networks.build_network(presets.network_config(name, dropout=0.1))
        c = networks.count_parameters(net)
        assert c['trainable'] == presets.NETWORK_PRESETS[name]['params'], (name, c)
        assert c['frozen'] == 3 * 64 * 64, (name, c)
        assert c['total'] <= 100_000_000
        print(f"  {name}: {c['trainable']:,} trainable + {c['frozen']:,} frozen = {c['total']:,}")
    net = networks.build_network(presets.network_config('DDPMpp128', dropout=0.1))
    c = networks.count_parameters(net)
    assert c['total'] <= 100_000_000
    print(f"  DDPMpp128: {c['trainable']:,} trainable + {c['frozen']:,} frozen = {c['total']:,}")

def test_e112_headroom():
    net = networks.build_network(presets.network_config('E112', dropout=0.1))
    c = networks.count_parameters(net)
    assert c['total'] == 95_175_157 and 100_000_000 - c['total'] == 4_824_843

def test_matches_official_precond():
    torch.manual_seed(1)
    ours = networks.EDM2Precond(**_small_edm2(), null_class=False, forced_wn='forward')
    torch.manual_seed(1)
    off = official.networks_edm2.Precond(**_small_edm2())
    sd_o, sd_u = off.state_dict(), ours.state_dict()
    assert set(sd_u) - set(sd_o) == {'mean_image'} and not set(sd_o) - set(sd_u)
    assert all(torch.equal(sd_o[k], sd_u[k]) for k in sd_o), 'initialisation differs from the official one'
    _nonzero_gains_(ours), _nonzero_gains_(off)
    x = torch.randn(4, 3, 64, 64)
    s = torch.tensor([0.01, 0.5, 3.0, 60.0])
    y = torch.nn.functional.one_hot(torch.tensor([0, 7, 150, 3]), 151).float()
    ours.eval(), off.eval()
    with torch.no_grad():
        a, la = ours(x, s, y, return_logvar=True)
        b, lb = off(x, s, y, return_logvar=True)
    assert torch.equal(a, b) and torch.equal(la, lb) and float((a - x * 0.25 / (s.reshape(-1, 1, 1, 1) ** 2 + 0.25)).abs().max()) > 0.1

def test_official_init_order_with_additions():
    torch.manual_seed(3)
    ours = networks.EDM2Precond(**_small_edm2(), augment_dim=9)
    torch.manual_seed(3)
    off = official.networks_edm2.Precond(**dict(_small_edm2(), label_dim=152))
    sd_o, sd_u = off.state_dict(), ours.state_dict()
    keys = [k for k in sd_o if k.startswith('unet.')]
    assert len(keys) > 50
    for k in keys:
        assert torch.equal(sd_o[k], sd_u[k]), k

def test_mean_centring_and_null_class():
    torch.manual_seed(0)
    net = networks.build_network(dict(family='edm2', kwargs=_small_edm2(augment_dim=9)))
    net.eval()
    _nonzero_gains_(net)
    x = torch.randn(3, 3, 64, 64)
    s = torch.tensor([0.1, 1.0, 10.0])
    y = net.one_hot(torch.tensor([1, 2, 151]))
    with torch.no_grad():
        d0 = net(x, s, y)
        mu = torch.randn(3, 64, 64) * 0.3
        net.mean_image.copy_(mu)
        d1 = net(x + mu, s, y)
        assert torch.allclose(d1 - mu, d0, atol=1e-5), 'D(x + mu) should be mu + D_0(x)'
        assert torch.allclose(net(x, s), net(x, s, net.null_labels(3, x.device)), atol=1e-6)
        assert torch.equal(net(x, s, y), net(x, s, y, augment_labels=torch.zeros(3, 9)))
    assert net.label_dim == 152 and net.num_classes == 151

def test_forced_wn_post_step_matches_official():
    torch.manual_seed(0)
    a = networks.EDM2Precond(**_small_edm2(augment_dim=9), forced_wn='forward')
    b = copy.deepcopy(a)
    b.forced_wn = 'post_step'
    b.init_weights_()
    a.train(), b.train()
    ema_a = official.PowerFunctionEMA(a, stds=[0.05, 0.10])
    ema_b = official.PowerFunctionEMA(b, stds=[0.05, 0.10])
    opt_a = torch.optim.Adam([p for p in a.parameters() if p.requires_grad], lr=1e-2, betas=(0.9, 0.99))
    opt_b = torch.optim.Adam([p for p in b.parameters() if p.requires_grad], lr=1e-2, betas=(0.9, 0.99))
    for step in range(20):
        g = torch.Generator().manual_seed(step)
        x = torch.randn(8, 3, 64, 64, generator=g) * 0.5
        y = torch.nn.functional.one_hot(torch.randint(0, 152, (8,), generator=g), 152).float()
        sigma = torch.exp(torch.randn(8, generator=g) * 1.6 - 0.8).reshape(-1, 1, 1, 1)
        n = torch.randn(8, 3, 64, 64, generator=g) * sigma
        losses = []
        for net, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad(set_to_none=True)
            D, logvar = net(x + n, sigma, y, return_logvar=True)
            w = (sigma ** 2 + 0.25) / (sigma * 0.5) ** 2
            loss = ((w / logvar.exp()) * (D - x) ** 2 + logvar).sum() / 8
            loss.backward()
            losses.append(loss.item())
            if net is b:
                grads_b = [p.grad.clone() for p in b.parameters() if p.requires_grad]
            else:
                grads_a = [p.grad.clone() for p in a.parameters() if p.requires_grad]
        assert abs(losses[0] - losses[1]) <= 1e-4 * abs(losses[0]) + 1e-5, (step, losses)
        for ga, gb in zip(grads_a, grads_b):
            assert torch.allclose(ga, gb, rtol=1e-3, atol=1e-6 * max(1.0, float(ga.abs().max()))), step
        opt_a.step(), opt_b.step()
        ema_a.update(cur_nimg=(step + 1) * 8, batch_size=8)
        ema_b.update(cur_nimg=(step + 1) * 8, batch_size=8)
        b.post_step_()
    for ea, eb in zip(ema_a.emas, ema_b.emas):
        for (k, pa), pb in zip(ea.named_parameters(), eb.parameters()):
            assert torch.allclose(pa, pb, rtol=1e-3, atol=1e-5 * max(1.0, float(pa.abs().max()))), k

def test_ema_deepcopy_and_update():
    torch.manual_seed(0)
    net = networks.build_network(dict(family='edm2', kwargs=_small_edm2(augment_dim=9)))
    ema = official.PowerFunctionEMA(net, stds=[0.05, 0.10])
    net.eval()
    for e in ema.emas:
        e.eval()
        assert type(e.unet.enc['64x64_conv']).__name__ == 'MPConv'
        assert not e.mean_image.requires_grad
    x, s = torch.randn(2, 3, 64, 64), torch.tensor([0.5, 5.0])
    y = net.one_hot(torch.tensor([4, 151]))
    with torch.no_grad():
        for e, _ in ema.get():
            assert torch.equal(e(x, s, y), net(x, s, y))
        for p in net.parameters():
            if p.requires_grad:
                p.add_(0.01)
        ema.update(cur_nimg=256, batch_size=256)
        for e, _ in ema.get():
            assert torch.allclose(e(x, s, y), net(x, s, y))

def test_dropout_placement():
    net = networks.build_network(presets.network_config('E112', dropout=0.15))
    d = net.get_dropout()
    assert all(v == 0.0 for k, v in d.items() if k.startswith('64x64')), d
    assert all(v == 0.15 for k, v in d.items() if not k.startswith('64x64')), d
    net.set_dropout(presets.dropout_table('edm2', 0.2, [1, 2, 3, 4]))
    assert max(net.get_dropout().values()) == 0.2

def test_ddpmpp_matches_official():
    kw = dict(img_resolution=64, img_channels=3, model_channels=16, channel_mult=[1, 2, 2], num_blocks=1, attn_resolutions=[16],
              embedding_type='positional', encoder_type='standard', decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
              augment_dim=9)
    torch.manual_seed(5)
    ours = networks.DDPMppPrecond(label_dim=151, null_class=False, **kw)
    torch.manual_seed(5)
    off = official.edm_networks.EDMPrecond(label_dim=151, model_type='SongUNet', **kw)
    with torch.no_grad():
        for p in off.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ours.load_state_dict(dict(off.state_dict(), mean_image=torch.zeros(3, 64, 64)))
    ours.eval(), off.eval()
    x, s = torch.randn(2, 3, 64, 64), torch.tensor([0.3, 7.0])
    y = torch.nn.functional.one_hot(torch.tensor([3, 9]), 151).float()
    aug = torch.zeros(2, 9)
    with torch.no_grad():
        a, b = ours(x, s, y, augment_labels=aug), off(x, s, y, augment_labels=aug)
    assert torch.equal(a, b) and float((a - x * 0.25 / (s.reshape(-1, 1, 1, 1) ** 2 + 0.25)).abs().max()) > 0.01

def test_sdpa_matches_einsum():
    torch.manual_seed(0)
    net = networks.build_network(dict(family='edm2', kwargs=_small_edm2(augment_dim=9))).eval()
    _nonzero_gains_(net)
    x, s = torch.randn(3, 3, 64, 64), torch.tensor([0.1, 1.0, 10.0])
    y = net.one_hot(torch.tensor([1, 2, 151]))
    with torch.no_grad():
        a = net(x, s, y)
        net.set_attention('sdpa')
        b = net(x, s, y)
        net.set_attention('einsum')
        assert torch.equal(net(x, s, y), a)
    assert float(a.abs().max()) > 0.1 and torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
    e = copy.deepcopy(net)
    e.set_attention('sdpa')
    assert e.unet.enc['16x16_block0'].attention_impl == 'sdpa' and net.unet.enc['16x16_block0'].attention_impl == 'einsum'

def test_student_hooks():
    torch.manual_seed(0)
    net = networks.build_network(dict(family='edm2', kwargs=_small_edm2(augment_dim=9))).eval()
    _nonzero_gains_(net)
    x, s = torch.randn(3, 3, 64, 64), torch.tensor([0.1, 1.0, 10.0])
    y = net.one_hot(torch.tensor([1, 2, 151]))
    torch.manual_seed(1)
    stu = networks.build_network(dict(family='edm2', kwargs=_small_edm2(augment_dim=9, extra_embeddings=['time2', 'guidance']))).eval()
    missing = networks.load_teacher_state_dict(stu, net.state_dict())
    assert missing and all(k.startswith('unet.emb_extra') for k in missing)
    extra = dict(time2=torch.tensor([0.05, 0.5, 5.0]).log() / 4, guidance=torch.tensor([1.0, 1.5, 2.0]))
    with torch.no_grad():
        a = net(x, s, y)
        d, logvar, enc = net(x, s, y, return_logvar=True, return_enc=True)
        assert torch.equal(d, a) and enc.shape == (3, 64, 16, 16) and logvar.shape == (3, 1, 1, 1)
        assert torch.equal(stu(x, s, y, extra_labels=extra), a)
        stu.unet.emb_extra_gain['time2'].fill_(1.0)
        assert not torch.allclose(stu(x, s, y, extra_labels=extra), a)
        assert torch.equal(stu(x, s, y), a)
        net.c_noise_fn = lambda sigma: sigma.flatten().log() / 4
        assert torch.equal(net(x, s, y), a)
    big = networks.build_network(presets.network_config('E112', dropout=0.1))
    big_slots = networks.EDM2Precond(**dict(presets.network_config('E112', dropout=0.1)['kwargs'], extra_embeddings=['time2', 'guidance']))
    c0, c1 = networks.count_parameters(big), networks.count_parameters(big_slots)
    assert c1['trainable'] - c0['trainable'] == 100_354 and c1['trainable'] == 95_263_223, c1
    with torch.no_grad():
        _d, enc = big.eval()(torch.randn(1, 3, 64, 64), torch.tensor([1.0]), big.one_hot(torch.tensor([3])), return_enc=True)
    assert enc.shape == (1, 448, 8, 8)
