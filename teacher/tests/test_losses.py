import numpy as np
import torch

from teacher import losses, networks, official

def _net(family):
    torch.manual_seed(0)
    if family == 'edm2':
        kw = dict(img_resolution=64, img_channels=3, label_dim=151, model_channels=16, channel_mult=[1, 2, 2], num_blocks=1,
                  attn_resolutions=[16], channels_per_head=16, augment_dim=9, sigma_data=0.4)
        net = networks.build_network(dict(family='edm2', kwargs=kw)).eval()
        with torch.no_grad():
            for p in net.parameters():
                if p.ndim == 0:
                    p.fill_(0.7)
        return net
    kw = dict(img_resolution=64, img_channels=3, label_dim=151, model_channels=16, channel_mult=[1, 2, 2], num_blocks=1,
              attn_resolutions=[16], augment_dim=9, sigma_data=0.4, use_fp16=False)
    net = networks.build_network(dict(family='edm', kwargs=kw)).eval()
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.05)
    return net

def test_edm2_loss_matches_official():
    net = _net('edm2')
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    y = net.one_hot(torch.tensor([0, 5, 151, 9]))
    P_mean, P_std, sd = -1.05, 1.6, 0.4
    torch.manual_seed(42)
    ref = official.EDM2Loss(P_mean=P_mean, P_std=P_std, sigma_data=sd)(net, x, y)
    torch.manual_seed(42)
    sigma = (torch.randn([4, 1, 1, 1]) * P_std + P_mean).exp()
    ours, info = losses.edm2_loss(net, x, y, sigma, sd)
    assert torch.equal(ref, ours)
    assert info['loss'].shape == (4,) and info['sigma'].shape == (4,)

def test_edm_loss_matches_official():
    net = _net('edm')
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    y = net.one_hot(torch.tensor([0, 5, 151, 9]))
    P_mean, P_std, sd = -1.45, 1.2, 0.4
    torch.manual_seed(7)
    ref = official.EDMLoss(P_mean=P_mean, P_std=P_std, sigma_data=sd)(net, x, y)
    torch.manual_seed(7)
    sigma = (torch.randn([4, 1, 1, 1]) * P_std + P_mean).exp()
    ours, _ = losses.edm_loss(net, x, y, sigma, sd)
    assert torch.equal(ref, ours)

def test_stratified_sigma():
    torch.manual_seed(0)
    z = losses.stratified_normal(256, torch.device('cpu'))
    u = torch.special.ndtr(z.double()).sort().values
    k = torch.arange(256, dtype=torch.float64)
    assert torch.all(u >= k / 256 - 1e-6) and torch.all(u < (k + 1) / 256 + 1e-6)
    assert not torch.equal(z, z.sort().values)
    s = losses.sample_sigma(4096, -0.8, 1.6, torch.device('cpu'))
    assert abs(float(s.log().mean()) + 0.8) < 0.02 and abs(float(s.log().std()) - 1.6) < 0.03

def test_learning_rate_schedule():
    for t in [0, 1, 2500, 5000, 20000, 35000, 100000, 600000]:
        lr = official.learning_rate_schedule(cur_nimg=t * 256, batch_size=256, ref_lr=0.0045, ref_batches=35000, rampup_Mimg=5000 * 256 / 1e6)
        expect = 0.0045 * min(t / 5000, 1) / np.sqrt(max(t / 35000, 1))
        assert abs(lr - expect) <= 1e-12, (t, lr, expect)
    lr = official.learning_rate_schedule(cur_nimg=10 ** 9, batch_size=256, ref_lr=2e-4, ref_batches=0, rampup_Mimg=1.28)
    assert lr == 2e-4

def test_bucket_stats():
    b = losses.BucketStats(torch.device('cpu'))
    info = dict(sigma=torch.tensor([0.01, 0.1, 0.3, 0.7, 1.5, 3.0, 10.0, 50.0]), loss=torch.arange(8.0), wmse=torch.ones(8), logvar=torch.zeros(8))
    b.add(info)
    s = b.summary()
    assert len(s['by_sigma']) == 8 and abs(s['loss'] - 3.5) < 1e-9
