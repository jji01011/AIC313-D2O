import torch

from . import official

edm_sampler = official.edm_sampler
StackedRandomGenerator = official.StackedRandomGenerator

DEFAULT_SAMPLER = dict(num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1)

def finalize(x):
    return (x.clamp(-1, 1) + 1 / 255).clamp(-1, 1)

class _NullClassGuide:
    def __init__(self, net):
        self.net = net

    def __call__(self, x, t, labels=None):
        n = x.shape[0]
        return self.net(x, t, self.net.null_labels(n, x.device))

class _LabelledGuide:
    def __init__(self, gnet):
        self.gnet = gnet

    def __call__(self, x, t, labels=None):
        return self.gnet(x, t, labels)

@torch.no_grad()
def sample(net, classes, seeds, guidance=1.0, guide='none', gnet=None, batch_size=256, device=None, sampler_kwargs=None, progress=None):
    device = device or next(net.parameters()).device
    kw = dict(DEFAULT_SAMPLER)
    kw.update(sampler_kwargs or {})
    classes = torch.as_tensor(classes, dtype=torch.long)
    seeds = [int(s) for s in seeds]
    assert len(classes) == len(seeds)
    if guidance == 1 or guide == 'none':
        guidance, guide_fn = 1, None
    elif guide == 'cfg':
        guide_fn = _NullClassGuide(net)
    elif guide == 'autoguidance':
        assert gnet is not None
        guide_fn = _LabelledGuide(gnet)
    else:
        raise ValueError(guide)
    mean = net.mean_image.to(device=device, dtype=torch.float32)
    out = []
    for start in range(0, len(seeds), batch_size):
        b_seeds = seeds[start : start + batch_size]
        b_classes = classes[start : start + batch_size].to(device)
        rnd = StackedRandomGenerator(device, b_seeds)
        noise = rnd.randn([len(b_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        labels = net.one_hot(b_classes)
        x = edm_sampler(net, noise + mean / kw['sigma_max'], labels=labels, gnet=guide_fn, guidance=guidance,
                        randn_like=rnd.randn_like, **kw)
        out.append(finalize(x).cpu())
        if progress is not None:
            progress(min(start + batch_size, len(seeds)), len(seeds))
    return torch.cat(out)

def seeds_for(run_seed, n):
    return [int(run_seed) * 1_000_003 + i for i in range(n)]

@torch.no_grad()
def save_grid(images, path, nrow=8):
    import torchvision
    from .data import export_png
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    export_png(grid, path)
