import torch

from . import data as data_mod
from . import losses
from . import sampler as sampler_mod

class MemorisationGuard:
    def __init__(self, prep, train_indices, device, sigma_data, widen_checks=3, copy_rate_threshold=0.05,
                 copy_ratio=1 / 3, sample_steps=32, sample_batch=256, max_images=None):
        splits = prep['splits']
        images, labels = prep['images'], prep['labels']
        self.device = device
        self.sigma_data = float(sigma_data)
        canary, probe = splits['canary'][:max_images], splits['probe'][:max_images]
        self.canary = (data_mod.u8_to_float(images[canary]).to(device), labels[canary].to(device))
        self.probe = (data_mod.u8_to_float(images[probe]).to(device), labels[probe].to(device))
        self.train_u8 = images[train_indices]
        self.num_classes = int(prep['stats']['num_classes'])
        self.widen_checks = widen_checks
        self.copy_rate_threshold = copy_rate_threshold
        self.copy_ratio = copy_ratio
        self.sample_kwargs = dict(num_steps=sample_steps)
        self.sample_batch = sample_batch

    @torch.no_grad()
    def denoising_gap(self, net):
        c = losses.fixed_sigma_wmse(net, self.canary[0], net.one_hot(self.canary[1]), self.sigma_data, seed=1)
        p = losses.fixed_sigma_wmse(net, self.probe[0], net.one_hot(self.probe[1]), self.sigma_data, seed=1)
        gap = {s: c[s] - p[s] for s in c}
        low = [s for s in gap if s <= 1.0]
        return dict(canary=c, probe=p, gap=gap, gap_low=sum(gap[s] for s in low) / len(low),
                    rel_gap_low=sum(gap[s] / max(p[s], 1e-12) for s in low) / len(low))

    @torch.no_grad()
    def copy_test(self, net, seed=12345):
        classes = torch.arange(self.num_classes)
        seeds = sampler_mod.seeds_for(seed, self.num_classes)
        x = sampler_mod.sample(net, classes, seeds, batch_size=self.sample_batch, device=self.device, sampler_kwargs=self.sample_kwargs)
        x = x.to(self.device).flatten(1).to(torch.float32)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        best = torch.full([x.shape[0], 2], float('inf'), device=self.device)
        for s in range(0, self.train_u8.shape[0], 2048):
            t = data_mod.u8_to_float(self.train_u8[s : s + 2048].to(self.device)).flatten(1)
            d = (x_sq + (t ** 2).sum(dim=1)[None] - 2 * x @ t.T).clamp_min(0)
            best = torch.cat([best, d], dim=1).topk(2, dim=1, largest=False).values
        d1, d2 = best[:, 0].sqrt(), best[:, 1].sqrt()
        ratio = d1 / d2.clamp_min(1e-12)
        copies = ratio < self.copy_ratio
        return dict(copy_rate=float(copies.float().mean()), num_copies=int(copies.sum()),
                    ratio_median=float(ratio.median()), ratio_min=float(ratio.min()), d1_median=float(d1.median()))

    def check(self, net):
        out = self.denoising_gap(net)
        out.update(self.copy_test(net))
        return out

    def should_raise_dropout(self, history):
        if len(history) < self.widen_checks:
            return False
        recent = [h['gap_low'] for h in history[-self.widen_checks:]]
        widening = all(b > a for a, b in zip(recent[:-1], recent[1:]))
        return widening and history[-1]['copy_rate'] > self.copy_rate_threshold
