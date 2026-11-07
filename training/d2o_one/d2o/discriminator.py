import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

NFC = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64}


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


class FullyConnectedLayer(nn.Module):

    def __init__(self, in_features, out_features, activation="linear"):
        super().__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.randn([out_features, in_features]))
        self.bias = nn.Parameter(torch.zeros([out_features]))
        self.weight_gain = 1 / np.sqrt(in_features)

    def forward(self, x):
        x = F.linear(x, self.weight * self.weight_gain, self.bias)
        if self.activation == "lrelu":
            x = F.leaky_relu(x, 0.2) * np.sqrt(2)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


class SingleDiscCond(nn.Module):

    def __init__(self, nc, start_sz, end_sz=8, c_dim=151, cmap_dim=64, embed_dim=128, width=1.0):
        super().__init__()
        self.cmap_dim = cmap_dim
        nfc = {size: int(round(channels * width)) for size, channels in NFC.items()}
        nfc[start_sz] = nc

        layers = []
        while start_sz > end_sz:
            layers.append(DownBlock(nfc[start_sz], nfc[start_sz // 2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)
        kernel = 4 if end_sz >= 8 else max(1, end_sz // 2)
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, kernel, 1, 0, bias=False)

        self.embed = nn.Embedding(c_dim, embed_dim)
        self.embed_proj = FullyConnectedLayer(embed_dim, self.cmap_dim, activation="lrelu")

    def forward(self, x, c):
        out = self.cls(self.main(x))
        cmap = self.embed_proj(self.embed(c)).unsqueeze(-1).unsqueeze(-1)
        return (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))


class MultiScaleD(nn.Module):

    def __init__(self, channels, resolutions, c_dim=151, width=1.0):
        super().__init__()
        end_sz = 8 if resolutions[-1] >= 8 else resolutions[-1]
        self.mini_discs = nn.ModuleDict({
            str(i): SingleDiscCond(nc=cin, start_sz=res, end_sz=end_sz, c_dim=c_dim, width=width)
            for i, (cin, res) in enumerate(zip(channels, resolutions))
        })

    def forward(self, features, c):
        return torch.cat([disc(features[k], c).flatten(1) for k, disc in self.mini_discs.items()], dim=1)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, expand=False):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features // 2 if expand else features, kernel_size=1, bias=True)

    def forward(self, *xs):
        output = xs[0] if len(xs) == 1 else xs[0] + xs[1]
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class EDMFeatureNetwork(nn.Module):

    def __init__(self, edm, proj_type=2, cout=64, expand=True, trainable_backbone=False):
        super().__init__()
        assert proj_type in [0, 1, 2]
        self.proj_type = proj_type
        self.edm = copy.deepcopy(edm)
        del self.edm.model.dec

        self.RESOLUTIONS = sorted(self.edm.model.feature_channels, reverse=True)
        self.CHANNELS = [self.edm.model.feature_channels[res] for res in self.RESOLUTIONS]
        assert len(self.RESOLUTIONS) == 4

        if proj_type >= 1:
            ccm_channels = [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4
            self.ccm = nn.ModuleList([nn.Conv2d(cin, c, kernel_size=1, bias=True) for cin, c in zip(self.CHANNELS, ccm_channels)])
            self.CHANNELS = ccm_channels
        if proj_type == 2:
            self.csm = nn.ModuleList([
                FeatureFusionBlock(ccm_channels[0]),
                FeatureFusionBlock(ccm_channels[1], expand=expand),
                FeatureFusionBlock(ccm_channels[2], expand=expand),
                FeatureFusionBlock(ccm_channels[3], expand=expand),
            ])
            self.CHANNELS = [cout, cout, cout * 2, cout * 4] if expand else [cout] * 4
            self.RESOLUTIONS = [res * 2 for res in self.RESOLUTIONS]
        self.eval().requires_grad_(False)
        self.edm.requires_grad_(trainable_backbone)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x, sigma, class_labels, force_fp32=False):
        feats = self.edm.features(x, sigma, class_labels, force_fp32=force_fp32)
        out = [feats[res] for res in sorted(feats, reverse=True)]
        if self.proj_type == 0:
            return {str(i): feat for i, feat in enumerate(out)}

        out = [ccm(feat) for ccm, feat in zip(self.ccm, out)]
        if self.proj_type == 1:
            return {str(i): feat for i, feat in enumerate(out)}

        out3 = self.csm[3](out[3])
        out2 = self.csm[2](out3, out[2])
        out1 = self.csm[1](out2, out[1])
        out0 = self.csm[0](out1, out[0])
        return {"0": out0, "1": out1, "2": out2, "3": out3}


class EDM2FeatureNetwork(nn.Module):

    def __init__(self, encoder, cout=64, shared=False, live=False):
        super().__init__()
        self.live = live
        if shared:
            object.__setattr__(self, "edm", encoder)
        else:
            self.edm = encoder
        resolutions = sorted(encoder.feature_channels, reverse=True)
        ccm_channels = [cout * 2 ** level for level in range(len(resolutions))]
        self.ccm = nn.ModuleList([nn.Conv2d(encoder.feature_channels[res], c, kernel_size=1, bias=True) for res, c in zip(resolutions, ccm_channels)])
        self.csm = nn.ModuleList([FeatureFusionBlock(c, expand=level > 0) for level, c in enumerate(ccm_channels)])
        self.CHANNELS = [ccm_channels[0]] + [c // 2 for c in ccm_channels[1:]]
        self.RESOLUTIONS = [res * 2 for res in resolutions]
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x, sigma, class_labels, force_fp32=False):
        features = self.edm.features_fixed if self.live else self.edm.features
        feats = features(x, sigma, class_labels, force_fp32=force_fp32)
        out = [ccm(feats[res]) for ccm, res in zip(self.ccm, sorted(feats, reverse=True))]
        fused = [None] * len(out)
        for level in reversed(range(len(out))):
            fused[level] = self.csm[level](out[level]) if level == len(out) - 1 else self.csm[level](fused[level + 1], out[level])
        return {str(level): feat for level, feat in enumerate(fused)}


class ProjectedDiscriminator(nn.Module):

    def __init__(self, edm, c_dim=151, proj_type=2, sigma_mean=-2.0, sigma_std=1.0, sigma_min=0.02, sigma_max=80.0, trainable_backbone=False,
                 feature_network=None, width=1.0):
        super().__init__()
        self.c_dim = c_dim
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if feature_network is None:
            feature_network = EDMFeatureNetwork(edm, proj_type=proj_type, cout=int(round(64 * width)), trainable_backbone=trainable_backbone)
        self.feature_network = feature_network
        self.heads = MultiScaleD(self.feature_network.CHANNELS, self.feature_network.RESOLUTIONS, c_dim=c_dim, width=width)

    def sample_sigma(self, batch_size, device):
        sigma = (torch.randn([batch_size, 1, 1, 1], device=device) * self.sigma_std + self.sigma_mean).exp()
        return sigma.clamp(self.sigma_min, self.sigma_max)

    def extract(self, images, c, force_fp32=False):
        sigma = self.sample_sigma(images.shape[0], images.device)
        noisy = images + sigma * torch.randn_like(images)
        return self.feature_network(noisy, sigma, F.one_hot(c, self.c_dim).to(torch.float32), force_fp32=force_fp32)

    def forward(self, images, c, force_fp32=False):
        return self.heads(self.extract(images, c, force_fp32=force_fp32), c)
