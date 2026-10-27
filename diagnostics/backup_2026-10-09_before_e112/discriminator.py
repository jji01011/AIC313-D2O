"""Projected discriminator for D2O without external pre-trained networks.

The official D2O discriminator (``pg_modules/`` in the D2O repository, taken
from Projected GAN / StyleGAN-XL: Sauer et al., NeurIPS 2021 and SIGGRAPH 2022)
feeds frozen ImageNet-pretrained feature networks into small trainable
multi-scale heads. ImageNet weights are not allowed in this project, so the
frozen feature network is replaced by the encoder of the EDM that stage 0
trains from scratch on the Pokemon train split. Everything downstream of the
features is ported from the official code: the frozen random cross-channel /
cross-scale mixing (CCM / CSM), the spectral-norm ``DownBlock`` heads and the
projection conditioning, with the class embedding learned from scratch (the
official ``rand_embedding`` path).

Because the EDM encoder was trained on noisy inputs, images are perturbed to a
noise level ``sigma`` before feature extraction and the encoder is conditioned
on that level.
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

# Channel widths of the official heads (``nfc_midas``), indexed by feature-map size.
NFC = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64}


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


class FullyConnectedLayer(nn.Module):
    """StyleGAN2 fully-connected layer with equalized learning rate."""

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
    """One projection-conditioned head operating on one feature scale."""

    def __init__(self, nc, start_sz, end_sz=8, c_dim=151, cmap_dim=64, embed_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim
        nfc = dict(NFC)
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
    """Trainable part of the discriminator: one head per feature scale."""

    def __init__(self, channels, resolutions, c_dim=151):
        super().__init__()
        end_sz = 8 if resolutions[-1] >= 8 else resolutions[-1]
        self.mini_discs = nn.ModuleDict({
            str(i): SingleDiscCond(nc=cin, start_sz=res, end_sz=end_sz, c_dim=c_dim)
            for i, (cin, res) in enumerate(zip(channels, resolutions))
        })

    def forward(self, features, c):
        """Return logits ``[N, num_logits]`` for class indices ``c``."""
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
    """Frozen feature network: stage-0 EDM encoder plus frozen random projections.

    ``proj_type``: 0 = raw encoder features, 1 = cross-channel mixing,
    2 = cross-channel and cross-scale mixing (official setting).
    ``trainable_backbone`` lets the encoder be fine-tuned together with the
    heads (an ablation; the official feature networks stay frozen).
    """

    def __init__(self, edm, proj_type=2, cout=64, expand=True, trainable_backbone=False):
        super().__init__()
        assert proj_type in [0, 1, 2]
        self.proj_type = proj_type
        self.edm = copy.deepcopy(edm)
        del self.edm.model.dec  # only the encoder is evaluated

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
            self.RESOLUTIONS = [res * 2 for res in self.RESOLUTIONS]  # CSM upsamples x2
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

        # From bottom to top.
        out3 = self.csm[3](out[3])
        out2 = self.csm[2](out3, out[2])
        out1 = self.csm[1](out2, out[1])
        out0 = self.csm[0](out1, out[0])
        return {"0": out0, "1": out1, "2": out2, "3": out3}


class EDM2FeatureNetwork(nn.Module):
    """Frozen feature network from the encoder of an EDM2 denoiser (``d2o/edm2.py``).

    Same frozen random cross-channel / cross-scale mixing as ``EDMFeatureNetwork`` (``proj_type=2``), written for
    however many resolutions the encoder provides: a large denoiser can be cut after its 16x16 blocks so that the
    whole discriminator stays inside the parameter budget.
    """

    def __init__(self, encoder, cout=64):
        super().__init__()
        self.edm = encoder
        resolutions = sorted(encoder.feature_channels, reverse=True)
        ccm_channels = [cout * 2 ** level for level in range(len(resolutions))]
        self.ccm = nn.ModuleList([nn.Conv2d(encoder.feature_channels[res], c, kernel_size=1, bias=True) for res, c in zip(resolutions, ccm_channels)])
        self.csm = nn.ModuleList([FeatureFusionBlock(c, expand=level > 0) for level, c in enumerate(ccm_channels)])
        self.CHANNELS = [ccm_channels[0]] + [c // 2 for c in ccm_channels[1:]]
        self.RESOLUTIONS = [res * 2 for res in resolutions]  # CSM upsamples x2
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x, sigma, class_labels, force_fp32=False):
        feats = self.edm.features(x, sigma, class_labels, force_fp32=force_fp32)
        out = [ccm(feats[res]) for ccm, res in zip(self.ccm, sorted(feats, reverse=True))]
        fused = [None] * len(out)
        for level in reversed(range(len(out))):  # from bottom to top
            fused[level] = self.csm[level](out[level]) if level == len(out) - 1 else self.csm[level](fused[level + 1], out[level])
        return {str(level): feat for level, feat in enumerate(fused)}


class ProjectedDiscriminator(nn.Module):
    """Frozen feature network (``EDMFeatureNetwork`` unless one is passed in) followed by trainable ``MultiScaleD`` heads."""

    def __init__(self, edm, c_dim=151, proj_type=2, sigma_mean=-2.0, sigma_std=1.0, sigma_min=0.02, sigma_max=80.0, trainable_backbone=False,
                 feature_network=None):
        super().__init__()
        self.c_dim = c_dim
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        if feature_network is None:
            feature_network = EDMFeatureNetwork(edm, proj_type=proj_type, trainable_backbone=trainable_backbone)
        self.feature_network = feature_network
        self.heads = MultiScaleD(self.feature_network.CHANNELS, self.feature_network.RESOLUTIONS, c_dim=c_dim)

    def sample_sigma(self, batch_size, device):
        """Log-normal noise level per image; ``sigma_std=0`` gives the fixed level ``exp(sigma_mean)``."""
        sigma = (torch.randn([batch_size, 1, 1, 1], device=device) * self.sigma_std + self.sigma_mean).exp()
        return sigma.clamp(self.sigma_min, self.sigma_max)

    def extract(self, images, c, force_fp32=False):
        """Noise the images and return the multi-scale features that the heads operate on."""
        sigma = self.sample_sigma(images.shape[0], images.device)
        noisy = images + sigma * torch.randn_like(images)
        return self.feature_network(noisy, sigma, F.one_hot(c, self.c_dim).to(torch.float32), force_fp32=force_fp32)

    def forward(self, images, c, force_fp32=False):
        return self.heads(self.extract(images, c, force_fp32=force_fp32), c)
