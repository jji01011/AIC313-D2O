"""Adaptive discriminator augmentation (ADA) in plain PyTorch.

Follows the ``AugmentPipe`` of StyleGAN2-ADA (Karras et al., "Training
Generative Adversarial Networks with Limited Data", NeurIPS 2020), which the
official D2O repository ships as ``training/augment.py``. The pixel blitting,
geometric and color transformations and their probabilities are the same; the
geometric warp uses bilinear ``grid_sample`` with reflection padding instead of
the wavelet-filtered resampling, which needs custom CUDA ops.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def matrix(*rows, device=None):
    """Build a batch of matrices from scalars and ``[N]`` tensors."""
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=device)
    elems = [x if isinstance(x, torch.Tensor) else torch.full(ref[0].shape, float(x), device=ref[0].device) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translate2d(tx, ty, **kwargs):
    return matrix([1, 0, tx], [0, 1, ty], [0, 0, 1], **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix([sx, 0, 0], [0, sy, 0], [0, 0, 1], **kwargs)


def rotate2d(theta, **kwargs):
    return matrix([torch.cos(theta), torch.sin(-theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1], **kwargs)


def translate3d(tx, ty, tz, **kwargs):
    return matrix([1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1], **kwargs)


def scale3d(sx, sy, sz, **kwargs):
    return matrix([sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1], **kwargs)


def rotate3d(v, theta, **kwargs):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    s, c = torch.sin(theta), torch.cos(theta)
    cc = 1 - c
    return matrix(
        [vx * vx * cc + c, vx * vy * cc - vz * s, vx * vz * cc + vy * s, 0],
        [vy * vx * cc + vz * s, vy * vy * cc + c, vy * vz * cc - vx * s, 0],
        [vz * vx * cc - vy * s, vz * vy * cc + vx * s, vz * vz * cc + c, 0],
        [0, 0, 0, 1],
        **kwargs,
    )


class AugmentPipe(nn.Module):
    """Each transformation is applied with probability ``p`` times its multiplier."""

    def __init__(
        self,
        xflip=1, rotate90=1, xint=1, xint_max=0.125,
        scale=1, rotate=1, aniso=1, xfrac=1, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
    ):
        super().__init__()
        self.register_buffer("p", torch.zeros([]))  # Overall multiplier for augmentation probability.
        self.xflip, self.rotate90, self.xint, self.xint_max = float(xflip), float(rotate90), float(xint), float(xint_max)
        self.scale, self.rotate, self.aniso, self.xfrac = float(scale), float(rotate), float(aniso), float(xfrac)
        self.scale_std, self.rotate_max, self.aniso_std, self.xfrac_std = float(scale_std), float(rotate_max), float(aniso_std), float(xfrac_std)
        self.brightness, self.contrast, self.lumaflip, self.hue, self.saturation = float(brightness), float(contrast), float(lumaflip), float(hue), float(saturation)
        self.brightness_std, self.contrast_std, self.hue_max, self.saturation_std = float(brightness_std), float(contrast_std), float(hue_max), float(saturation_std)

    def forward(self, images):
        if float(self.p) == 0:
            return images
        batch_size, _, height, width = images.shape
        device = images.device
        rand = lambda: torch.rand([batch_size], device=device)
        randn = lambda: torch.randn([batch_size], device=device)

        # Inverse geometric transform in pixel coordinates centred on the image.
        G_inv = torch.eye(3, device=device).expand(batch_size, 3, 3)
        if self.xflip > 0:
            i = torch.floor(rand() * 2)
            i = torch.where(rand() < self.xflip * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d(1 - 2 * i, 1)
        if self.rotate90 > 0:
            i = torch.floor(rand() * 4)
            i = torch.where(rand() < self.rotate90 * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ rotate2d(np.pi / 2 * i)
        if self.xint > 0:
            tx, ty = (rand() * 2 - 1) * self.xint_max, (rand() * 2 - 1) * self.xint_max
            keep = rand() < self.xint * self.p
            tx, ty = torch.where(keep, tx, torch.zeros_like(tx)), torch.where(keep, ty, torch.zeros_like(ty))
            G_inv = G_inv @ translate2d(-torch.round(tx * width), -torch.round(ty * height))
        if self.scale > 0:
            s = torch.exp2(randn() * self.scale_std)
            s = torch.where(rand() < self.scale * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d(1 / s, 1 / s)
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))  # P(pre OR post) = p
        if self.rotate > 0:
            theta = (rand() * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(rand() < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d(theta)
        if self.aniso > 0:
            s = torch.exp2(randn() * self.aniso_std)
            s = torch.where(rand() < self.aniso * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d(1 / s, s)
        if self.rotate > 0:
            theta = (rand() * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(rand() < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d(theta)
        if self.xfrac > 0:
            tx, ty = randn() * self.xfrac_std, randn() * self.xfrac_std
            keep = rand() < self.xfrac * self.p
            tx, ty = torch.where(keep, tx, torch.zeros_like(tx)), torch.where(keep, ty, torch.zeros_like(ty))
            G_inv = G_inv @ translate2d(-tx * width, -ty * height)

        # Pixel coordinates -> normalized [-1, 1] coordinates expected by affine_grid.
        theta = scale2d(2 / width, 2 / height, device=device) @ G_inv @ scale2d(width / 2, height / 2, device=device)
        grid = F.affine_grid(theta[:, :2, :], list(images.shape), align_corners=False)
        images = F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

        # Color transform on homogeneous RGB.
        C = torch.eye(4, device=device).expand(batch_size, 4, 4)
        luma_axis = torch.as_tensor(np.asarray([1, 1, 1, 0]) / np.sqrt(3), dtype=torch.float32, device=device)
        if self.brightness > 0:
            b = randn() * self.brightness_std
            b = torch.where(rand() < self.brightness * self.p, b, torch.zeros_like(b))
            C = translate3d(b, b, b) @ C
        if self.contrast > 0:
            c = torch.exp2(randn() * self.contrast_std)
            c = torch.where(rand() < self.contrast * self.p, c, torch.ones_like(c))
            C = scale3d(c, c, c) @ C
        if self.lumaflip > 0:
            i = torch.floor(rand() * 2)
            i = torch.where(rand() < self.lumaflip * self.p, i, torch.zeros_like(i))
            C = (torch.eye(4, device=device) - 2 * luma_axis.ger(luma_axis) * i[:, None, None]) @ C
        if self.hue > 0:
            theta = (rand() * 2 - 1) * np.pi * self.hue_max
            theta = torch.where(rand() < self.hue * self.p, theta, torch.zeros_like(theta))
            C = rotate3d(luma_axis, theta) @ C
        if self.saturation > 0:
            s = torch.exp2(randn() * self.saturation_std)
            s = torch.where(rand() < self.saturation * self.p, s, torch.ones_like(s))
            C = (luma_axis.ger(luma_axis) + (torch.eye(4, device=device) - luma_axis.ger(luma_axis)) * s[:, None, None]) @ C
        images = images.reshape(batch_size, 3, height * width)
        images = C[:, :3, :3] @ images + C[:, :3, 3:]
        return images.reshape(batch_size, 3, height, width)
