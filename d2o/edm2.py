"""EDM2 network blocks, the encoder of an EDM2 denoiser as a feature network, and the whole denoiser as a generator.

The building blocks (``normalize`` ... ``Block``) are a port of ``training/networks_edm2.py`` from NVlabs/edm2
(Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models", CVPR 2024; CC BY-NC-SA 4.0)
without its ``dnnlib`` / ``torch_utils`` dependencies. Module and parameter names are those of the public code, so a
checkpoint of that network loads by name.

``EDM2Encoder`` is the part of such a denoiser that the projected discriminator can use as its frozen feature
network: the embeddings and the encoder blocks down to ``min_resolution``. It reads the checkpoint format of the
teammate's teacher (``format: fastgen-teacher-ema-v1``), which differs from the public network in three places:
images are centred with a per-pixel ``mean_image``, the label embedding has one extra input for a null class, and
there is an augmentation embedding that is zero at inference time and only rescales the embedding
(``augment_balance``). The teacher was trained from scratch on the train split, like every other network here.

``EDM2Precond`` is the whole denoiser (encoder, decoder, EDM preconditioning) read from the same checkpoint: the
one-step generator of ``train_d2o.py --g_edm2``. It is an ``EDM2Encoder`` as well, so that a discriminator can take
its features from the generator's own (frozen) encoder instead of holding a second copy of it.
"""

import numpy as np
import torch
from torch import nn


def normalize(x, dim=None, eps=1e-4):
    """Normalize to unit magnitude over ``dim`` (default: all dimensions except the first)."""
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def resample(x, f=(1, 1), mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = torch.as_tensor(np.outer(f, f)[np.newaxis, np.newaxis, :, :], dtype=x.dtype, device=x.device)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a, b, dim=1, t=0.5):
    Na, Nb = a.shape[dim], b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    return torch.cat([C / np.sqrt(Na) * (1 - t) * a, C / np.sqrt(Nb) * t * b], dim=dim)


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32).ger(self.freqs.to(torch.float32)) + self.phases.to(torch.float32)
        return (y.cos() * np.sqrt(2)).to(x.dtype)


class MPConv(nn.Module):
    """Magnitude-preserving convolution or fully-connected layer with forced weight normalization."""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class Block(nn.Module):
    """U-Net encoder / decoder block with optional self-attention."""

    def __init__(
        self, in_channels, out_channels, emb_channels, flavor="enc", resample_mode="keep", resample_filter=(1, 1),
        attention=False, channels_per_head=64, dropout=0, res_balance=0.3, attn_balance=0.3, clip_act=256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=[3, 3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations (out of place: the features are also returned to the caller).
        if self.clip_act is not None:
            x = x.clip(-self.clip_act, self.clip_act)
        return x


class _EncoderUNet(nn.Module):
    """Embeddings and encoder blocks of the public ``UNet``, down to ``min_resolution``."""

    def __init__(self, img_resolution, img_channels, label_dim, model_channels, channel_mult, num_blocks, attn_resolutions,
                 channels_per_head, min_resolution, label_balance=0.5):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cemb = max(cblock)
        self.label_balance = label_balance
        self.emb_fourier = MPFourier(cblock[0])
        self.emb_noise = MPConv(cblock[0], cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[])
        self.enc = nn.ModuleDict()
        self.feature_channels = {}
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if res < min_resolution:
                break
            if level == 0:
                cin, cout = cout, channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(cout, cout, cemb, flavor="enc", resample_mode="down", channels_per_head=channels_per_head)
            for idx in range(num_blocks):
                cin, cout = cout, channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(cin, cout, cemb, flavor="enc", attention=(res in attn_resolutions), channels_per_head=channels_per_head)
            self.feature_channels[res] = cout


class _UNet(_EncoderUNet):
    """The whole public ``UNet``: the embeddings and encoder above, plus the decoder and the output convolution."""

    def __init__(self, img_resolution, img_channels, label_dim, model_channels, channel_mult, num_blocks, attn_resolutions,
                 channels_per_head, label_balance=0.5, concat_balance=0.5):
        super().__init__(img_resolution, img_channels, label_dim, model_channels, channel_mult, num_blocks, attn_resolutions,
                         channels_per_head, min_resolution=1, label_balance=label_balance)
        cblock = [model_channels * x for x in channel_mult]
        cemb = max(cblock)
        self.concat_balance = concat_balance
        self.out_gain = nn.Parameter(torch.zeros([]))
        self.dec = nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        cout = cblock[-1]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(cout, cout, cemb, flavor="dec", attention=True, channels_per_head=channels_per_head)
                self.dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec", channels_per_head=channels_per_head)
            else:
                self.dec[f"{res}x{res}_up"] = Block(cout, cout, cemb, flavor="dec", resample_mode="up", channels_per_head=channels_per_head)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(cin, cout, cemb, flavor="dec", attention=(res in attn_resolutions), channels_per_head=channels_per_head)
        self.out_conv = MPConv(cout, img_channels, kernel=[3, 3])


class EDM2Encoder(nn.Module):
    """Frozen encoder of an EDM2 denoiser: ``features()`` has the interface of ``EDMPrecond.features()``."""

    def __init__(self, network_kwargs, min_resolution=16, use_fp16=False):
        super().__init__()
        kw = network_kwargs
        self.network_kwargs = dict(kw)
        self.img_resolution = kw["img_resolution"]
        self.label_dim = kw["label_dim"]  # categories, without the null class
        self.sigma_data = kw["sigma_data"]
        self.use_fp16 = use_fp16
        # The augmentation embedding of a zero input merges as mp_sum(emb, 0, t=augment_balance): a constant factor.
        balance = kw.get("augment_balance", 0) if kw.get("augment_dim", 0) else 0
        self.augment_factor = (1 - balance) / np.sqrt((1 - balance) ** 2 + balance ** 2)
        self.unet = self._make_unet(kw, min_resolution)
        self.feature_channels = self.unet.feature_channels
        self.register_buffer("mean_image", torch.zeros([kw["img_channels"], kw["img_resolution"], kw["img_resolution"]]))

    @staticmethod
    def _make_unet(kw, min_resolution):
        return _EncoderUNet(
            kw["img_resolution"], kw["img_channels"], kw["label_dim"] + (1 if kw.get("null_class") else 0), kw["model_channels"],
            kw["channel_mult"], kw["num_blocks"], kw["attn_resolutions"], kw["channels_per_head"], min_resolution,
        )

    @classmethod
    def from_checkpoint(cls, path, min_resolution=16, use_fp16=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["network"]["family"] == "edm2", checkpoint["network"]["family"]
        encoder = cls(checkpoint["network"]["kwargs"], min_resolution=min_resolution, use_fp16=use_fp16)
        wanted = encoder.state_dict()
        state = {name: tensor.float() for name, tensor in checkpoint["state_dict"].items() if name in wanted}
        encoder.load_state_dict(state, strict=True)  # every tensor of the encoder must be in the checkpoint
        return encoder.eval().requires_grad_(False)

    def _dtype(self, x, force_fp32):
        return torch.bfloat16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

    def _encode(self, x, sigma, class_labels, force_fp32):
        """Embedding and encoder pass: the centred input, ``sigma``, the embedding, every encoder output, one map per resolution."""
        unet = self.unet
        x = x.to(torch.float32) - self.mean_image
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = self._dtype(x, force_fp32)
        labels = torch.zeros([x.shape[0], unet.emb_label.weight.shape[1]], device=x.device)
        if class_labels is None:
            labels[:, -1] = 1  # null class
        else:
            labels[:, : self.label_dim] = class_labels.to(torch.float32).reshape(-1, self.label_dim)

        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = (sigma.flatten().log() / 4).to(dtype)
        emb = unet.emb_noise(unet.emb_fourier(c_noise))
        emb = mp_sum(emb, unet.emb_label(labels.to(dtype) * np.sqrt(labels.shape[1])), t=unet.label_balance)
        emb = mp_silu(emb * self.augment_factor)

        h = (c_in * x).to(dtype)
        h = torch.cat([h, torch.ones_like(h[:, :1])], dim=1)
        skips, feats = [], {}
        for name, block in unet.enc.items():
            h = block(h) if "conv" in name else block(h, emb)
            skips.append(h)
            feats[h.shape[-1]] = h  # the last block of a resolution overwrites the earlier ones
        return x, sigma, emb, skips, feats

    def features(self, x, sigma, class_labels=None, force_fp32=False):
        """Encoder features ``{resolution: [N, C, res, res]}`` of ``x`` (in [-1, 1], already noised to level ``sigma``)."""
        feats = self._encode(x, sigma, class_labels, force_fp32)[4]
        return {res: feat.to(torch.float32) for res, feat in feats.items()}


class EDM2Precond(EDM2Encoder):
    """The whole EDM2 denoiser ``D(x, sigma, category)`` on images in [-1, 1], with the interface of ``EDMPrecond``.

    ``encoder_frozen`` (set by the trainer together with ``apply_freeze(freeze_encoder=True)``) keeps the embeddings and
    the encoder in eval mode, so that forced weight normalization never rewrites the weights that are meant to stay fixed.
    """

    encoder_frozen = False

    def __init__(self, network_kwargs, use_fp16=False):
        super().__init__(network_kwargs, min_resolution=1, use_fp16=use_fp16)

    @staticmethod
    def _make_unet(kw, min_resolution):
        return _UNet(
            kw["img_resolution"], kw["img_channels"], kw["label_dim"] + (1 if kw.get("null_class") else 0), kw["model_channels"],
            kw["channel_mult"], kw["num_blocks"], kw["attn_resolutions"], kw["channels_per_head"],
        )

    @classmethod
    def from_checkpoint(cls, path, use_fp16=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["network"]["family"] == "edm2", checkpoint["network"]["family"]
        net = cls(checkpoint["network"]["kwargs"], use_fp16=use_fp16)
        wanted = net.state_dict()
        # Left out: the uncertainty head (training only) and the augmentation embedding (zero input, see augment_factor).
        net.load_state_dict({name: tensor.float() for name, tensor in checkpoint["state_dict"].items() if name in wanted}, strict=True)
        return net

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            for module in (self.unet.emb_noise, self.unet.emb_label, self.unet.enc):
                module.eval()
        return self

    def features_fixed(self, x, sigma, class_labels=None, force_fp32=False):
        """``features()`` of an encoder that is being trained, taken as it is right now and as a constant.

        For a discriminator that reads its features from the generator's own encoder while the generator trains all of
        it (``train_d2o.py --d_shared 2``). The embeddings and the encoder run in evaluation mode, so that forced weight
        normalization does not rewrite weights that the generator's pass of the same step still needs for its backward
        pass, and their parameters take no gradient from here; the gradient with respect to ``x`` is kept.
        """
        modules = (self.unet.emb_noise, self.unet.emb_label, self.unet.enc)
        params = [param for module in modules for param in module.parameters()]
        was_training = [module.training for module in modules]
        needed_grad = [param.requires_grad for param in params]
        for module in modules:
            module.eval()
        for param in params:
            param.requires_grad_(False)
        try:
            return self.features(x, sigma, class_labels, force_fp32)
        finally:
            for module, mode in zip(modules, was_training):
                module.train(mode)
            for param, flag in zip(params, needed_grad):
                param.requires_grad_(flag)

    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        unet = self.unet
        x, sigma, emb, skips, _ = self._encode(x, sigma, class_labels, force_fp32)
        h = skips[-1]
        skips = list(skips)
        for name, block in unet.dec.items():
            if "block" in name:
                h = mp_cat(h, skips.pop(), t=unet.concat_balance)
            h = block(h, emb)
        F_x = unet.out_conv(h, gain=unet.out_gain)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return self.mean_image + c_skip * x + c_out * F_x.to(torch.float32)
