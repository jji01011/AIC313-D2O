import numpy as np
import torch
from torch import nn
from torch.nn.functional import silu


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel, bias=True, up=False, down=False, groups=1,
        resample_filter=[1, 1], fused_resample=False, init_mode="kaiming_normal", init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.groups = groups
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = nn.Parameter(weight_init([out_channels, in_channels // groups, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0), groups=self.groups)
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad, groups=self.groups)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad, groups=self.groups)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(nn.Module):

    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x32 = x.to(torch.float32)
        if x32.device.type == "cpu" and not x32.requires_grad:
            x32 = x32.contiguous()
        return torch.nn.functional.group_norm(x32, num_groups=self.num_groups, weight=self.weight, bias=self.bias, eps=self.eps).to(x.dtype)


class UNetBlock(nn.Module):

    def __init__(
        self, in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init)
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = torch.einsum("ncq,nck->nqk", q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class FourierEmbedding(nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class SongUNet(nn.Module):

    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        label_dim           = 0,
        model_channels      = 128,
        channel_mult        = [1, 2, 2, 2],
        channel_mult_emb    = 4,
        num_blocks          = 4,
        attn_resolutions    = [16],
        dropout             = 0.10,
        label_dropout       = 0,
        embedding_type      = "positional",
        channel_mult_noise  = 1,
        encoder_type        = "standard",
        decoder_type        = "standard",
        resample_filter     = [1, 1],
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.img_resolution = img_resolution
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == "positional" else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.enc = nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if "aux" not in name]
        self.feature_channels = {img_resolution >> level: model_channels * mult for level, mult in enumerate(channel_mult)}

        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def embed(self, x, noise_labels, class_labels):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        emb = silu(self.map_layer0(emb))
        return silu(self.map_layer1(emb))

    def encode(self, x, emb):
        skips = []
        features = {}
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)
            features[x.shape[-1]] = x
        return x, skips, features

    def forward(self, x, noise_labels, class_labels):
        emb = self.embed(x, noise_labels, class_labels)
        x, skips, _ = self.encode(x, emb)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux

    def forward_features(self, x, noise_labels, class_labels):
        emb = self.embed(x, noise_labels, class_labels)
        return self.encode(x, emb)[2]


ARCHITECTURES = {
    "ddpmpp": dict(embedding_type="positional", encoder_type="standard", decoder_type="standard", channel_mult_noise=1, resample_filter=[1, 1]),
    "ncsnpp": dict(embedding_type="fourier", encoder_type="residual", decoder_type="standard", channel_mult_noise=2, resample_filter=[1, 3, 3, 1]),
}


class EDMPrecond(nn.Module):

    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim       = 0,
        use_fp16        = False,
        sigma_min       = 0,
        sigma_max       = float("inf"),
        sigma_data      = 0.5,
        arch            = "ddpmpp",
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = SongUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **ARCHITECTURES[arch], **model_kwargs)

    def _dtype(self, x, force_fp32):
        return torch.bfloat16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

    def _labels(self, x, class_labels):
        if self.label_dim == 0:
            return None
        if class_labels is None:
            return torch.zeros([1, self.label_dim], device=x.device)
        return class_labels.to(torch.float32).reshape(-1, self.label_dim)

    def coefficients(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out

    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = self._labels(x, class_labels)
        dtype = self._dtype(x, force_fp32)

        c_skip, c_out = self.coefficients(sigma)
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels)
        assert F_x.dtype == dtype
        return c_skip * x + c_out * F_x.to(torch.float32)

    def features(self, x, sigma, class_labels=None, force_fp32=False):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = self._labels(x, class_labels)
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        feats = self.model.forward_features((c_in * x).to(self._dtype(x, force_fp32)), c_noise.flatten(), class_labels=class_labels)
        return {res: feat.to(torch.float32) for res, feat in feats.items()}


class D2OPrecond(EDMPrecond):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 2e-3

    def coefficients(self, sigma):
        c_skip = self.sigma_data ** 2 / ((sigma - self.eps) ** 2 + self.sigma_data ** 2)
        c_out = (sigma - self.eps) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        return c_skip, c_out
