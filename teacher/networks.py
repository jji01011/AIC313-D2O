import numpy as np
import torch

from . import official
from .official import MPFourier, MPConv, normalize, mp_silu, mp_sum, mp_cat

_DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def _is_official(module, name):
    return type(module).__name__ == name

def mpconv_modules(net):
    return [m for m in net.modules() if _is_official(m, 'MPConv')]

@torch.no_grad()
def project_mpconv_weights_(net):
    for m in mpconv_modules(net):
        m.weight.copy_(normalize(m.weight.to(torch.float32)))

def block_resolution(name):
    return int(name.split('x')[0])


class Block(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        emb_channels,
        flavor              = 'enc',
        resample_mode       = 'keep',
        resample_filter     = [1,1],
        attention           = False,
        channels_per_head   = 64,
        dropout             = 0,
        res_balance         = 0.3,
        attn_balance        = 0.3,
        clip_act            = 256,
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
        self.attention_impl = 'einsum'
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        x = official.networks_edm2.resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)
            if self.attention_impl == 'sdpa':
                y = torch.nn.functional.scaled_dot_product_attention(q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)).transpose(2, 3)
            else:
                w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
                y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


class EDM2UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        label_dim,
        model_channels      = 192,
        channel_mult        = [1,2,3,4],
        channel_mult_noise  = None,
        channel_mult_emb    = None,
        num_blocks          = 3,
        attn_resolutions    = [16,8],
        label_balance       = 0.5,
        concat_balance      = 0.5,
        augment_dim         = 0,
        augment_balance     = 0.2,
        dropout_res         = None,
        extra_embeddings    = (),
        **block_kwargs,
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

        self.augment_dim = augment_dim
        self.augment_balance = augment_balance
        self.emb_augment = MPConv(augment_dim, cemb, kernel=[]) if augment_dim != 0 else None
        self.extra_embeddings = list(extra_embeddings)
        if self.extra_embeddings:
            self.emb_extra_fourier = torch.nn.ModuleDict({n: MPFourier(cnoise) for n in self.extra_embeddings})
            self.emb_extra = torch.nn.ModuleDict({n: MPConv(cnoise, cemb, kernel=[]) for n in self.extra_embeddings})
            self.emb_extra_gain = torch.nn.ParameterDict({n: torch.nn.Parameter(torch.zeros([])) for n in self.extra_embeddings})
        if dropout_res is not None:
            self.set_dropout(dropout_res)

    def blocks(self):
        for name, block in list(self.enc.items()) + list(self.dec.items()):
            if _is_official(block, 'Block'):
                yield name, block

    def set_dropout(self, dropout_res):
        dropout_res = {int(k): float(v) for k, v in dropout_res.items()}
        for name, block in self.blocks():
            block.dropout = dropout_res.get(block_resolution(name), 0.0)

    def get_dropout(self):
        return {name: block.dropout for name, block in self.blocks()}

    def set_attention(self, impl):
        assert impl in ('einsum', 'sdpa')
        for _name, block in self.blocks():
            block.attention_impl = impl

    def forward(self, x, noise_labels, class_labels, augment_labels=None, extra_labels=None, return_enc=False):
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        if self.emb_augment is not None:
            if augment_labels is None:
                augment_labels = torch.zeros([emb.shape[0], self.augment_dim], device=emb.device)
            augment_labels = augment_labels.to(torch.float32).reshape(-1, self.augment_dim)
            emb = mp_sum(emb, self.emb_augment(augment_labels * np.sqrt(self.augment_dim)), t=self.augment_balance)
        for name in self.extra_embeddings:
            if extra_labels is not None and extra_labels.get(name) is not None:
                v = extra_labels[name].to(torch.float32).flatten()
                emb = emb + self.emb_extra[name](self.emb_extra_fourier[name](v), gain=self.emb_extra_gain[name])
        emb = mp_silu(emb)

        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)
        enc = x

        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return (x, enc) if return_enc else x


class EDM2Precond(torch.nn.Module):
    family = 'edm2'

    def __init__(self,
        img_resolution,
        img_channels,
        label_dim,
        use_fp16        = True,
        sigma_data      = 0.5,
        logvar_channels = 128,
        precision       = 'fp16',
        null_class      = True,
        forced_wn       = 'post_step',
        **unet_kwargs,
    ):
        super().__init__()
        assert precision in _DTYPES and forced_wn in ('forward', 'post_step')
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = label_dim
        self.null_class = bool(null_class and label_dim != 0)
        self.label_dim = label_dim + (1 if self.null_class else 0)
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.precision = precision
        self.forced_wn = forced_wn
        self.unet = EDM2UNet(img_resolution=img_resolution, img_channels=img_channels, label_dim=self.label_dim, **unet_kwargs)
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])
        self.mean_image = torch.nn.Parameter(torch.zeros([img_channels, img_resolution, img_resolution]), requires_grad=False)

    def train(self, mode=True):
        super().train(mode)
        if self.forced_wn == 'post_step':
            for m in mpconv_modules(self):
                m.training = False
        return self

    def one_hot(self, labels):
        return torch.nn.functional.one_hot(labels.long(), self.label_dim).to(torch.float32)

    def null_labels(self, batch_size, device):
        assert self.null_class
        return self.one_hot(torch.full([batch_size], self.num_classes, device=device, dtype=torch.long))

    def c_noise_fn(self, sigma):
        return sigma.flatten().log() / 4

    def forward(self, x, sigma, class_labels=None, force_fp32=False, return_logvar=False, return_enc=False, **unet_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else (self.null_labels(1, x.device) if self.null_class else torch.zeros([1, self.label_dim], device=x.device)) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = _DTYPES[self.precision] if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        mean = self.mean_image.to(torch.float32)
        x = x - mean

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = self.c_noise_fn(sigma)

        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, c_noise, class_labels, return_enc=return_enc, **unet_kwargs)
        if return_enc:
            F_x, enc = F_x
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        D_x = D_x + mean

        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return (D_x, logvar, enc) if return_enc else (D_x, logvar)
        return (D_x, enc) if return_enc else D_x

    def set_dropout(self, dropout_res):
        self.unet.set_dropout(dropout_res)

    def get_dropout(self):
        return self.unet.get_dropout()

    def set_attention(self, impl):
        self.unet.set_attention(impl)

    def project_weights_(self):
        project_mpconv_weights_(self)

    def post_step_(self):
        if self.forced_wn == 'post_step':
            self.project_weights_()

    def init_weights_(self):
        self.project_weights_()


class DDPMppPrecond(torch.nn.Module):
    family = 'edm'

    def __init__(self,
        img_resolution,
        img_channels,
        label_dim       = 0,
        use_fp16        = False,
        sigma_min       = 0,
        sigma_max       = float('inf'),
        sigma_data      = 0.5,
        model_type      = 'SongUNet',
        precision       = 'bf16',
        null_class      = True,
        **model_kwargs,
    ):
        super().__init__()
        assert model_type == 'SongUNet' and precision in _DTYPES
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = label_dim
        self.null_class = bool(null_class and label_dim != 0)
        self.label_dim = label_dim + (1 if self.null_class else 0)
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.precision = precision
        self.model = official.SongUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=self.label_dim, **model_kwargs)
        self.mean_image = torch.nn.Parameter(torch.zeros([img_channels, img_resolution, img_resolution]), requires_grad=False)

    def one_hot(self, labels):
        return torch.nn.functional.one_hot(labels.long(), self.label_dim).to(torch.float32)

    def null_labels(self, batch_size, device):
        assert self.null_class
        return self.one_hot(torch.full([batch_size], self.num_classes, device=device, dtype=torch.long))

    def forward(self, x, sigma, class_labels=None, force_fp32=False, return_logvar=False, **model_kwargs):
        assert not return_logvar, 'EDM networks have no uncertainty head'
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else (self.null_labels(1, x.device) if self.null_class else torch.zeros([1, self.label_dim], device=x.device)) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = _DTYPES[self.precision] if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        mean = self.mean_image.to(torch.float32)
        x = x - mean

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x + mean

    def unet_blocks(self):
        return [m for m in self.model.modules() if _is_official(m, 'UNetBlock')]

    def set_dropout(self, dropout_res):
        p = max(float(v) for v in dropout_res.values()) if isinstance(dropout_res, dict) else float(dropout_res)
        for block in self.unet_blocks():
            block.dropout = p

    def get_dropout(self):
        return {str(i): b.dropout for i, b in enumerate(self.unet_blocks())}

    def set_attention(self, impl):
        pass

    def project_weights_(self):
        pass

    def post_step_(self):
        pass

    def init_weights_(self):
        pass


def build_network(cfg):
    family = cfg['family']
    kwargs = dict(cfg['kwargs'])
    if family == 'edm2':
        net = EDM2Precond(**kwargs)
    elif family == 'edm':
        net = DDPMppPrecond(**kwargs)
    else:
        raise ValueError(f'Unknown network family {family!r}')
    net.init_weights_()
    return net

def load_teacher_state_dict(net, state_dict):
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    bad = [k for k in missing if not k.startswith('unet.emb_extra')]
    if bad or unexpected:
        raise RuntimeError(f'state_dict mismatch: missing {bad}, unexpected {list(unexpected)}')
    return missing

def count_parameters(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, frozen=total - trainable)
