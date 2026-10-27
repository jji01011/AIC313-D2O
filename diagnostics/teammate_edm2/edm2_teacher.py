"""Loader for the teammate's EDM2 teacher (checkpoints/edm/ema-*.pt, format ``fastgen-teacher-ema-v1``).

The checkpoint holds weights and a config but no network code. Its tensor names and shapes are those of the public
EDM2 network (Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models", CVPR 2024;
https://github.com/NVlabs/edm2, ``training/networks_edm2.py``, CC BY-NC-SA 4.0), which is imported from a local clone
(``--edm2_repo``, default /home/jiin/edm2-official). Three things differ from the public network and are reconstructed
here; how they enter the computation was inferred, not read from the teammate's code:

- ``mean_image``: the network works on images minus this per-pixel mean (``sigma_data`` equals the standard deviation of
  the train images around it to three digits);
- ``unet.emb_label`` takes 152 inputs: the 151 categories plus a null class (index 151);
- ``unet.emb_augment`` (9 augmentation inputs, ``augment_balance`` 0.2): zero at sampling time; ``augment_mode`` selects
  how the zero input is merged, because that changes the embedding by a constant factor.

Analysis only so far: nothing here is used by training or by the submitted model.
"""

import sys

import numpy as np
import torch


def load_teacher(path, edm2_repo="/home/jiin/edm2-official", device="cuda"):
    if edm2_repo not in sys.path:
        sys.path.insert(0, edm2_repo)
    from training import networks_edm2 as edm2

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    kwargs = dict(ckpt["network"]["kwargs"])
    label_dim = kwargs["label_dim"] + (1 if kwargs.get("null_class") else 0)
    net = edm2.Precond(
        img_resolution=kwargs["img_resolution"], img_channels=kwargs["img_channels"], label_dim=label_dim,
        use_fp16=kwargs["use_fp16"], sigma_data=kwargs["sigma_data"], logvar_channels=kwargs["logvar_channels"],
        model_channels=kwargs["model_channels"], channel_mult=kwargs["channel_mult"], num_blocks=kwargs["num_blocks"],
        attn_resolutions=kwargs["attn_resolutions"], channels_per_head=kwargs["channels_per_head"],
    )
    state = {name: tensor.float() for name, tensor in ckpt["state_dict"].items()}
    extra = {name: state.pop(name) for name in ("mean_image", "unet.emb_augment.weight")}
    report = net.load_state_dict(state, strict=True)
    net.eval().requires_grad_(False).to(device)
    return Teacher(net, edm2, extra["mean_image"].to(device), kwargs, null_index=kwargs["label_dim"]), report


class Teacher(torch.nn.Module):
    """Denoiser ``D(x, sigma, category)`` on images in [-1, 1]."""

    def __init__(self, net, edm2, mean_image, kwargs, null_index, center=True, label_scale=None, augment_mode="after_label"):
        super().__init__()
        self.net, self.edm2, self.kwargs, self.null_index = net, edm2, kwargs, null_index
        self.register_buffer("mean_image", mean_image)
        self.center = center  # False: ignore mean_image (a reconstruction variant, for probing)
        self.label_scale = label_scale  # None: sqrt(number of label inputs), as in the public code
        self.augment_mode = augment_mode  # "none", "after_label" or "before_label"
        self.augment_balance = kwargs["augment_balance"]

    def _unet(self, x, noise_labels, class_labels):
        unet, edm2 = self.net.unet, self.edm2
        zero_merge = (1 - self.augment_balance) / np.sqrt((1 - self.augment_balance) ** 2 + self.augment_balance ** 2)
        emb = unet.emb_noise(unet.emb_fourier(noise_labels))
        if self.augment_mode == "before_label":
            emb = emb * zero_merge  # mp_sum(emb, emb_augment(0), t=augment_balance)
        scale = np.sqrt(class_labels.shape[1]) if self.label_scale is None else self.label_scale
        emb = edm2.mp_sum(emb, unet.emb_label(class_labels * scale), t=unet.label_balance)
        if self.augment_mode == "after_label":
            emb = emb * zero_merge
        emb = edm2.mp_silu(emb)

        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in unet.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)
        for name, block in unet.dec.items():
            if "block" in name:
                x = edm2.mp_cat(x, skips.pop(), t=unet.concat_balance)
            x = block(x, emb)
        return unet.out_conv(x, gain=unet.out_gain)

    def forward(self, x, sigma, category=None, fp16=True):
        """``category``: int64 ``[N]`` (``None`` = null class). Returns the denoised images."""
        net = self.net
        x = x.to(torch.float32)
        if self.center:
            x = x - self.mean_image
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).reshape(-1, 1, 1, 1).expand(x.shape[0], 1, 1, 1)
        if category is None:
            category = torch.full([x.shape[0]], self.null_index, device=x.device)
        class_labels = torch.nn.functional.one_hot(category, net.label_dim).to(torch.float32)
        dtype = torch.float16 if (fp16 and x.device.type == "cuda") else torch.float32

        c_skip = net.sigma_data ** 2 / (sigma ** 2 + net.sigma_data ** 2)
        c_out = sigma * net.sigma_data / (sigma ** 2 + net.sigma_data ** 2).sqrt()
        c_in = 1 / (net.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        F_x = self._unet((c_in * x).to(dtype), c_noise.to(dtype), class_labels.to(dtype))
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x + self.mean_image if self.center else D_x
