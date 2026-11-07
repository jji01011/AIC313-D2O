"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import os
import sys

import torch
from torch import nn

from src.utils import count_parameters, parameter_summary

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher import networks as edm2_networks
from d2o.config import GENERATOR_EDM2_KWARGS
from d2o.edm2 import EDM2Precond

ONE_NOISE_SEED = 4

EDM2_CONFIG = {
    "family": "edm2",
    "preset": "E112",
    "kwargs": {
        "img_resolution": 64, "img_channels": 3, "label_dim": 151, "sigma_data": 0.634817864072234,
        "model_channels": 112, "channel_mult": [1, 2, 3, 4], "num_blocks": 3, "attn_resolutions": [16, 8],
        "channels_per_head": 56, "augment_dim": 9, "augment_balance": 0.2, "logvar_channels": 128,
        "null_class": True, "use_fp16": True, "precision": "fp16", "forced_wn": "post_step",
        "dropout_res": {"64": 0.0, "32": 0.15, "16": 0.15, "8": 0.15},
    },
}
FEW_SIGMAS = (80.0, 5.0, 1.8, 0.4)
FEW_NOISE_SEED = 3


def stratified_latents(seed, group, shape, n=20):
    g = torch.Generator()
    z = torch.stack([torch.randn(shape, generator=g.manual_seed(seed * 10**7 + n * group + j)) for j in range(n)])
    b = shape[-1] // 8
    k = torch.arange(8, dtype=torch.float64)
    dct = (torch.cos(torch.pi * (2 * k[None] + 1) * k[:, None] / 16) * torch.where(k[:, None] == 0, 0.125**0.5, 0.25**0.5)).float()
    blocks = nn.functional.avg_pool2d(z, b) * b
    g.manual_seed(seed * 10**7 + 5 * 10**6 + group)
    strata = torch.argsort(torch.rand(blocks.shape, generator=g), dim=0)
    u = ((strata + torch.rand(blocks.shape, generator=g)) / n).clamp(1e-6, 1 - 1e-6)
    new = dct.T @ (2**0.5 * torch.erfinv(2 * u - 1)) @ dct
    return z + nn.functional.interpolate((new - blocks) / b, scale_factor=b, mode="nearest")


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()


class ModelOneNFE(Model):

    def __init__(self, device="cpu", **kwargs):
        super().__init__(EDM2Precond({**GENERATOR_EDM2_KWARGS, **kwargs}))
        self.register_buffer("sigma_g", torch.tensor(80.0))
        self.num_sampled = 0
        self.group = (-1, None)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch = shape[0]
        if category is None:
            category = torch.randint(0, self.backbone.label_dim, (batch,))
        z = []
        for i in range(self.num_sampled, self.num_sampled + batch):
            if self.group[0] != i // 20:
                self.group = (i // 20, stratified_latents(ONE_NOISE_SEED, i // 20, shape[1:]))
            z.append(self.group[1][i % 20])
        z = torch.stack(z).to(device)
        self.num_sampled += batch
        labels = nn.functional.one_hot(category.to(device).long(), self.backbone.label_dim)
        sigma = self.sigma_g.to(device).expand(batch)
        return self.backbone(z * sigma.view(-1, 1, 1, 1), sigma, class_labels=labels, force_fp32=True).clamp(-1, 1)


class ModelFewNFE(Model):

    def __init__(self, device="cpu", **kwargs):
        super().__init__(edm2_networks.build_network(EDM2_CONFIG))
        self.num_sampled = 0
        self.group = (-1, None)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        net = self.backbone
        batch = shape[0]
        if category is None:
            category = torch.randint(0, net.num_classes, (batch,))
        labels = net.one_hot(category.to(device))
        idx = range(self.num_sampled, self.num_sampled + batch)
        self.num_sampled += batch
        z = []
        for i in idx:
            if self.group[0] != i // 20:
                self.group = (i // 20, [stratified_latents(FEW_NOISE_SEED + 100 * k, i // 20, shape[1:]) for k in range(4)])
            z.append(torch.stack([zk[i % 20] for zk in self.group[1]]))
        z = torch.stack(z, 1).to(device)
        x = net.mean_image.to(device=device, dtype=torch.float32) + FEW_SIGMAS[0] * z[0]
        x = net(x, torch.full((batch,), FEW_SIGMAS[0], device=device), labels).float()
        for k, sigma in enumerate(FEW_SIGMAS[1:], 1):
            x = x + sigma * z[k]
            x = net(x, torch.full((batch,), sigma, device=device), labels).float()
        return x.clamp(-1, 1)
