"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import torch
from torch import nn

from d2o.config import GENERATOR_EDM2_KWARGS, GENERATOR_FAMILY, GENERATOR_KWARGS, MAX_FEW_NFE, SIGMA_G, SIGMAS_FEW
from d2o.edm2 import EDM2Precond
from d2o.networks import D2OPrecond
from src.utils import count_parameters, parameter_summary


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
    """Student model evaluated with exactly one model function evaluation.

    D2O one-step generator: an EDM U-Net (pre-trained from scratch on the train
    split, then fine-tuned with a GAN objective only) evaluated once at a fixed
    noise level, ``x = G(sigma_g * z, sigma_g, category)``.

    ``family`` selects the U-Net: "edm" is our own EDM, "edm2" the EDM2 denoiser of ``d2o/edm2.py``.
    """

    def __init__(self, device="cpu", sigma_g=SIGMA_G, family=GENERATOR_FAMILY, **kwargs):
        super().__init__(self._make_backbone(family, kwargs))
        self.family = family
        # Stored in the checkpoint so sampling always matches the training setting.
        self.register_buffer("sigma_g", torch.tensor(float(sigma_g)))
        self.to(device)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        latents = torch.randn(shape, device=device)
        if category is None:
            category = torch.randint(self.backbone.label_dim, (shape[0],), device=device)
        class_labels = nn.functional.one_hot(category.to(device).long(), self.backbone.label_dim)
        sigma = self.sigma_g.to(device).expand(shape[0])
        # The only network evaluation.
        images = self.backbone(latents * self.sigma_g.to(device), sigma, class_labels=class_labels, force_fp32=True)
        return images.clamp(-1, 1)

    @staticmethod
    def _make_backbone(family, kwargs):
        if family == "edm2":
            return EDM2Precond({**GENERATOR_EDM2_KWARGS, **kwargs})
        return D2OPrecond(**{**GENERATOR_KWARGS, **kwargs})

    def load_state_dict(self, state_dict, *args, **kwargs):
        # evaluate.py builds the model without arguments: a checkpoint of the other family gets its own U-Net first.
        family = "edm2" if "backbone.unet.out_gain" in state_dict else "edm"
        if family != self.family:
            self.backbone = self._make_backbone(family, {}).to(self.sigma_g.device)
            self.family = family
        return super().load_state_dict(state_dict, *args, **kwargs)


class ModelFewNFE(Model):
    """Student model evaluated with no more than four model evaluations.

    Few-step D2O generator: the same EDM U-Net and GAN-only fine-tuning as
    ``ModelOneNFE``, trained at several fixed noise levels. Sampling starts from
    pure noise and alternates one network evaluation with re-noising,
    ``x = G(x + sigma_k * noise, sigma_k, category)``.
    """

    def __init__(self, device="cpu", sigmas=SIGMAS_FEW, **kwargs):
        super().__init__(D2OPrecond(**{**GENERATOR_KWARGS, **kwargs}))
        assert 1 <= len(sigmas) <= MAX_FEW_NFE and all(sigma > 0 for sigma in sigmas)
        # Stored in the checkpoint so sampling always matches the training setting;
        # zero-padded to a fixed length so that any number of steps loads strictly.
        self.register_buffer("sigmas", torch.tensor([float(sigma) for sigma in sigmas] + [0.0] * (MAX_FEW_NFE - len(sigmas))))
        self.to(device)

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if category is None:
            category = torch.randint(self.backbone.label_dim, (shape[0],), device=device)
        class_labels = nn.functional.one_hot(category.to(device).long(), self.backbone.label_dim)
        images = torch.zeros(shape, device=device)
        # One network evaluation per stored noise level: at most MAX_FEW_NFE.
        for sigma in self.sigmas[self.sigmas > 0].to(device):
            noisy = images + sigma * torch.randn(shape, device=device)
            images = self.backbone(noisy, sigma.expand(shape[0]), class_labels=class_labels, force_fp32=True)
        return images.clamp(-1, 1)
