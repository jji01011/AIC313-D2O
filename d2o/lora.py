"""Low-rank fake score for the distribution-matching term (``train_d2o.py --dmd_fake_lora``).

DMD2 (Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024) trains a
second diffusion network next to the generator, the fake score, which follows the distribution of the generator's
samples. A copy of the teacher in that role is a whole trainable network of the teacher's size, and every network
that is trained together with the generator counts towards the parameter limit of the generator's framework.

Here the fake score is the frozen teacher itself plus a low-rank branch next to each of its convolutions (LoRA, Hu et
al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022; ProlificDreamer, Wang et al., NeurIPS 2023,
parameterises the fake score of variational score distillation the same way). Only the branches are trained. The
teacher's weights are not copied: the branches hook into the teacher's own modules and are switched on for the fake
score's pass only, so the teacher stays a frozen model of its own and the trainable part is the branches alone.

The up-projections start at zero: at the start the fake score is exactly the teacher, as in DMD2.

``shared=True`` puts the branches on the generator instead of the teacher: the fake score is then the generator's own
network, read as it is at that moment and as a constant, plus the branches. Every weight the fake score uses is in
that case a weight of the generator's framework already, whichever way a frozen base is counted. The base is no
longer the teacher's denoiser but a network fine-tuned at one noise level, so the branches have more to make up for.
"""

import numpy as np
import torch
from torch import nn

from d2o.edm2 import MPConv


class LoRAFakeScore(nn.Module):
    """``D_fake(x, sigma, category)`` = ``base`` (an ``EDM2Precond``) with rank-``rank`` branches switched on.

    ``base`` is the frozen teacher, or with ``shared=True`` the generator that is being trained.
    """

    def __init__(self, base, rank, shared=False, decoder_only=False):
        super().__init__()
        self._base = (base,)  # in a tuple: not registered here, so its weights are neither trained nor saved with the branches
        self.shared = shared
        self.sigma_data = base.sigma_data
        self.active = False
        self.down = nn.ParameterList()
        self.up = nn.ParameterList()
        for name, module in base.unet.named_modules():
            if decoder_only and not (name.startswith("dec.") or name == "out_conv"):
                continue  # e.g. a generator whose encoder was never trained: its encoder is the teacher's as it stands
            if isinstance(module, MPConv) and module.weight.ndim == 4:
                cout, cin, kernel, _ = module.weight.shape
                down = nn.Parameter(torch.randn(rank, cin, kernel, kernel) / np.sqrt(cin * kernel * kernel))
                up = nn.Parameter(torch.zeros(cout, rank, 1, 1))
                self.down.append(down)
                self.up.append(up)
                module.register_forward_hook(self._branch(down, up))

    def _branch(self, down, up):
        def hook(module, inputs, output):
            if not self.active:
                return None  # the base network's own pass
            x = inputs[0]
            y = torch.nn.functional.conv2d(x, down.to(x.dtype), padding=down.shape[-1] // 2)
            return output + torch.nn.functional.conv2d(y, up.to(x.dtype))
        return hook

    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        base = self._base[0]
        modules, params = [], []
        if self.shared:
            # The generator as a constant: evaluation mode, so that forced weight normalization does not rewrite weights
            # that a pending backward pass of the generator still needs, and no gradient into its parameters from here.
            modules = [module for module in base.modules() if module.training]
            params = [param for param in base.parameters() if param.requires_grad]
            for module in modules:
                module.training = False
            for param in params:
                param.requires_grad_(False)
        self.active = True
        try:
            return base(x, sigma, class_labels=class_labels, force_fp32=force_fp32)
        finally:
            self.active = False
            for module in modules:
                module.training = True
            for param in params:
                param.requires_grad_(True)
