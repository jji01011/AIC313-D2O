import numpy as np
import torch
from torch import nn

from d2o.edm2 import MPConv


class LoRAFakeScore(nn.Module):

    def __init__(self, base, rank, shared=False, decoder_only=False):
        super().__init__()
        self._base = (base,)
        self.shared = shared
        self.sigma_data = base.sigma_data
        self.active = False
        self.down = nn.ParameterList()
        self.up = nn.ParameterList()
        for name, module in base.unet.named_modules():
            if decoder_only and not (name.startswith("dec.") or name == "out_conv"):
                continue
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
                return None
            x = inputs[0]
            y = torch.nn.functional.conv2d(x, down.to(x.dtype), padding=down.shape[-1] // 2)
            return output + torch.nn.functional.conv2d(y, up.to(x.dtype))
        return hook

    def forward(self, x, sigma, class_labels=None, force_fp32=False):
        base = self._base[0]
        modules, params = [], []
        if self.shared:
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
