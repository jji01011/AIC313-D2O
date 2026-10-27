"""Which reconstruction of the teammate's EDM2 teacher is the trained one?

The network was trained to denoise, so the right reconstruction is the one with the lowest denoising error on train
images. Every variant sees the same images and the same noise. Train split only.

    PYTHONPATH=. python diagnostics/teammate_edm2/probe_denoise.py
"""

import itertools
import sys

import numpy as np
import torch

sys.path.insert(0, "diagnostics/teammate_edm2")
from edm2_teacher import load_teacher  # noqa: E402

torch.manual_seed(0)
device = "cuda"
teacher, _ = load_teacher("checkpoints/edm/ema-0016777216-0.100.pt", device=device)
cache = torch.load("data/cache/train_64x64.pt", weights_only=True)
index = torch.randperm(len(cache["images"]))[:256]
images = (cache["images"][index].float() / 127.5 - 1).to(device)
labels = cache["labels"][index].to(device)
sigmas = [0.05, 0.2, 0.5, 1.0, 2.5, 10.0, 40.0]
noise = torch.randn_like(images)

print("mean squared denoising error per sigma (lower is better); 'input' is the error of the noisy image itself")
print(f"{'variant':58s}" + "".join(f"{s:>9g}" for s in sigmas))
print(f"{'input':58s}" + "".join(f"{s * s:9.4f}" for s in sigmas))
rows = []
for center, label_scale, augment_mode, conditional in itertools.product(
        (True, False), (None, np.sqrt(151)), ("after_label", "before_label", "none"), (True, False)):
    teacher.center, teacher.label_scale, teacher.augment_mode = center, label_scale, augment_mode
    errors = []
    for sigma in sigmas:
        with torch.no_grad():
            out = torch.cat([teacher(images[i:i + 64] + sigma * noise[i:i + 64], sigma, labels[i:i + 64] if conditional else None, fp16=False)
                             for i in range(0, len(images), 64)])
        errors.append((out - images).pow(2).mean().item())
    name = f"center={int(center)} label_scale={'sqrt152' if label_scale is None else 'sqrt151'} augment={augment_mode:12s} {'class' if conditional else 'null '}"
    rows.append((name, errors))
    print(f"{name:58s}" + "".join(f"{e:9.4f}" for e in errors), flush=True)
