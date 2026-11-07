import json
import signal
import time

import torch

FREEZE_LEVELS = {
    0: [],
    1: ["conv", "proj"],
    2: ["conv", "proj", "qkv"],
    3: ["conv", "proj", "qkv", "skip"],
}


def is_frozen(name, freeze_layer):
    return ("block" in name or "up" in name or "down" in name or "in" in name) and any(key in name for key in freeze_layer)


ENCODER_PREFIXES = ("model.map_", "model.enc.", "unet.emb_", "unet.enc.")


def apply_freeze(net, freeze_level, freeze_encoder=False):
    freeze_layer = FREEZE_LEVELS[freeze_level]
    trainable = []
    for name, param in net.named_parameters():
        frozen = is_frozen(name, freeze_layer) or (freeze_encoder and name.startswith(ENCODER_PREFIXES))
        param.requires_grad_(not frozen)
        if not frozen:
            trainable.append(param)
    return trainable


@torch.no_grad()
def update_ema(ema, net, cur_nimg, batch_size, halflife_kimg=500, rampup_ratio=0.05):
    halflife_nimg = halflife_kimg * 1000
    if rampup_ratio:
        halflife_nimg = min(halflife_nimg, cur_nimg * rampup_ratio)
    beta = 0.5 ** (batch_size / max(halflife_nimg, 1e-8))
    for p_ema, p_net in zip(ema.parameters(), net.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))


def sanitize_gradients(params):
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return 0
    foreach_norm = getattr(torch, "_foreach_norm", None)
    norms = foreach_norm(grads) if foreach_norm is not None else [grad.norm() for grad in grads]
    if bool(torch.stack(norms).isfinite().all()):
        return 0
    nonfinite = 0
    for grad in grads:
        nonfinite = nonfinite + (~torch.isfinite(grad)).sum()
        torch.nan_to_num(grad, nan=0, posinf=1e5, neginf=-1e5, out=grad)
    return nonfinite


class StopSignal:

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        self.requested = True


class JsonlLogger:
    def __init__(self, path):
        self.file = open(path, "at")
        self.start = time.time()

    def log(self, **fields):
        fields["time_sec"] = round(time.time() - self.start, 1)
        self.file.write(json.dumps(fields) + "\n")
        self.file.flush()


class RunningMean:

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, **values):
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def pop(self):
        means = {key: self.sums[key] / self.counts[key] for key in self.sums}
        self.sums, self.counts = {}, {}
        return means
