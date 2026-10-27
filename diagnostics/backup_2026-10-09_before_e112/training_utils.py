"""Small helpers shared by the two training scripts."""

import json
import signal
import time

import torch

# Official D2O --freeze levels: parameter-name substrings excluded from the generator optimizer.
FREEZE_LEVELS = {
    0: [],                               # D2O
    1: ["conv", "proj"],                 # D2O-F (about 85% of the parameters frozen)
    2: ["conv", "proj", "qkv"],
    3: ["conv", "proj", "qkv", "skip"],
}


def is_frozen(name, freeze_layer):
    """Official D2O rule (training_loop.py) deciding whether a generator parameter stays fixed."""
    return ("block" in name or "up" in name or "down" in name or "in" in name) and any(key in name for key in freeze_layer)


def apply_freeze(net, freeze_level):
    """Set ``requires_grad`` by the official rule and return the trainable parameters."""
    freeze_layer = FREEZE_LEVELS[freeze_level]
    trainable = []
    for name, param in net.named_parameters():
        frozen = is_frozen(name, freeze_layer)
        param.requires_grad_(not frozen)
        if not frozen:
            trainable.append(param)
    return trainable


@torch.no_grad()
def update_ema(ema, net, cur_nimg, batch_size, halflife_kimg=500, rampup_ratio=0.05):
    """EDM / D2O EMA with a half-life that ramps up with training progress."""
    halflife_nimg = halflife_kimg * 1000
    if rampup_ratio:
        halflife_nimg = min(halflife_nimg, cur_nimg * rampup_ratio)
    beta = 0.5 ** (batch_size / max(halflife_nimg, 1e-8))
    for p_ema, p_net in zip(ema.parameters(), net.parameters()):
        p_ema.copy_(p_net.detach().lerp(p_ema, beta))


def sanitize_gradients(params):
    """Replace non-finite gradient entries, as the official training loop does; return how many there were."""
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return 0
    # Fast path: finite norms mean every entry is finite, and nan_to_num would leave such gradients untouched.
    # Checking each tensor separately costs several hundred small GPU ops per step (about 12% of an EDM iteration).
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
    """Turns SIGINT / SIGTERM into a flag so a run can save its state before exiting."""

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
    """Accumulates scalar statistics between two log lines."""

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
