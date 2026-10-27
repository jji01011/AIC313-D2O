"""Pixel-level statistics of real train images against saved generator samples, with and without the output gain."""
import sys, glob, random
import numpy as np
from PIL import Image

def load(folder, n=3020, seed=0):
    files = sorted(glob.glob(folder + "/*.png"))
    random.Random(seed).shuffle(files)
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files[:n]]).astype(np.float32)

def gain(x, g):
    return np.clip(((x / 127.5 - 1) * g).clip(-1, 1) * 127.5 + 127.5, 0, 255).round()

def describe(name, x):
    lum = x.mean(-1)
    chroma = x.max(-1) - x.min(-1)
    gx = np.abs(np.diff(lum, axis=2))[:, :-1]
    gy = np.abs(np.diff(lum, axis=1))[:, :, :-1]
    grad = gx + gy
    bins = [0, 1, 26, 77, 128, 179, 230, 250, 255, 256]
    hist = np.histogram(x, bins=bins)[0] / x.size * 100
    print(f"{name:22s} flat(grad<=2) {100 * (grad <= 2).mean():5.1f}%  grad<=8 {100 * (grad <= 8).mean():5.1f}%  grad p90 {np.percentile(grad, 90):5.1f} p99 {np.percentile(grad, 99):6.1f}"
          f"  chroma mean {chroma.mean():5.1f} p90 {np.percentile(chroma, 90):5.1f}  lum mean {lum.mean():6.1f}")
    print(f"{'':22s} value histogram %: " + " ".join(f"[{a}-{b - 1}] {h:4.1f}" for a, b, h in zip(bins[:-1], bins[1:], hist)))

real = load("results/reference_train")
describe("real train", real)
for folder in sys.argv[1:]:
    fake = load(folder)
    describe(folder.split("/")[-2], fake)
    describe("  + gain 1.3", gain(fake, 1.3))
