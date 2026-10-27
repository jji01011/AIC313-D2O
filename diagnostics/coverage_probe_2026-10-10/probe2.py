"""Second pass of the coverage probe on the saved pool features (train split only). Frechet distance through two
symmetric eigendecompositions (scipy's sqrtm took minutes per call here); checked once against clean-fid."""
import os, sys, time
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
ROOT = "/home/jiin/AIC313-D2O"; os.chdir(ROOT)
SCRATCH, M = sys.argv[1], int(sys.argv[2])
C, K = 151, 20
pool = np.load(f"{SCRATCH}/pool_feats.npy").astype(np.float64).reshape(C, M, -1)
real = np.load("results/reference_train_feats.npy").astype(np.float64)
labels = torch.load("data/cache/train_64x64.pt", weights_only=True)["labels"].numpy()
ref = np.load("results/reference_train_stats.npz"); mu_ref, sig_ref = ref["mu"], ref["sigma"]

def half(sig):
    w, v = np.linalg.eigh(sig)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T
ref_half = half(sig_ref)
def frechet(mu1, sig1, mu2, sig2, sig2_half):
    w = np.linalg.eigvalsh(sig2_half @ sig1 @ sig2_half)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sig1) + np.trace(sig2) - 2 * np.sqrt(np.clip(w, 0, None)).sum())
def fd(a):
    return frechet(a.mean(0), np.cov(a, rowvar=False), mu_ref, sig_ref, ref_half)
def fd2(a, b):
    sb = np.cov(b, rowvar=False)
    return frechet(a.mean(0), np.cov(a, rowvar=False), b.mean(0), sb, half(sb))
def d2(a, b):
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T

t0 = time.time()
a = pool[:, :K].reshape(C * K, -1)
print(f"check: eigh {fd(a):.3f}  ({time.time() - t0:.0f}s)", flush=True)
if "--check" in sys.argv:
    from cleanfid import fid
    print(f"       clean-fid {fid.frechet_distance(a.mean(0), np.cov(a, rowvar=False), mu_ref, sig_ref):.3f}  ({time.time() - t0:.0f}s)", flush=True)

for m in [int(v) for v in sys.argv[3].split(",")]:
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        rand, match, near, tgt, held, gap = [], [], [], [], [], []
        for c in range(C):
            idx = rng.permutation(np.flatnonzero(labels == c))
            targets, other = real[idx[:K]], real[idx[K:2 * K]]
            p = pool[c, rng.permutation(M)[:m]]
            rand.append(p[:K])
            cost = d2(targets, p)
            sel = linear_sum_assignment(cost)[1]
            match.append(p[sel]); gap.append(np.sqrt(np.clip(cost[np.arange(K), sel], 0, None)))
            near.append(p[np.argsort(d2(p, real[idx]).min(1))[:K]])
            tgt.append(targets); held.append(other)
        rand, match, near, tgt, held = map(np.concatenate, (rand, match, near, tgt, held))
        rows.append([fd(rand), fd(match), fd(near), fd(tgt), fd2(rand, held), fd2(match, held), fd2(near, held), fd2(tgt, held), float(np.concatenate(gap).mean())])
        print(f"  m={m} seed {seed}: " + " ".join(f"{v:.2f}" for v in rows[-1]) + f"  ({time.time() - t0:.0f}s)", flush=True)
    r = np.array(rows).mean(0)
    print(f"m = {m:3d} | vs all train: random {r[0]:.2f}  matched-to-real {r[1]:.2f}  nearest-to-real {r[2]:.2f}  real {r[3]:.2f}"
          f" | vs disjoint 3,020 real: random {r[4]:.2f}  matched {r[5]:.2f}  nearest {r[6]:.2f}  real {r[7]:.2f} | mean match distance {r[8]:.2f}", flush=True)
