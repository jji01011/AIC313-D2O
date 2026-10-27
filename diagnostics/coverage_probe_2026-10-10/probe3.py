import os, sys, numpy as np, torch
from scipy.optimize import linear_sum_assignment
os.chdir("/home/jiin/AIC313-D2O")
C, K, M = 151, 20, 160
pool = np.load(sys.argv[1] + "/pool_feats.npy").astype(np.float64).reshape(C, M, -1)
real = np.load("results/reference_train_feats.npy").astype(np.float64)
labels = torch.load("data/cache/train_64x64.pt", weights_only=True)["labels"].numpy()
def d2(a, b): return np.clip((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0, None)
rng = np.random.default_rng(0)
out = {k: [] for k in ("real->gen nn", "real->real nn", "gen->real nn", "gen->gen nn", "real->gen matched", "real->real matched")}
ncount = []
for c in range(C):
    idx = rng.permutation(np.flatnonzero(labels == c))
    t, o = real[idx[:K]], real[idx[K:K + 80]]
    n = len(o); ncount.append(n)
    g = pool[c, rng.permutation(M)[:n]]            # same number of candidates as the other real images
    g2 = pool[c, rng.permutation(M)[:K]]
    out["real->gen nn"].append(np.sqrt(d2(t, g).min(1)))
    out["real->real nn"].append(np.sqrt(d2(t, o).min(1)))
    out["gen->real nn"].append(np.sqrt(d2(g2, o).min(1)))
    rest = np.setdiff1d(np.arange(M), np.arange(K))
    out["gen->gen nn"].append(np.sqrt(d2(pool[c, :K], pool[c, K:K + n]).min(1)))
    cg, cr = d2(t, g), d2(t, o)
    out["real->gen matched"].append(np.sqrt(cg[np.arange(K), linear_sum_assignment(cg)[1]]))
    out["real->real matched"].append(np.sqrt(cr[np.arange(K), linear_sum_assignment(cr)[1]]))
print("candidates per class: median", int(np.median(ncount)), "min", min(ncount))
for k, v in out.items():
    v = np.concatenate(v); print(f"{k:20s} mean {v.mean():6.2f}  median {np.median(v):6.2f}  10% {np.percentile(v, 10):6.2f}  share below 5: {np.mean(v < 5):.3f}")
