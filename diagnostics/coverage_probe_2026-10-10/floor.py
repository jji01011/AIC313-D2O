"""How low can the official score go? Real train images (20 per class, the size the evaluation generates) against the
validation reference, for ten different random draws. Reporting only: nothing is selected with these numbers."""
import os, numpy as np, torch
os.chdir("/home/jiin/AIC313-D2O")
from cleanfid import fid
device = torch.device("cuda")
feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
val = fid.get_folder_features("results/reference", feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False).astype(np.float64)
real = np.load("results/reference_train_feats.npy").astype(np.float64)
labels = torch.load("data/cache/train_64x64.pt", weights_only=True)["labels"].numpy()
def half(sig):
    w, v = np.linalg.eigh(sig); return (v * np.sqrt(np.clip(w, 0, None))) @ v.T
mu_v, sig_v = val.mean(0), np.cov(val, rowvar=False); sig_v_half = half(sig_v)
def fd(a):
    sa = np.cov(a, rowvar=False)
    w = np.linalg.eigvalsh(sig_v_half @ sa @ sig_v_half)
    return float(((a.mean(0) - mu_v) ** 2).sum() + np.trace(sa) + np.trace(sig_v) - 2 * np.sqrt(np.clip(w, 0, None)).sum())
scores = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:20] for c in range(151)])
    scores.append(fd(real[rows]))
print("real train images, 20 per class, against val:", " ".join(f"{s:.2f}" for s in scores))
print(f"mean {np.mean(scores):.2f}  min {np.min(scores):.2f}  max {np.max(scores):.2f}  std {np.std(scores):.2f}")
print(f"all {len(real)} train images against val: {fd(real):.2f}")
