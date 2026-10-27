"""Train split only: how class-specific are the samples? Nearest class centroid in Inception feature space
(centroids from real train images that are not in the held-out real subset)."""
import os, sys, glob, numpy as np, torch
os.chdir("/home/jiin/AIC313-D2O")
from cleanfid import fid
device = torch.device("cuda")
feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
real = np.load("results/reference_train_feats.npy").astype(np.float64)
labels = torch.load("data/cache/train_64x64.pt", weights_only=True)["labels"].numpy()
rng = np.random.default_rng(0)
held = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:20] for c in range(151)])
mask = np.ones(len(real), bool); mask[held] = False
centroids = np.stack([real[mask & (labels == c)].mean(0) for c in range(151)])
def d2(a, b): return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
def report(name, feats, cls):
    pred = d2(feats, centroids).argmin(1)
    # distance to the own class centroid, relative to the mean distance to the other centroids
    dist = np.sqrt(np.clip(d2(feats, centroids), 0, None))
    own = dist[np.arange(len(cls)), cls]
    other = (dist.sum(1) - own) / 150
    print(f"{name:34s} nearest-centroid accuracy {np.mean(pred == cls):.3f}   own-centroid distance {own.mean():.2f}   other centroids {other.mean():.2f}   ratio {np.mean(own / other):.3f}", flush=True)
report("real train (held out, 20 per class)", real[held], labels[held])
for arg in sys.argv[1:]:
    name, folder = arg.split("=")
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.png")))
    feats = fid.get_folder_features(folder, feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False).astype(np.float64)
    report(name, feats, np.array([int(f.split("_")[0]) for f in files]))
