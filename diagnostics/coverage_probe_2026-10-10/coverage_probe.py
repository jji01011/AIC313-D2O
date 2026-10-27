"""Upper-bound probe (train split only, diagnostic): how far would the train-reference FID of the current candidate
drop if its samples were allocated to cover the training images? A pool of M samples per class is generated once;
then 20 per class are picked (a) at random, (b) by matching 20 random real train images of the class to their nearest
pool samples in Inception feature space, (c) by keeping the 20 pool samples closest to any real image of the class."""
import os, sys, time
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from torchvision.transforms.functional import to_pil_image
ROOT = "/home/jiin/AIC313-D2O"; os.chdir(ROOT); sys.path.insert(0, ROOT)
from cleanfid import fid
from model import Model

SCRATCH = sys.argv[1]; CKPT = sys.argv[2]; M = int(sys.argv[3])
pool_dir, feat_path = f"{SCRATCH}/pool", f"{SCRATCH}/pool_feats.npy"
device = torch.device("cuda")
C, K = 151, 20
if not os.path.exists(feat_path):
    model = Model.load_checkpoint(CKPT, evaluate_mode="one_nfe", device=device).eval()
    torch.manual_seed(1234)
    cats = torch.arange(C).repeat_interleave(M)
    counts, t0 = {}, time.time()
    for s in range(0, len(cats), 64):
        c = cats[s:s + 64].to(device)
        x = model.sample((len(c), 3, 64, 64), device=device, category=c)
        for img, k in zip(x, c.tolist()):
            i = counts.get(k, 0); counts[k] = i + 1
            to_pil_image(((img.float().cpu() + 1) / 2).clamp(0, 1)).convert("RGB").save(f"{pool_dir}/{k:04d}_{i:03d}.png")
    print(f"generated {len(cats)} in {time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
    feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    feats = fid.get_folder_features(pool_dir, feat_model, num_workers=4, batch_size=64, device=device, mode="clean", verbose=False)
    np.save(feat_path, feats.astype(np.float32))
pool = np.load(feat_path).astype(np.float64).reshape(C, M, -1)   # sorted file names = (class, index)
real = np.load("results/reference_train_feats.npy").astype(np.float64)
labels = torch.load("data/cache/train_64x64.pt", weights_only=True)["labels"].numpy()
ref = np.load("results/reference_train_stats.npz"); mu_ref, sig_ref = ref["mu"], ref["sigma"]

def fd(a, mu=mu_ref, sig=sig_ref):
    return float(fid.frechet_distance(a.mean(0), np.cov(a, rowvar=False), mu, sig))
def fd2(a, b):
    return float(fid.frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)))
def d2(a, b):
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T

print(f"pool {pool.shape}; real per class min {np.bincount(labels).min()} max {np.bincount(labels).max()}")
for m in [K, 40, 80, M] if M > 80 else [K, 40, M]:
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        rand, match, near, tgt, held = [], [], [], [], []
        for c in range(C):
            idx = rng.permutation(np.flatnonzero(labels == c))
            targets, other = real[idx[:K]], real[idx[K:2 * K]]
            p = pool[c, rng.permutation(M)[:m]]
            rand.append(p[:K])
            cost = d2(targets, p)
            match.append(p[linear_sum_assignment(cost)[1]])
            near.append(p[np.argsort(d2(p, real[idx]).min(1))[:K]])
            tgt.append(targets); held.append(other)
        rand, match, near, tgt, held = map(np.concatenate, (rand, match, near, tgt, held))
        rows.append([fd(rand), fd(match), fd(near), fd(tgt), fd2(rand, held), fd2(match, held), fd2(near, held), fd2(tgt, held)])
    r = np.array(rows)
    names = ["random 20", "matched to 20 real", "20 nearest to real", "the 20 real themselves"]
    print(f"\npool size per class m = {m}  (mean of 3 target draws; per-seed in brackets)")
    for j, n in enumerate(names):
        print(f"  {n:24s} vs all train: {r[:, j].mean():6.2f} [{', '.join(f'{v:.2f}' for v in r[:, j])}]   vs disjoint 3,020 real: {r[:, j + 4].mean():6.2f}")
