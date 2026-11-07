import random
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
sys.path.insert(0, "/root/AIC313-Project-FastGen/ganmodel")
from dataset import PokemonDataModule

THRESH, PER_CLASS, SEED = float(sys.argv[1]), int(sys.argv[2]), 0
OUT = Path("/root/AIC313-Project-FastGen/ganmodel/runs/heldout")
ds = PokemonDataModule(num_workers=8, return_category=True).train_dataset
X, Y = [], []
for x, y in torch.utils.data.DataLoader(ds, batch_size=512, num_workers=8):
    X.append(x); Y.append(y)
X = torch.cat(X); Y = torch.cat(Y)
Xs = F.avg_pool2d((X.cuda() + 1) / 2, 2)
small, smallf = Xs.flatten(1), Xs.flip(-1).flatten(1)

rng = random.Random(SEED)
held, ref, sizes = [], [], []
for c in range(int(Y.max()) + 1):
    idx = (Y == c).nonzero().squeeze(1)
    a = small[idx.cuda()]
    d = torch.minimum(torch.cdist(a, a), torch.cdist(a, smallf[idx.cuda()])) / a.shape[1] ** 0.5
    adj = (d < THRESH).cpu()
    parent = list(range(len(idx)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in adj.nonzero().tolist():
        parent[find(i)] = find(j)
    comps = {}
    for i in range(len(idx)):
        comps.setdefault(find(i), []).append(int(idx[i]))
    comps = list(comps.values())
    rng.shuffle(comps)
    chosen = [m for m in comps if len(m) <= 3][:PER_CLASS]
    for m in chosen:
        held.extend(m); ref.append(m[0]); sizes.append(len(m))
print(f"thresh {THRESH}: held-out {len(held)} images ({len(held)/len(X):.1%}), reference {len(ref)} distinct, "
      f"component sizes mean {sum(sizes)/len(sizes):.2f}")
torch.save({"heldout_indices": sorted(held), "reference_indices": ref, "thresh": THRESH, "per_class": PER_CLASS},
           OUT / "heldout_split.pt")
ref_dir = OUT / "reference"
ref_dir.mkdir(exist_ok=True)
counts = {}
for i in ref:
    c = int(Y[i]); k = counts.get(c, 0); counts[c] = k + 1
    save_image((X[i] + 1) / 2, ref_dir / f"{c:04d}_{k:02d}.png")
print("reference images written:", len(list(ref_dir.glob('*.png'))))
