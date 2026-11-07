import sys
from pathlib import Path
import torch
from torchvision.utils import save_image
sys.path.insert(0, "/root/AIC313-Project-FastGen/ganmodel")
from dataset import PokemonDataModule
from evaluate import compute_fid

ds = PokemonDataModule(num_workers=8, return_category=True).train_dataset
X, Y = [], []
for x, y in torch.utils.data.DataLoader(ds, batch_size=512, num_workers=8):
    X.append(x); Y.append(y)
X, Y = torch.cat(X), torch.cat(Y)
drop = torch.zeros(len(X), dtype=torch.bool)
drop[torch.load("/root/AIC313-Project-FastGen/ganmodel/runs/heldout/heldout_split.pt")["heldout_indices"]] = True
g = torch.Generator().manual_seed(0)
for seed in (0, 1):
    out = Path(f"/root/AIC313-Project-FastGen/ganmodel/runs/heldout/floor_heldin_{seed}")
    out.mkdir(exist_ok=True)
    for c in range(151):
        idx = ((Y == c) & ~drop).nonzero().squeeze(1)
        pick = idx[torch.randperm(len(idx), generator=g)[:20]]
        for k, i in enumerate(pick.tolist()):
            save_image((X[i] + 1) / 2, out / f"{c:04d}_{k:02d}.png")
    fid = compute_fid(out, "/root/AIC313-Project-FastGen/ganmodel/runs/heldout/reference", device="cuda", batch_size=16)
    print(f"FLOOR seed{seed} {fid:.2f}", flush=True)
