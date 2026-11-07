import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, "/root/AIC313-Project-FastGen/ganmodel")
from dataset import PokemonDataModule

ds = PokemonDataModule(num_workers=8, return_category=True).train_dataset
X, Y = [], []
for x, y in torch.utils.data.DataLoader(ds, batch_size=512, num_workers=8):
    X.append(x); Y.append(y)
X = ((torch.cat(X) + 1) / 2).cuda(); Y = torch.cat(Y).cuda()
small = F.avg_pool2d(X, 2).flatten(1)
smallf = F.avg_pool2d(X.flip(-1), 2).flatten(1)
nn_d = torch.empty(len(X), device="cuda")
for c in Y.unique():
    idx = (Y == c).nonzero().squeeze(1)
    a = small[idx]
    d = torch.minimum(torch.cdist(a, a), torch.cdist(a, smallf[idx])) / a.shape[1] ** 0.5
    d.fill_diagonal_(float("inf"))
    nn_d[idx] = d.min(1).values
q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device="cuda")
print("NN pixel-RMSE quantiles (10/25/50/75/90%):", [round(v, 4) for v in nn_d.quantile(q).tolist()])
for t in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1):
    print(f"fraction with a near-duplicate (RMSE < {t}): {(nn_d < t).float().mean().item():.3f}")
torch.save({"nn_d": nn_d.cpu(), "labels": Y.cpu()}, "/root/AIC313-Project-FastGen/ganmodel/runs/heldout/train_nn.pt")
