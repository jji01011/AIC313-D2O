"""Does the discriminator of a training state see the difference that the output gain makes?

Logits of the trained heads for real train images and for EMA-generator samples (unclamped as in training, clamped,
clamped with gain), once with batch-norm statistics of each batch on its own (how the heads are trained) and once
with the running statistics. Also the per-channel batch statistics that the first batch norm of every head removes.
"""
import sys
import torch
import torch.nn.functional as F
from d2o.data import load_train_tensors
from d2o.discriminator import ProjectedDiscriminator
from d2o.networks import EDMPrecond
from model import ModelOneNFE

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
gains = [float(v) for v in sys.argv[2].split(",")]
device = torch.device("cuda")
B, NB = 32, 48
model = ModelOneNFE(device=device, sigma_g=state["args"]["sigma_g"], **state["net_kwargs"])
model.load_state_dict(state["ema_model"])
model.eval().requires_grad_(False)
edm_ckpt = torch.load("runs/edm_c96/edm-005000.pt", map_location="cpu", weights_only=False)
edm = EDMPrecond(**edm_ckpt["net_kwargs"])
edm.load_state_dict(edm_ckpt["ema"])
D = ProjectedDiscriminator(edm, c_dim=151, proj_type=2, sigma_mean=-3.0, sigma_std=1.0).to(device)
D.load_state_dict(state["discriminator"])
D.requires_grad_(False)

images, labels = load_train_tensors("./data/pokemon-generation-one-22k", "./data/pokemon-generation-one-22k")
g = torch.Generator().manual_seed(0)
index = torch.randperm(len(images), generator=g)[: B * NB]
real = (images[index].float() / 127.5 - 1).to(device)
lab = labels[index].to(device)
torch.manual_seed(0)
fake = []
for chunk in lab.split(B):
    z = torch.randn((len(chunk), 3, 64, 64), device=device)
    with torch.no_grad():
        fake.append(model.backbone(z * model.sigma_g, model.sigma_g.expand(len(chunk)), class_labels=F.one_hot(chunk, 151), force_fp32=True))
fake = torch.cat(fake)
sets = {"real": real, "fake unclamped (training)": fake}
for gain in gains:
    sets[f"fake clamped gain {gain:g}"] = (fake * gain).clamp(-1, 1)

first_bn_inputs = {}
def hook(name):
    def fn(module, inputs, output):
        x = inputs[0].double()
        acc = first_bn_inputs.setdefault(name, [0, 0, 0])
        acc[0] += x.sum(dim=(0, 2, 3)).cpu(); acc[1] += x.square().sum(dim=(0, 2, 3)).cpu(); acc[2] += x.numel() // x.shape[1]
    return fn
for k, disc in D.heads.mini_discs.items():
    disc.main[0].main[1].register_forward_hook(hook(k))

@torch.no_grad()
def logits(x, train_bn):
    D.heads.train(train_bn)
    out = []
    for i in range(NB):
        torch.manual_seed(1000 + i)  # the same noise levels and noise for every set
        out.append(D(x[i * B:(i + 1) * B], lab[i * B:(i + 1) * B]).float())
    return torch.cat(out)

stats = {}
for name, x in sets.items():
    first_bn_inputs.clear()
    lt = logits(x, True)
    stats[name] = {k: (v[0] / v[2], (v[1] / v[2] - (v[0] / v[2]) ** 2).sqrt()) for k, v in first_bn_inputs.items()}
    le = logits(x, False)
    print(f"{name:30s} batch statistics: mean logit {lt.mean():+.3f} sign {lt.sign().mean():+.3f} | running statistics: mean logit {le.mean():+.3f} sign {le.sign().mean():+.3f}", flush=True)

print("first batch norm of each head: RMS over channels of (mean - mean_real) / std_real, and of ln(std / std_real)")
for name in sets:
    if name == "real":
        continue
    parts = []
    for k in stats[name]:
        m, s = stats[name][k]
        mr, sr = stats["real"][k]
        parts.append(f"head {k}: {((m - mr) / sr).square().mean().sqrt():.3f} / {(s / sr).log().square().mean().sqrt():.3f}")
    print(f"{name:30s} " + "   ".join(parts))
