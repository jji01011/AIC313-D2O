"""Does the discriminator recognise the exact training images? Logits for real train images against slightly shifted copies."""
import sys
import torch
import torch.nn.functional as F
from d2o.data import load_train_tensors
from d2o.discriminator import ProjectedDiscriminator
from d2o.networks import EDMPrecond
from model import ModelOneNFE

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
device = torch.device("cuda")
B, NB = 32, 64
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
index = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))[: B * NB]
real = (images[index].float() / 127.5 - 1).to(device)
lab = labels[index].to(device)

def shift(x, dx, dy):
    p = max(abs(dx), abs(dy))
    padded = F.pad(x, (p, p, p, p), mode="reflect")
    return padded[:, :, p + dy: p + dy + 64, p + dx: p + dx + 64]

def zoom(x, factor):
    size = round(64 * factor)
    big = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    off = (size - 64) // 2
    return big[:, :, off: off + 64, off: off + 64] if factor >= 1 else F.pad(big, [(64 - size) // 2, 64 - size - (64 - size) // 2] * 2, mode="reflect")

torch.manual_seed(0)
fake = []
for chunk in lab.split(B):
    z = torch.randn((len(chunk), 3, 64, 64), device=device)
    with torch.no_grad():
        fake.append(model.backbone(z * model.sigma_g, model.sigma_g.expand(len(chunk)), class_labels=F.one_hot(chunk, 151), force_fp32=True))
fake = torch.cat(fake)
sets = {
    "real": real, "real flipped (seen in training)": real.flip(3),
    "real shifted 1 px": shift(real, 1, 1), "real shifted 2 px": shift(real, 2, -2), "real shifted 4 px": shift(real, -4, 4),
    "real zoomed 1.06": zoom(real, 1.0625), "real zoomed 1.12": zoom(real, 1.125),
    "fake": fake, "fake shifted 2 px": shift(fake, 2, -2), "fake zoomed 1.06": zoom(fake, 1.0625),
}

@torch.no_grad()
def logits(x):
    D.heads.train(True)
    out = []
    for i in range(NB):
        torch.manual_seed(1000 + i)
        out.append(D(x[i * B:(i + 1) * B], lab[i * B:(i + 1) * B]).float())
    return torch.cat(out)

for name, x in sets.items():
    lt = logits(x)
    print(f"{name:34s} mean logit {lt.mean():+.3f}  sign {lt.sign().mean():+.3f}  share of images with mean logit > 0: {100 * (lt.mean(1) > 0).float().mean():5.1f}%", flush=True)
