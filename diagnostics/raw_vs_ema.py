"""Output statistics of the raw and the EMA generator of one training state, next to the real train images."""
import sys
import torch
import torch.nn.functional as F
from d2o.data import load_train_tensors
from model import ModelOneNFE

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
device = torch.device("cuda")
model = ModelOneNFE(device=device, sigma_g=state["args"]["sigma_g"], **state["net_kwargs"])
model.load_state_dict(state["ema_model"])
model.eval().requires_grad_(False)
ema = {k: v.clone() for k, v in model.backbone.state_dict().items()}
raw = {k: v.to(device) for k, v in state["net"].items()}
categories = torch.arange(151, device=device).repeat_interleave(10)
lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)[None, None].repeat(3, 1, 1, 1)

def describe(name, u8):
    x = u8.to(device).float()
    hf = F.conv2d(x, lap, groups=3).abs().mean().item()
    print(f"{name:28s} ==255 {100 * (u8 == 255).float().mean():5.2f}%  >=250 {100 * (u8 >= 250).float().mean():5.2f}%  ==0 {100 * (u8 == 0).float().mean():5.2f}%  <=5 {100 * (u8 <= 5).float().mean():5.2f}%"
          f"  mean {x.mean():6.2f}  std {x.std():6.2f}  |laplacian| {hf:5.2f}", flush=True)

def sample(weights, gain=1.0, unclamped=False):
    model.backbone.load_state_dict(weights)
    out = []
    torch.manual_seed(0)
    for chunk in categories.split(32):
        latents = torch.randn((len(chunk), 3, 64, 64), device=device)
        labels = F.one_hot(chunk, 151)
        with torch.no_grad():
            img = model.backbone(latents * model.sigma_g, model.sigma_g.expand(len(chunk)), class_labels=labels, force_fp32=True)
        out.append(img)
    return torch.cat(out)

def to_u8(x, gain=1.0):
    return ((x * gain).clamp(-1, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).cpu()

images, labels = load_train_tensors("./data/pokemon-generation-one-22k", "./data/pokemon-generation-one-22k")
real = torch.cat([images[labels == c][:10] for c in range(151)])
describe("real train", real)
x_ema, x_raw = sample(ema), sample(raw)
describe("EMA", to_u8(x_ema))
describe("raw", to_u8(x_raw))
describe("EMA gain 1.3", to_u8(x_ema, 1.3))
describe("raw gain 1.3", to_u8(x_raw, 1.3))
# How far beyond [-1, 1] do the unclamped outputs go, and how different are the two generators on the same latents?
for name, x in (("EMA", x_ema), ("raw", x_raw)):
    print(f"{name}: unclamped max {x.max():.3f} min {x.min():.3f}  share > 0.99: {100 * (x > 0.99).float().mean():.2f}%  share > 1.0: {100 * (x > 1.0).float().mean():.2f}%  99.9th pct {x.flatten()[::37].quantile(0.999):.3f}")
print(f"RMS(raw - EMA) on the same latents: {(x_raw - x_ema).square().mean().sqrt():.4f}   RMS(EMA): {x_ema.square().mean().sqrt():.4f}")
# Weight-space distance, relative.
num = sum((raw[k].float() - ema[k].float()).square().sum() for k in ema if ema[k].is_floating_point())
den = sum(ema[k].float().square().sum() for k in ema if ema[k].is_floating_point())
print(f"relative weight distance |raw - EMA| / |EMA|: {(num / den).sqrt():.5f}")
