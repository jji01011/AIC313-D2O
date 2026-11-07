import os
import sys

import torch

import model as M
from evaluate import SAMPLES_PER_CATEGORY, compute_fid, generate_samples

REF = os.environ.get("HELDOUT_REF", "/workspace/fastgen/ganmodel/runs/heldout/reference")


class Sampler(torch.nn.Module):
    def __init__(self, net, sigmas):
        super().__init__()
        self.net, self.sigmas = net, sigmas

    def sample(self, shape, device="cuda", category=None, **kw):
        return M.consistency_sample(self.net, shape, device, category, sigmas=self.sigmas)


ckpt = sys.argv[1]
net = M.build_network()
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
net.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()})
net = net.cuda().eval()
cats = torch.arange(151).repeat_interleave(SAMPLES_PER_CATEGORY)
for spec in sys.argv[2:]:
    sigmas = [float(s) for s in spec.split(",")]
    torch.manual_seed(0)
    out = generate_samples(Sampler(net, sigmas), f"/tmp/ect_eval_{abs(hash((ckpt, spec)))}", cats, batch_size=64, device="cuda")
    print(f"HELDOUT\t{len(sigmas)}\t{spec}\t{compute_fid(out, REF, device='cuda', batch_size=64):.2f}", flush=True)
