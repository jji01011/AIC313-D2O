import torch


def stratified_normal(n, device, generator=None):
    u = (torch.arange(n, dtype=torch.float64, device=device) + torch.rand([], dtype=torch.float64, device=device, generator=generator)) / n
    u = u.clamp(1e-12, 1 - 1e-12)
    z = torch.special.ndtri(u).to(torch.float32)
    return z[torch.randperm(n, device=device, generator=generator)]

def sample_sigma(n, P_mean, P_std, device, stratified=True):
    rnd_normal = stratified_normal(n, device) if stratified else torch.randn([n], device=device)
    return (rnd_normal * P_std + P_mean).exp()


def edm2_loss(net, images, labels, sigma, sigma_data, augment_labels=None):
    sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    noise = torch.randn_like(images) * sigma
    denoised, logvar = net(images + noise, sigma, labels, augment_labels=augment_labels, return_logvar=True)
    loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
    with torch.no_grad():
        wmse = (weight * (denoised.detach() - images) ** 2).mean(dim=(1, 2, 3))
        info = dict(loss=loss.detach().mean(dim=(1, 2, 3)), wmse=wmse, logvar=logvar.detach().flatten(), sigma=sigma.detach().flatten())
    return loss, info

def edm_loss(net, images, labels, sigma, sigma_data, augment_labels=None):
    sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(images) * sigma
    D_yn = net(images + n, sigma, labels, augment_labels=augment_labels)
    loss = weight * ((D_yn - images) ** 2)
    with torch.no_grad():
        per = loss.detach().mean(dim=(1, 2, 3))
        info = dict(loss=per, wmse=per, logvar=torch.zeros_like(per), sigma=sigma.detach().flatten())
    return loss, info

LOSSES = dict(edm2=edm2_loss, edm=edm_loss)


SIGMA_GRID = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0)

@torch.no_grad()
def fixed_sigma_wmse(net, images, labels_onehot, sigma_data, sigmas=SIGMA_GRID, seed=0, batch_size=128):
    out = {}
    device = images.device
    for s_idx, s in enumerate(sigmas):
        gen = torch.Generator(device).manual_seed(int(seed) * 1000 + s_idx)
        total = 0.0
        for start in range(0, images.shape[0], batch_size):
            x = images[start : start + batch_size]
            y = labels_onehot[start : start + batch_size]
            sigma = torch.full([x.shape[0], 1, 1, 1], float(s), device=device)
            n = torch.randn(x.shape, device=device, generator=gen) * sigma
            D = net(x + n, sigma, y).to(torch.float32)
            weight = (s ** 2 + sigma_data ** 2) / (s * sigma_data) ** 2
            total += float((weight * (D - x) ** 2).mean(dim=(1, 2, 3)).sum())
        out[float(s)] = total / images.shape[0]
    return out


SIGMA_BUCKET_EDGES = (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0)
SIGMA_BUCKET_NAMES = ('<0.05', '0.05-0.2', '0.2-0.5', '0.5-1', '1-2', '2-5', '5-20', '>=20')

class BucketStats:
    def __init__(self, device):
        self.device = device
        self.edges = torch.tensor(SIGMA_BUCKET_EDGES, device=device)
        self.reset()

    def reset(self):
        nb = len(SIGMA_BUCKET_NAMES)
        self.count = torch.zeros([nb], dtype=torch.float64, device=self.device)
        self.loss = torch.zeros([nb], dtype=torch.float64, device=self.device)
        self.wmse = torch.zeros([nb], dtype=torch.float64, device=self.device)
        self.logvar = torch.zeros([nb], dtype=torch.float64, device=self.device)

    def add(self, info):
        idx = torch.bucketize(info['sigma'], self.edges, right=True)
        ones = torch.ones_like(info['sigma'], dtype=torch.float64)
        self.count.index_add_(0, idx, ones)
        self.loss.index_add_(0, idx, info['loss'].to(torch.float64))
        self.wmse.index_add_(0, idx, info['wmse'].to(torch.float64))
        self.logvar.index_add_(0, idx, info['logvar'].to(torch.float64))

    def summary(self):
        count = self.count.cpu().numpy()
        loss, wmse, logvar = self.loss.cpu().numpy(), self.wmse.cpu().numpy(), self.logvar.cpu().numpy()
        n = max(count.sum(), 1)
        out = dict(loss=float(loss.sum() / n), wmse=float(wmse.sum() / n), logvar=float(logvar.sum() / n), by_sigma={})
        for i, name in enumerate(SIGMA_BUCKET_NAMES):
            if count[i] > 0:
                out['by_sigma'][name] = dict(n=int(count[i]), loss=float(loss[i] / count[i]), wmse=float(wmse[i] / count[i]), logvar=float(logvar[i] / count[i]))
        return out
