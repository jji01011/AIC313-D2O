import numpy as np
import torch
import torch.nn.functional as F

from d2o.config import NUM_CLASSES


def one_hot(category, num_classes=NUM_CLASSES):
    return F.one_hot(category.long(), num_classes).to(torch.float32)


@torch.no_grad()
def one_step_sampler(net, latents, class_labels, sigma=80.0):
    sigma = torch.full([latents.shape[0], 1, 1, 1], sigma, device=latents.device)
    return net(latents * sigma, sigma, class_labels=class_labels, force_fp32=True)


@torch.no_grad()
def edm_sampler(net, latents, class_labels, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        sigma = t_cur.expand(latents.shape[0])
        denoised = net(x_cur, sigma, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if i < num_steps - 1:
            sigma = t_next.expand(latents.shape[0])
            denoised = net(x_next, sigma, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.to(torch.float32)


def save_image_grid(images, path, columns=None):
    from PIL import Image

    images = (images.detach().float().cpu().clamp(-1, 1) * 127.5 + 127.5).round().to(torch.uint8)
    n, c, h, w = images.shape
    columns = columns or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / columns))
    grid = torch.full([c, rows * h, columns * w], 255, dtype=torch.uint8)
    for index, image in enumerate(images):
        y, x = divmod(index, columns)
        grid[:, y * h : (y + 1) * h, x * w : (x + 1) * w] = image
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(path)
