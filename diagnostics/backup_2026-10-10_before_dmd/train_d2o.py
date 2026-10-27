"""Stage 1: D2O. Fine-tune the pre-trained EDM into a one-step (or few-step) generator with a GAN objective only.

Follows the official D2O training loop and loss (``training/training_loop.py``
and ``training/loss.py``): generator and EMA initialised from the EDM EMA
weights, fixed input noise level, non-saturating GAN loss without any
distillation term, generator step then discriminator step, lazy R1, Adam with
betas (0, 0.99), and the optional name-based freezing of D2O-F. The
discriminator is described in ``d2o/discriminator.py``.

``--sigmas`` extends the same objective to the few-step generator
(``ModelFewNFE``): every sample is trained at one position of the sampling
chain, and the positions before it are simulated with the current generator
(the backward simulation of DMD2, Yin et al., NeurIPS 2024), so the training
inputs are the ones met at sampling time.

``--imle_ratio`` adds a coverage term to the generator step, after Implicit
Maximum Likelihood Estimation (Li & Malik, 2018): real images of a class are
matched one-to-one to the nearest of freshly generated samples of that class,
and each matched sample is pulled towards its image. The distance is read in
pixels and in the discriminator's frozen feature network, so no further model
is involved. It is off by default.

``--ect_ratio`` adds the consistency term of Easy Consistency Tuning (Geng et
al., "Consistency Models Made Easy", 2024), ported from ``training/loss.py``
(``ECMLoss``) of the official code, https://github.com/locuslab/ect: a real
image is noised to two levels ``t > r`` with the same noise, and the
generator's output at ``t`` is pulled towards its own output at ``r`` (no
gradient through the target), weighted by ``1 / (t - r)``. The generator is
the only network involved. It is off by default.

``--loss_type rp`` replaces the non-saturating loss with the relativistic
pairing loss (Jolicoeur-Martineau, 2019), and ``--r2_gamma`` adds the
zero-centred gradient penalty on generated images next to R1, the two
regularizers that R3GAN (Huang et al., "The GAN is Dead; Long Live the GAN!",
NeurIPS 2024) combines with it. Every real image is compared with a generated
image of its class at the same discriminator noise level.

``--kd_ratio`` adds a paired distillation term: (noise, class, image) triples
that a multi-step teacher produced offline with the deterministic sampler
(``diagnostics/teammate_edm2/generate_teacher_samples.py``) are regression
targets for the one-step generator, next to the GAN loss, as in Diffusion2GAN
(Kang et al., "Distilling Diffusion Models into Conditional GANs", ECCV 2024).
The teacher itself is not loaded here. It is off by default.
"""

import argparse
import glob
import math
import os
import shutil
import time

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from d2o.augment import AugmentPipe
from d2o.config import GENERATOR_KWARGS, SIGMA_G
from d2o.data import InfiniteBatches, load_train_tensors
from d2o.discriminator import EDM2FeatureNetwork, ProjectedDiscriminator
from d2o.edm2 import EDM2Encoder, EDM2Precond
from d2o.networks import D2OPrecond, EDMPrecond
from d2o.sampling import one_hot, save_image_grid
from d2o.training_utils import JsonlLogger, RunningMean, StopSignal, apply_freeze, sanitize_gradients, update_ema
from model import ModelFewNFE, ModelOneNFE


def generator_loss(logits, loss_type):
    return F.softplus(-logits).mean() if loss_type == "ns" else -logits.mean()


def discriminator_loss(logits_fake, logits_real, loss_type):
    if loss_type == "rp":
        return F.softplus(logits_fake - logits_real).mean()  # relativistic pairing: real i against fake i
    if loss_type == "ns":
        return F.softplus(logits_fake).mean() + F.softplus(-logits_real).mean()
    return F.relu(1 + logits_fake).mean() + F.relu(1 - logits_real).mean()


def intra_class_distance(images):
    """Mean pairwise RMS distance between images of the same class, ``images``: [classes, samples, ...]."""
    flat = images.flatten(2).to(torch.float32)
    return torch.cdist(flat, flat).mean().item() / flat.shape[-1] ** 0.5


def main(args):
    state_path = os.path.join(args.outdir, "d2o-state-latest.pt")
    if not args.resume and (os.path.exists(state_path) or os.path.exists(os.path.join(args.outdir, "best.ckpt"))):
        # A fresh start evaluates the untrained generator first, which would replace best.ckpt and the resumable state.
        raise SystemExit(f"{args.outdir} already holds a run: continue it with --resume {state_path} or use another --outdir.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.outdir, exist_ok=True)
    assert args.batch % args.batch_gpu == 0
    rounds = args.batch // args.batch_gpu

    images, labels = load_train_tensors(args.data_root, args.split_dir)
    batches = InfiniteBatches(images, labels, args.batch_gpu, device, xflip=args.xflip, seed=args.seed)
    # Optional second source of "real" images for the discriminator: offline samples of a stronger teacher (--extra_real).
    extra_batches = None
    if args.extra_real:
        shards = [torch.load(path, weights_only=True) for path in sorted(glob.glob(os.path.join(args.extra_real, "shard-*.pt")))]
        extra_images, extra_labels = torch.cat([shard["images"] for shard in shards]), torch.cat([shard["labels"] for shard in shards])
        extra_batches = InfiniteBatches(extra_images, extra_labels, args.batch_gpu, device, xflip=args.xflip, seed=args.seed + 1)
        print(f"Extra real images: {len(extra_images):,} from {args.extra_real}, each real image replaced with probability {args.extra_real_p}")
        del shards, extra_images, extra_labels

    # Generator and its EMA, both initialised from the pre-trained EDM (official --transfer).
    if args.g_edm2:
        # --g_edm2: an EDM2 denoiser (the teammate's teacher) is the pre-trained model instead of our EDM.
        edm_checkpoint = None
        net = EDM2Precond.from_checkpoint(args.g_edm2, use_fp16=bool(args.fp16)).to(device)
        net.encoder_frozen = bool(args.freeze_encoder)
        net_kwargs = dict(family="edm2", **net.network_kwargs)
        init_state = {name: tensor.clone() for name, tensor in net.state_dict().items()}
    else:
        edm_checkpoint = torch.load(args.edm, map_location="cpu", weights_only=False)
        net_kwargs = edm_checkpoint["net_kwargs"]
        if net_kwargs != GENERATOR_KWARGS:
            print("WARNING: architecture differs from d2o/config.py; update GENERATOR_KWARGS before submitting this run.")
        net = D2OPrecond(**net_kwargs, use_fp16=bool(args.fp16)).to(device)
        init_state = edm_checkpoint["ema"]
        net.load_state_dict(init_state)
    net.train()
    # The EMA generator lives in a ModelOneNFE / ModelFewNFE, so snapshots are directly loadable by evaluate.py.
    if args.sigmas:
        ema_model = ModelFewNFE(device=device, sigmas=args.sigmas, **net_kwargs)
    else:
        ema_model = ModelOneNFE(device=device, sigma_g=args.sigma_g, **net_kwargs)
    # Noise levels of the sampling chain, in sampling order.
    sigmas = torch.tensor(args.sigmas or [args.sigma_g], device=device)
    ema_model.backbone.load_state_dict(init_state)
    ema_model.eval().requires_grad_(False)
    del init_state

    # Frozen discriminator feature network: by default the same pre-trained EDM.
    edm = feature_checkpoint = None
    if not (args.d_shared or args.d_teacher):
        feature_checkpoint = edm_checkpoint if args.d_edm is None else torch.load(args.d_edm, map_location="cpu", weights_only=False)
        assert feature_checkpoint is not None, "--g_edm2 needs --d_shared 1, --d_teacher or --d_edm for the discriminator features"
        edm = EDMPrecond(**feature_checkpoint["net_kwargs"])
        edm.load_state_dict(feature_checkpoint["ema"])
    # --d_teacher: the features come from the encoder of an EDM2 denoiser instead (cut after --d_teacher_min_res).
    teacher_features = None
    if args.d_shared:
        # --d_shared: the features come from the generator's own encoder, which --freeze_encoder keeps as pre-trained.
        # No second encoder is held: generator and discriminator together are the generator plus the heads.
        # --d_shared 2: the same without freezing. The generator trains its whole encoder, and the discriminator reads the
        # features of that encoder as it is at every step, as constants (EDM2Precond.features_fixed).
        assert args.g_edm2 and not args.d_teacher, "--d_shared needs --g_edm2 and no --d_teacher"
        assert (args.d_shared == 2) != bool(args.freeze_encoder), "--d_shared 1 goes with --freeze_encoder 1, --d_shared 2 with --freeze_encoder 0"
        teacher_features = EDM2FeatureNetwork(net, cout=int(round(64 * args.d_width)), shared=True, live=args.d_shared == 2)
    elif args.d_teacher:
        teacher_features = EDM2FeatureNetwork(EDM2Encoder.from_checkpoint(args.d_teacher, min_resolution=args.d_teacher_min_res), cout=int(round(64 * args.d_width)))
    discriminator = ProjectedDiscriminator(
        edm, c_dim=net.label_dim, proj_type=args.d_proj, sigma_mean=args.d_sigma_mean, sigma_std=args.d_sigma_std,
        trainable_backbone=args.d_backbone_lr_mult > 0, feature_network=teacher_features, width=args.d_width,
    ).to(device)
    discriminator.feature_network.edm.use_fp16 = bool(args.fp16)
    del edm, edm_checkpoint, feature_checkpoint

    g_params = apply_freeze(net, args.freeze, freeze_encoder=bool(args.freeze_encoder))
    head_params = list(discriminator.heads.parameters())
    backbone_params = [param for param in discriminator.feature_network.parameters() if param.requires_grad]
    d_params = head_params + backbone_params
    count = lambda params: sum(p.numel() for p in params)
    print(f"Generator: {count(net.parameters()):,} parameters, {count(g_params):,} trainable (freeze={args.freeze}, freeze_encoder={args.freeze_encoder})")
    print(f"Discriminator: {count(discriminator.parameters()):,} parameters, {count(d_params):,} trainable"
          + (f" (features from the generator's {'frozen' if args.d_shared == 1 else 'own, trained'} encoder, not counted again)" if args.d_shared else ""))
    print(f"Generator + discriminator: {count(net.parameters()) + count(discriminator.parameters()):,} parameters")

    use_r1 = args.r1_gamma > 0
    lazy = args.r1_interval / (args.r1_interval + 1) if use_r1 else 1
    g_optimizer = torch.optim.Adam(g_params, lr=args.lr, betas=(0.0, 0.99), eps=1e-8)
    d_groups = [dict(params=head_params, lr=args.dlr * lazy)]
    if backbone_params:
        d_groups.append(dict(params=backbone_params, lr=args.dlr * lazy * args.d_backbone_lr_mult))
    d_optimizer = torch.optim.Adam(d_groups, betas=(0.0, 0.99 ** lazy), eps=1e-8)

    augment_pipe = None
    if args.augment_p > 0 or args.ada_target > 0:
        augment_pipe = AugmentPipe().to(device)
        augment_pipe.p.fill_(args.augment_p)

    cur_nimg, batch_idx, best_fid = 0, 0, float("inf")
    d_is_new = False  # the state was written with another discriminator (the feature network was switched)
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        ema_model.load_state_dict(state["ema_model"])
        g_optimizer.load_state_dict(state["g_optimizer"])
        own = discriminator.state_dict()
        d_is_new = {name: tensor.shape for name, tensor in own.items()} != {name: tensor.shape for name, tensor in state["discriminator"].items()}
        if d_is_new:
            print("The state holds a different discriminator: this one and its optimizer start fresh.")
        else:
            discriminator.load_state_dict(state["discriminator"])
            d_optimizer.load_state_dict(state["d_optimizer"])
        # The optimizer states carry the learning rates they were saved with; --lr / --dlr of this run take over.
        for group, lr in zip(g_optimizer.param_groups + d_optimizer.param_groups, [args.lr] + [group["lr"] for group in d_groups]):
            if group["lr"] != lr:
                print(f"Learning rate changed on resume: {group['lr']:.3g} -> {lr:.3g}")
                group["lr"] = lr
        if augment_pipe is not None and args.ada_target > 0:
            augment_pipe.p.fill_(state["augment_p"])  # adaptive: continue from the saved value; fixed: --augment_p of this run
        cur_nimg, batch_idx, best_fid = state["cur_nimg"], state["batch_idx"], state["best_fid"]
        print(f"Resumed from {args.resume} at {cur_nimg / 1e3:.0f} kimg")

    def run_G(batch_labels):
        """One generator evaluation per sample, at a random position of the sampling chain."""
        class_labels = one_hot(batch_labels, net.label_dim)
        x = torch.randn([len(batch_labels), 3, net.img_resolution, net.img_resolution], device=device) * sigmas[0]
        step = torch.zeros(len(batch_labels), dtype=torch.long, device=device)
        if len(sigmas) > 1:
            step = torch.randperm(len(batch_labels), device=device) % len(sigmas)  # every position equally often in a batch
        # Backward simulation: the samples trained at a later position first run the chain up to it, without gradients.
        for k in range(1, len(sigmas)):
            active = step >= k
            with torch.no_grad():
                denoised = net(x[active], sigmas[k - 1].expand(int(active.sum())), class_labels=class_labels[active])
            x[active] = denoised + sigmas[k] * torch.randn_like(denoised)
        return net(x, sigmas[step], class_labels=class_labels)

    def next_real():
        """One batch of real images; with --extra_real every image is swapped for an extra one with probability --extra_real_p."""
        real, real_labels = next(batches)
        if extra_batches is not None:
            extra, extra_labels = next(extra_batches)
            take = torch.rand(len(real), device=device) < args.extra_real_p
            real = torch.where(take[:, None, None, None], extra, real)
            real_labels = torch.where(take, extra_labels, real_labels)
        return real, real_labels

    def run_D_features(batch_images, batch_labels, augment=True, force_fp32=False):
        if augment_pipe is not None and augment:
            batch_images = augment_pipe(batch_images)
        return discriminator.extract(batch_images, batch_labels, force_fp32=force_fp32)

    def run_D(batch_images, batch_labels, augment=True, force_fp32=False):
        return discriminator.heads(run_D_features(batch_images, batch_labels, augment, force_fp32), batch_labels)

    def run_D_pair(fake, real, batch_labels, real_grad=True):
        """Logits of a generated and a real batch read at the same noise levels (--loss_type rp compares them pairwise)."""
        if augment_pipe is not None:
            fake, real = augment_pipe(fake), augment_pipe(real)
        sigma = discriminator.sample_sigma(len(real), device)
        class_labels = one_hot(batch_labels, discriminator.c_dim)
        logits = lambda images: discriminator.heads(discriminator.feature_network(images + sigma * torch.randn_like(images), sigma, class_labels), batch_labels)
        with torch.set_grad_enabled(real_grad):
            logits_real = logits(real)
        return logits(fake), logits_real

    def image_distance(fake, targets, class_labels):
        """Pixel L1 plus, weighted by --imle_feat, the MSE of the frozen features of both images under the same noise."""
        fake = fake.to(torch.float32)
        pixel = (fake - targets).abs().mean()
        sigma = torch.full([len(fake), 1, 1, 1], args.imle_sigma, device=device)
        noise = torch.randn_like(fake) * sigma
        with torch.no_grad():
            real_feats = frozen_features(targets + noise, sigma, class_labels)
        fake_feats = frozen_features(fake + noise, sigma, class_labels)
        feat = sum(F.mse_loss(a, b) for a, b in zip(fake_feats, real_feats)) / len(real_feats)
        return pixel + args.imle_feat * feat, pixel.detach(), feat.detach()

    # Paired distillation (--kd_ratio): the teacher's (latent, class, image) triples, the latents rebuilt from the shard seeds.
    kd_images = kd_labels = kd_latents = None
    if args.kd_ratio > 0:
        assert len(sigmas) == 1 and args.kd_pairs and 0 < args.kd_rounds, "--kd_ratio needs --kd_pairs and is written for the one-step generator"
        shards = [torch.load(path, weights_only=True) for path in sorted(glob.glob(os.path.join(args.kd_pairs, "shard-*.pt")))]
        assert shards, f"no shard-*.pt in {args.kd_pairs}"
        kd_images = torch.cat([shard["images"] for shard in shards]).to(device)
        kd_labels = torch.cat([shard["labels"] for shard in shards]).to(device)
        # generate_teacher_samples.py drew the latents of a shard from a CPU generator seeded with the stored seed.
        kd_latents = torch.cat([torch.randn(list(shard["images"].shape), generator=torch.Generator().manual_seed(int(shard["seed"]))).to(torch.float16)
                                for shard in shards]).to(device)
        print(f"Distillation pairs: {len(kd_images):,} from {args.kd_pairs} (teacher {shards[0].get('teacher')}, {shards[0].get('steps')} Heun steps)")
        del shards

    def distillation_gradients():
        """Gradient of the paired regression loss with respect to the trainable generator parameters, and its statistics."""
        total, values = None, dict(loss_kd=0.0, kd_pixel=0.0, kd_feat=0.0)
        for _ in range(args.kd_rounds):
            rows = torch.randint(len(kd_images), [args.batch_gpu], device=device)
            class_labels = one_hot(kd_labels[rows], net.label_dim)
            targets = kd_images[rows].to(torch.float32) / 127.5 - 1
            fake = net(kd_latents[rows].to(torch.float32) * sigmas[0], sigmas[0].expand(len(rows)), class_labels=class_labels)
            loss, pixel, feat = image_distance(fake, targets, class_labels)
            grads = torch.autograd.grad(loss / args.kd_rounds, g_params, allow_unused=True)
            total = grads if total is None else [a if b is None else b if a is None else a + b for a, b in zip(total, grads)]
            for key, value in zip(values, (loss.detach(), pixel, feat)):
                values[key] += float(value) / args.kd_rounds
        return total, values

    # Coverage term (--imle_ratio): the training images by class, and the frozen features both images are compared in.
    class_rows = None
    if args.imle_ratio > 0:
        assert len(sigmas) == 1, "--imle_ratio is written for the one-step generator"
        class_rows = [torch.nonzero(labels == c).flatten().to(device) for c in range(net.label_dim)]
        assert min(len(rows) for rows in class_rows) >= args.imle_reals and args.imle_pool >= args.imle_reals

    def frozen_features(noisy, sigma, class_labels):
        feature_network = discriminator.feature_network
        read = feature_network.edm.features_fixed if getattr(feature_network, "live", False) else feature_network.edm.features
        feats = read(noisy, sigma, class_labels)
        return [feat.to(torch.float32) for feat in (feats.values() if isinstance(feats, dict) else feats)]

    def coverage_loss():
        """IMLE: every chosen real image pulls the nearest of the freshly generated samples of its class towards itself."""
        resolution = net.img_resolution
        targets, target_labels, latents, match = [], [], [], []
        for c in torch.randperm(net.label_dim, device=device)[: args.imle_classes].tolist():
            rows = class_rows[c][torch.randperm(len(class_rows[c]), device=device)[: args.imle_reals]]
            real = batches.images[rows].to(torch.float32) / 127.5 - 1
            if args.xflip:
                real = torch.where(torch.rand(len(rows), device=device)[:, None, None, None] < 0.5, real.flip(3), real)
            pool_labels = torch.full([args.imle_pool], c, device=device)
            z = torch.randn([args.imle_pool, 3, resolution, resolution], device=device)
            with torch.no_grad():
                candidates = net(z * sigmas[0], sigmas[0].expand(args.imle_pool), class_labels=one_hot(pool_labels, net.label_dim))
                # Matched on 16x16 averages: layout and colour decide which sample belongs to which image.
                cost = torch.cdist(F.avg_pool2d(real, 4).flatten(1), F.avg_pool2d(candidates.to(torch.float32), 4).flatten(1))
            _, picked = linear_sum_assignment(cost.cpu().numpy())  # one-to-one: no sample is pulled towards two images
            picked = torch.as_tensor(picked, device=device)
            targets.append(real); target_labels.append(pool_labels[: len(rows)]); latents.append(z[picked])
            match.append(cost[torch.arange(len(rows), device=device), picked])
        targets, target_labels, latents = torch.cat(targets), torch.cat(target_labels), torch.cat(latents)
        class_labels = one_hot(target_labels, net.label_dim)
        fake = net(latents * sigmas[0], sigmas[0].expand(len(latents)), class_labels=class_labels).to(torch.float32)
        pixel = (fake - targets).abs().mean()
        # Both images are read under the same noise, so that the feature distance is a distance between the images.
        sigma = torch.full([len(fake), 1, 1, 1], args.imle_sigma, device=device)
        noise = torch.randn_like(fake) * sigma
        with torch.no_grad():
            real_feats = frozen_features(targets + noise, sigma, class_labels)
        fake_feats = frozen_features(fake + noise, sigma, class_labels)
        feat = sum(F.mse_loss(a, b) for a, b in zip(fake_feats, real_feats)) / len(real_feats)
        return pixel + args.imle_feat * feat, dict(imle_pixel=pixel.detach(), imle_feat=feat.detach(), imle_match=torch.cat(match).mean())

    # Consistency term (--ect_ratio), after ECMLoss of the official ECT code.
    if args.ect_ratio > 0:
        assert len(sigmas) == 1 and 0 < args.ect_rounds <= rounds, "--ect_ratio is written for the one-step generator; --ect_rounds at most batch / batch_gpu"
    dropout_modules = [module for module in net.modules() if isinstance(getattr(module, "dropout", None), (int, float))]
    dropout_values = [module.dropout for module in dropout_modules]

    def ect_stage():
        """How often dt has been divided by q so far (ECT: update_schedule, every --double ticks)."""
        if args.ect_double_kimg <= 0:
            return 0
        return int(max(cur_nimg / 1000 - args.ect_from_kimg, 0) // args.ect_double_kimg)

    def ect_r(t):
        """Mapping function t -> r (ECMLoss.t_to_r_sigmoid / t_to_r_const): r = t * (1 - adjust(t) / q^(stage + 1))."""
        decay = 1 / args.ect_q ** (ect_stage() + 1)
        adjust = 1 + args.ect_k * torch.sigmoid(-args.ect_b * t) if args.ect_mapping == "sigmoid" else 1
        return (t * (1 - decay * adjust)).clamp(min=0)

    def consistency_gradients(real_batches):
        """Gradient of the ECT loss with respect to the trainable generator parameters, and the loss value."""
        total, value = None, 0.0
        for real, real_labels in real_batches[: args.ect_rounds]:
            class_labels = one_hot(real_labels, net.label_dim)
            # t ~ p(t), log-normal as in ECT, capped at the generator's input noise level; r from the mapping function.
            t = (torch.randn([len(real), 1, 1, 1], device=device) * args.ect_std + args.ect_mean).exp().clamp(max=float(sigmas[0]))
            r = ect_r(t)
            eps = torch.randn_like(real)  # shared noise direction
            rng_state = torch.cuda.get_rng_state(device) if (args.ect_dropout > 0 and device.type == "cuda") else None
            for module in dropout_modules:
                module.dropout = args.ect_dropout
            try:
                # The target pass comes first: in training mode every pass renormalizes the weights in place, which must
                # not happen between the trained pass and its backward pass. Both passes run in float32: the two outputs
                # differ by a small fraction of an image, which bfloat16 does not resolve.
                with torch.no_grad():
                    target = net(real + eps * r, r.flatten(), class_labels=class_labels, force_fp32=True)
                    target = torch.where(r > 0, torch.nan_to_num(target), real)  # r = 0: the image itself
                if rng_state is not None:
                    torch.cuda.set_rng_state(rng_state, device)  # shared dropout mask
                pred = net(real + eps * t, t.flatten(), class_labels=class_labels, force_fp32=True)
            finally:
                for module, value in zip(dropout_modules, dropout_values):
                    module.dropout = value
            # L2 distance, pseudo-Huber with c (c = 0: the plain norm, the official default), weighted by 1 / (t - r).
            distance = (pred - target).square().flatten(1).sum(1)
            distance = (distance + args.ect_c ** 2).clamp_min(1e-12).sqrt() - args.ect_c
            loss = (distance / (t - r).flatten()).mean() / args.ect_rounds
            grads = torch.autograd.grad(loss, g_params, allow_unused=True)
            total = grads if total is None else [a if b is None else b if a is None else a + b for a, b in zip(total, grads)]
            value += float(loss.detach())
        return total, value

    monitor = None
    if args.fid:
        from d2o.fid import FIDMonitor
        monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.fid_reference, device=device)

    # Fixed probes: a sample grid and an intra-class diversity estimate (8 classes x 8 samples).
    grid_categories = torch.arange(64, device=device) % net.label_dim
    probe_categories = torch.arange(8, device=device).repeat_interleave(8)
    real_probe = torch.stack([images[labels == c][:8] for c in range(8)]).to(device).to(torch.float32) / 127.5 - 1
    print(f"Intra-class distance of real images: {intra_class_distance(real_probe):.4f}")

    logger = JsonlLogger(os.path.join(args.outdir, "log.jsonl"))
    stats = RunningMean()
    ada_signs = RunningMean()
    stop = StopSignal()

    def evaluate_and_save():
        nonlocal best_fid
        kimg = cur_nimg // 1000
        torch.manual_seed(0)  # same latents at every snapshot, so the grids and FID values are comparable
        save_image_grid(ema_model.sample((64, 3, 64, 64), device=device, category=grid_categories), os.path.join(args.outdir, f"samples-{kimg:06d}.png"))
        probe = ema_model.sample((64, 3, 64, 64), device=device, category=probe_categories)
        fields = dict(kimg=round(cur_nimg / 1e3, 1), intra_class_distance=round(intra_class_distance(probe.reshape(8, 8, -1)), 4))
        snapshot_path = os.path.join(args.outdir, f"d2o-{kimg:06d}.ckpt")
        torch.save(dict(state_dict=ema_model.state_dict(), net_kwargs=net_kwargs, cur_nimg=cur_nimg, args=vars(args)), snapshot_path)
        if monitor is not None:
            fields["fid_train_ref"] = round(monitor(ema_model, os.path.join(args.outdir, "fid_samples")), 3)
            if fields["fid_train_ref"] < best_fid:
                best_fid = fields["fid_train_ref"]
                shutil.copyfile(snapshot_path, os.path.join(args.outdir, "best.ckpt"))
            fields["best_fid_train_ref"] = round(best_fid, 3)
            if not args.keep_snapshots:
                os.remove(snapshot_path)  # best.ckpt and the resumable state remain
        torch.manual_seed(args.seed + cur_nimg)
        logger.log(**fields)
        print("EVAL " + " ".join(f"{key} {value}" for key, value in fields.items()), flush=True)
        torch.save(
            dict(
                net=net.state_dict(), ema_model=ema_model.state_dict(), discriminator=discriminator.state_dict(),
                g_optimizer=g_optimizer.state_dict(), d_optimizer=d_optimizer.state_dict(),
                augment_p=float(augment_pipe.p) if augment_pipe is not None else 0.0,
                cur_nimg=cur_nimg, batch_idx=batch_idx, best_fid=best_fid, net_kwargs=net_kwargs, args=vars(args),
            ),
            state_path,
        )

    if d_is_new and args.d_warmup_kimg > 0:
        # A fresh discriminator would hand the trained generator meaningless gradients: train it alone first.
        # These images do not count as training progress (cur_nimg), and nothing is saved before the main loop.
        print(f"Discriminator warm-up: {args.d_warmup_kimg:g} kimg with the generator frozen", flush=True)
        warmup = RunningMean()
        for step in range(1, int(args.d_warmup_kimg * 1000) // args.batch + 1):
            data = [next_real() for _ in range(rounds)]
            d_optimizer.zero_grad(set_to_none=True)
            for real, real_labels in data:
                with torch.no_grad():
                    fake = run_G(real_labels)
                logits_fake, logits_real = run_D(fake, real_labels), run_D(real, real_labels)
                loss_D = discriminator_loss(logits_fake, logits_real, args.loss_type)
                (loss_D / rounds).backward()
                warmup.add(loss_D=loss_D.detach(), logits_real=logits_real.detach().mean(), logits_fake=logits_fake.detach().mean())
            sanitize_gradients(d_params)
            d_optimizer.step()
            if use_r1 and step % args.r1_interval == 0:
                d_optimizer.zero_grad(set_to_none=True)
                for real, real_labels in data:
                    real = real[: args.r1_batch].detach().requires_grad_(True)
                    logits = run_D(real, real_labels[: args.r1_batch], augment=False, force_fp32=True)
                    (grads,) = torch.autograd.grad(outputs=[logits.mean(dim=1).sum()], inputs=[real], create_graph=True)
                    (grads.square().sum(dim=[1, 2, 3]).mean() * (args.r1_gamma / 2) * args.r1_interval / rounds).backward()
                sanitize_gradients(d_params)
                d_optimizer.step()
            if step * args.batch % 5000 < args.batch or stop.requested:
                print(f"warm-up kimg {step * args.batch / 1e3:.1f} " + " ".join(f"{key} {value:.4f}" for key, value in warmup.pop().items()), flush=True)
            if stop.requested:
                break

    total_nimg = int(args.kimg * 1000)
    next_log = (cur_nimg // (args.log_kimg * 1000) + 1) * args.log_kimg * 1000
    next_snap = (cur_nimg // (args.snap_kimg * 1000) + 1) * args.snap_kimg * 1000
    if cur_nimg == 0 and args.eval_init:
        evaluate_and_save()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tick_time, tick_nimg = time.time(), cur_nimg
    while cur_nimg < total_nimg and not stop.requested:
        data = [next_real() for _ in range(rounds)]

        # Generator step.
        for param in d_params:
            param.requires_grad_(False)
        g_optimizer.zero_grad(set_to_none=True)
        reused = []  # per round: what the discriminator step can take over from this step (--reuse_fake)
        for real, real_labels in data:
            fake = run_G(real_labels)
            if args.loss_type == "rp":
                logits_fake, logits_real = run_D_pair(fake, real, real_labels, real_grad=False)
                loss_G = F.softplus(logits_real - logits_fake).mean()
            else:
                features = run_D_features(fake, real_labels)
                loss_G = generator_loss(discriminator.heads(features, real_labels), args.loss_type)
            (loss_G / rounds).backward()
            stats.add(loss_G=loss_G.detach())
            if args.reuse_fake:
                # A frozen feature network gives the same features in both steps; a trainable one needs its own graph.
                reused.append(fake.detach() if backbone_params else {key: value.detach() for key, value in features.items()})
        if args.imle_ratio > 0 or args.ect_ratio > 0 or args.kd_ratio > 0:
            # Adam follows the direction of the summed gradient, so an auxiliary term is weighted by gradient norm, not by
            # loss value: its gradient is rescaled to its ratio times the norm of the GAN gradient of this step (the
            # adaptive weighting between a consistency loss and a GAN loss of CTM, Kim et al., 2024, over all parameters).
            gan_norm = torch.stack([param.grad.float().norm() for param in g_params if param.grad is not None]).norm()
            auxiliary = []
            if args.imle_ratio > 0:
                loss_cov, cov_stats = coverage_loss()
                auxiliary.append(("imle", args.imle_ratio, torch.autograd.grad(loss_cov, g_params, allow_unused=True)))
                stats.add(loss_imle=loss_cov.detach(), **cov_stats)
            if args.ect_ratio > 0:
                ect_grads, loss_ect = consistency_gradients(data)
                auxiliary.append(("ect", args.ect_ratio, ect_grads))
                stats.add(loss_ect=loss_ect)
            if args.kd_ratio > 0:
                kd_grads, kd_stats = distillation_gradients()
                auxiliary.append(("kd", args.kd_ratio, kd_grads))
                stats.add(**kd_stats)
            for name, ratio, grads in auxiliary:
                norm = torch.stack([grad.float().norm() for grad in grads if grad is not None]).norm()
                scale = ratio * gan_norm / norm
                stats.add(**{f"{name}_grad_ratio": norm / gan_norm})
                if bool(torch.isfinite(scale)):
                    for param, grad in zip(g_params, grads):
                        if grad is not None:
                            param.grad = grad * scale if param.grad is None else param.grad.add_(grad * scale)
        stats.add(nonfinite_grads_G=sanitize_gradients(g_params))
        g_optimizer.step()

        # Discriminator step.
        for param in d_params:
            param.requires_grad_(True)
        d_optimizer.zero_grad(set_to_none=True)
        for round_idx, (real, real_labels) in enumerate(data):
            if args.loss_type == "rp":
                with torch.no_grad():
                    fake = run_G(real_labels)
                logits_fake, logits_real = run_D_pair(fake, real, real_labels)
            else:
                if not args.reuse_fake:
                    with torch.no_grad():
                        fake = run_G(real_labels)
                    logits_fake = run_D(fake, real_labels)
                elif backbone_params:
                    logits_fake = run_D(reused[round_idx], real_labels)
                else:
                    logits_fake = discriminator.heads(reused[round_idx], real_labels)
                logits_real = run_D(real, real_labels)
            loss_D = discriminator_loss(logits_fake, logits_real, args.loss_type)
            (loss_D / rounds).backward()
            sign_real = logits_real.detach().sign().mean()
            stats.add(loss_D=loss_D.detach(), logits_real=logits_real.detach().mean(), logits_fake=logits_fake.detach().mean(),
                      sign_real=sign_real, sign_fake=logits_fake.detach().sign().mean())
            ada_signs.add(sign_real=sign_real)
        stats.add(nonfinite_grads_D=sanitize_gradients(d_params))
        d_optimizer.step()

        # Lazy R1 on real images.
        if use_r1 and batch_idx != 0 and batch_idx % args.r1_interval == 0:
            d_optimizer.zero_grad(set_to_none=True)
            for real, real_labels in data:
                real = real[: args.r1_batch].detach().requires_grad_(True)
                logits = run_D(real, real_labels[: args.r1_batch], augment=False, force_fp32=True)
                (grads,) = torch.autograd.grad(outputs=[logits.mean(dim=1).sum()], inputs=[real], create_graph=True)
                r1_penalty = grads.square().sum(dim=[1, 2, 3])
                (r1_penalty.mean() * (args.r1_gamma / 2) * args.r1_interval / rounds).backward()
                stats.add(r1_penalty=r1_penalty.detach().mean())
            if args.r2_gamma > 0:
                # R2: the same penalty on generated images (R3GAN uses R1 and R2 together with the relativistic loss).
                for _, real_labels in data:
                    with torch.no_grad():
                        fake = run_G(real_labels[: args.r1_batch])
                    fake = fake.to(torch.float32).requires_grad_(True)
                    logits = run_D(fake, real_labels[: args.r1_batch], augment=False, force_fp32=True)
                    (grads,) = torch.autograd.grad(outputs=[logits.mean(dim=1).sum()], inputs=[fake], create_graph=True)
                    r2_penalty = grads.square().sum(dim=[1, 2, 3])
                    (r2_penalty.mean() * (args.r2_gamma / 2) * args.r1_interval / rounds).backward()
                    stats.add(r2_penalty=r2_penalty.detach().mean())
            sanitize_gradients(d_params)
            d_optimizer.step()

        batch_idx += 1
        cur_nimg += args.batch
        update_ema(ema_model.backbone, net, cur_nimg, args.batch, args.ema_halflife_kimg, args.ema_rampup_ratio)

        # ADA: move p towards the target fraction of positive real logits.
        if augment_pipe is not None and args.ada_target > 0 and batch_idx % args.ada_interval == 0:
            sign = 1 if ada_signs.pop()["sign_real"] > args.ada_target else -1
            adjust = sign * (args.batch * args.ada_interval) / (args.ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).clamp(0, 1))

        done = cur_nimg >= total_nimg or stop.requested
        if cur_nimg >= next_log or done:
            now = time.time()
            fields = dict(
                kimg=round(cur_nimg / 1e3, 1),
                **{key: round(value, 4) for key, value in stats.pop().items()},
                augment_p=round(float(augment_pipe.p), 4) if augment_pipe is not None else 0.0,
                img_per_sec=round((cur_nimg - tick_nimg) / (now - tick_time), 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if device.type == "cuda" else 0.0,
            )
            logger.log(**fields)
            print(" ".join(f"{key} {value}" for key, value in fields.items()), flush=True)
            if not (math.isfinite(fields["loss_G"]) and math.isfinite(fields["loss_D"])):
                raise RuntimeError("Non-finite loss")
            tick_time, tick_nimg = time.time(), cur_nimg
            next_log += args.log_kimg * 1000

        if cur_nimg >= next_snap or done:
            eval_start = time.time()
            evaluate_and_save()
            next_snap += args.snap_kimg * 1000
            tick_time += time.time() - eval_start  # keep evaluation out of img_per_sec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D2O one-step GAN fine-tuning (stage 1)")
    parser.add_argument("--edm", default=None, help="Stage-0 snapshot (edm-*.pt) used as initialisation")
    parser.add_argument("--g_edm2", default=None, help="EDM2 checkpoint (teammate's teacher) used as the generator's initialisation instead of --edm; repeat it on --resume")
    parser.add_argument("--outdir", default="./runs/d2o")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--resume", default=None, help="d2o-state-latest.pt of a previous run")
    # Generator.
    parser.add_argument("--freeze", type=int, default=1, choices=(0, 1, 2, 3), help="Official --freeze: 0 = D2O, 1 = D2O-F (conv, proj frozen)")
    parser.add_argument("--freeze_encoder", type=int, default=0, help="1 = the embeddings and the encoder stay fixed, only the decoder is fine-tuned; repeat it on --resume")
    parser.add_argument("--sigma_g", type=float, default=SIGMA_G, help="Fixed generator noise level (official: 80)")
    parser.add_argument("--sigmas", type=lambda text: [float(value) for value in text.split(",")], default=None,
                        help="Few-step generator: comma-separated noise levels in sampling order, at most four (e.g. 80,2,0.8,0.3). Trains a ModelFewNFE and replaces --sigma_g")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ema_halflife_kimg", type=float, default=500)
    parser.add_argument("--ema_rampup_ratio", type=float, default=0.05)
    # Discriminator.
    parser.add_argument("--dlr", type=float, default=4e-5)
    parser.add_argument("--d_edm", default=None, help="Stage-0 snapshot for the discriminator features (default: same as --edm)")
    parser.add_argument("--d_teacher", default=None, help="EDM2 checkpoint whose encoder replaces the EDM encoder as the discriminator's feature network; repeat it on --resume")
    parser.add_argument("--d_teacher_min_res", type=int, default=16, help="Coarsest resolution of the --d_teacher encoder that is used")
    parser.add_argument("--d_width", type=float, default=1.0, help="Channel multiplier of the discriminator's projections and heads (parameters go with its square); repeat it on --resume")
    parser.add_argument("--d_shared", type=int, default=0, help="With --g_edm2, the discriminator reads its features from the generator's encoder instead of holding one: 1 = that encoder is frozen (--freeze_encoder 1), 2 = it is trained with the rest and read as it is at every step; repeat it on --resume")
    parser.add_argument("--d_warmup_kimg", type=float, default=0, help="When a resumed run meets a different discriminator: train the new one alone for this long first")
    parser.add_argument("--d_backbone_lr_mult", type=float, default=0.0, help="0 = frozen feature network (official); > 0 fine-tunes the encoder at dlr times this factor")
    parser.add_argument("--d_proj", type=int, default=2, choices=(0, 1, 2), help="0 = raw features, 1 = CCM, 2 = CCM + CSM (official)")
    parser.add_argument("--d_sigma_mean", type=float, default=-2.0, help="Mean of ln(sigma) of the noise added before feature extraction")
    parser.add_argument("--d_sigma_std", type=float, default=1.0, help="Std of ln(sigma); 0 = fixed noise level")
    parser.add_argument("--loss_type", default="ns", choices=("ns", "hinge", "rp"), help="ns = non-saturating (official), rp = relativistic pairing (R3GAN); not stored in the state, repeat it on --resume")
    parser.add_argument("--r2_gamma", type=float, default=0.0, help="R2 weight (gradient penalty on generated images, same form and interval as R1); needs --r1_gamma > 0; repeat it on --resume")
    parser.add_argument("--r1_gamma", type=float, default=0.02, help="R1 weight on the per-image mean logit; 0 disables R1")
    parser.add_argument("--r1_interval", type=int, default=16)
    parser.add_argument("--r1_batch", type=int, default=32, help="Images per round used for the R1 penalty")
    # Augmentation (off by default, as in the official D2O runs).
    parser.add_argument("--augment_p", type=float, default=0.0, help="Initial / fixed ADA probability")
    parser.add_argument("--ada_target", type=float, default=0.0, help="Target for E[sign(D(real))]; 0 = fixed probability")
    parser.add_argument("--ada_kimg", type=float, default=500)
    parser.add_argument("--ada_interval", type=int, default=4)
    parser.add_argument("--xflip", type=int, default=0, help="Random horizontal flips of the real images")
    parser.add_argument("--extra_real", default=None, help="Folder of shard-*.pt files (images uint8, labels) mixed into the real images; not stored in the state, repeat it on --resume")
    parser.add_argument("--extra_real_p", type=float, default=0.5, help="Probability that a real image is replaced by one from --extra_real")
    # Coverage term (off by default; none of it is stored in the state, repeat the options on --resume).
    parser.add_argument("--imle_ratio", type=float, default=0.0, help="> 0 adds the IMLE coverage term to the generator step, its gradient rescaled to this multiple of the GAN gradient's norm")
    parser.add_argument("--imle_classes", type=int, default=2, help="Classes drawn per generator step for the coverage term")
    parser.add_argument("--imle_reals", type=int, default=8, help="Real images per drawn class")
    parser.add_argument("--imle_pool", type=int, default=32, help="Freshly generated samples per drawn class that its real images are matched to")
    parser.add_argument("--imle_feat", type=float, default=1.0, help="Weight of the frozen-feature distance next to the pixel L1 distance")
    parser.add_argument("--imle_sigma", type=float, default=0.1, help="Noise level at which the frozen feature network reads both images")
    # Paired distillation from offline teacher samples (off by default; not stored in the state, repeat the options on --resume).
    parser.add_argument("--kd_ratio", type=float, default=0.0, help="> 0 adds the paired regression towards the teacher's samples to the generator step, its gradient rescaled to this multiple of the GAN gradient's norm")
    parser.add_argument("--kd_pairs", default=None, help="Folder of shard-*.pt files written by generate_teacher_samples.py (images, labels, seed of the latents)")
    parser.add_argument("--kd_rounds", type=int, default=1, help="Batches of --batch_gpu pairs per generator step")
    # Consistency term of ECT (off by default; none of it is stored in the state, repeat the options on --resume).
    parser.add_argument("--ect_ratio", type=float, default=0.0, help="> 0 adds the ECT consistency term to the generator step, its gradient rescaled to this multiple of the GAN gradient's norm")
    parser.add_argument("--ect_rounds", type=int, default=2, help="Real batches (of --batch_gpu images each) per generator step that the consistency term is computed on")
    parser.add_argument("--ect_mean", type=float, default=-1.1, help="Mean of ln(t) (ECT --mean)")
    parser.add_argument("--ect_std", type=float, default=2.0, help="Std of ln(t) (ECT --std)")
    parser.add_argument("--ect_q", type=float, default=256.0, help="dt = t * adjust / q^(stage + 1) (ECT -q). 256 without doubling is the official recipe for tuning a pre-trained diffusion model (run_ecm_1hour.sh); 2 with doubling is the full schedule")
    parser.add_argument("--ect_double_kimg", type=float, default=0.0, help="> 0: the stage goes up by one every this many kimg after --ect_from_kimg (ECT --double); 0 = dt stays as it is")
    parser.add_argument("--ect_from_kimg", type=float, default=0.0, help="kimg at which the consistency term was switched on (start of the --ect_double_kimg stages)")
    parser.add_argument("--ect_mapping", default="sigmoid", choices=("sigmoid", "const"), help="Mapping function (ECT --mapping)")
    parser.add_argument("--ect_k", type=float, default=8.0, help="Mapping function: adjust = 1 + k * sigmoid(-b * t) (ECT -k)")
    parser.add_argument("--ect_b", type=float, default=1.0, help="Mapping function (ECT -b)")
    parser.add_argument("--ect_c", type=float, default=0.0, help="Pseudo-Huber constant (ECT -c); 0 = plain L2 norm")
    parser.add_argument("--ect_dropout", type=float, default=0.0, help="Dropout during the two consistency passes, with a shared mask (ECT trains with dropout; our generator otherwise runs without)")
    # Schedule.
    parser.add_argument("--kimg", type=float, default=1000)
    parser.add_argument("--batch", type=int, default=64, help="Images per optimizer step")
    parser.add_argument("--batch_gpu", type=int, default=32, help="Images per forward pass; batch / batch_gpu accumulation rounds")
    parser.add_argument("--reuse_fake", type=int, default=0, help="1 = the discriminator step reuses the generator step's fake batch instead of sampling a new one (official: 0). 30%% faster, but FID was 8 worse at 150 kimg, so it stays off")
    parser.add_argument("--fp16", type=int, default=1, help="Run the U-Nets in bfloat16, as the official D2O code does")
    parser.add_argument("--log_kimg", type=float, default=2)
    parser.add_argument("--snap_kimg", type=float, default=20)
    parser.add_argument("--fid", type=int, default=1, help="Compute FID at every snapshot and keep the best one as best.ckpt")
    parser.add_argument("--keep_snapshots", type=int, default=1, help="0 = delete each d2o-*.ckpt once it is scored (139 MB each); needs --fid 1")
    parser.add_argument("--fid_reference", default="./results/reference_train", help="Reference PNG folder, built from the train split if missing")
    parser.add_argument("--eval_init", type=int, default=1, help="Evaluate the initial generator before training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.edm or args.g_edm2, "--edm or --g_edm2 is required"
    assert not (args.loss_type == "rp" and args.reuse_fake), "--loss_type rp does not go with --reuse_fake"
    assert args.r2_gamma == 0 or args.r1_gamma > 0, "--r2_gamma shares the lazy interval of R1"
    main(args)
