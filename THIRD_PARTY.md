# Third-party code used by the teacher

The teacher code (`teacher/`) runs the official implementations of every open-source method it uses.
Those files are vendored unmodified under `third_party/` at the commits below, with their licences.
`python -m teacher.verify_third_party` checks every vendored file against the SHA-256 in
`third_party/MANIFEST.json`.

| Repository | Commit | Licence | Files |
|---|---|---|---|
| [NVlabs/edm2](https://github.com/NVlabs/edm2) | `4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc` | CC BY-NC-SA 4.0 | `training/networks_edm2.py`, `training/phema.py`, `training/training_loop.py`, `generate_images.py`, `reconstruct_phema.py`, `dnnlib/`, `torch_utils/` |
| [NVlabs/edm](https://github.com/NVlabs/edm) | `008a4e5316c8e3bfe61a62f874bddba254295afb` | CC BY-NC-SA 4.0 | `training/augment.py`, `training/loss.py`, `training/networks.py` |

The course code (dataset.py, download_dataset.py, evaluate.py, model.py, src/, data/) is the
[course repository](https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen) at commit `3ccf326`,
unmodified.

## One-NFE generator code (`d2o/`)

`d2o/` holds three files copied unchanged from the teammate's repository `jji01011/AIC313-D2O` (branch
`d2o-handoff`, commit `32f0704`); `ModelOneNFE` uses `GENERATOR_EDM2_KWARGS` and `EDM2Precond` from them.
The training code of that repository (a port of the official D2O code,
[Zyriix/D2O](https://github.com/Zyriix/D2O) at `8857b2dd09c51384974948e1a190773e95f3c09b`, with a DMD2-style
term) is cited in TRAINING.md and is not needed for sampling.

| File | Origin | Licence |
|---|---|---|
| `d2o/edm2.py` | Port of `training/networks_edm2.py` from [NVlabs/edm2](https://github.com/NVlabs/edm2) at `4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc` (blocks keep the official names), plus the EDM2 preconditioning wrapper that reads the teammate's teacher checkpoint format | CC BY-NC-SA 4.0 |
| `d2o/config.py`, `d2o/__init__.py` | Teammate's configuration | - |

## What is used from each file

`teacher/official.py` loads the vendored files by path and re-exports the pieces below; nothing in
`third_party/` is edited.

**EDM2** (Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models", CVPR 2024)

| Official code | Used for | Where |
|---|---|---|
| `networks_edm2.py`: `MPFourier`, `MPConv`, `normalize`, `resample`, `mp_silu`, `mp_sum`, `mp_cat` | Building blocks of the teacher, imported unchanged | `teacher/networks.py` |
| `networks_edm2.py`: `Block`, `UNet`, `Precond` | Copied into `teacher/networks.py` as `Block`, `EDM2UNet`, `EDM2Precond`; every changed line is marked `# FastGen:`. Tests compare them with the official classes: same initialisation, bit-identical outputs | `teacher/networks.py`, `teacher/tests/test_networks.py` |
| `training_loop.py`: `EDM2Loss`, `learning_rate_schedule` | The loss (re-expressed with a stratified sigma, equal to `EDM2Loss` for the same draws) and the learning-rate schedule (called directly) | `teacher/losses.py`, `teacher/training.py` |
| `phema.py`: `PowerFunctionEMA`, `solve_posthoc_coefficients` | The two tracked EMA profiles during training and post-hoc EMA reconstruction | `teacher/training.py`, `teacher/phema_tools.py` |
| `generate_images.py`: `edm_sampler`, `StackedRandomGenerator` | All sampling (Heun, 32 steps, sigma 0.002 to 80, rho 7), with autoguidance through its `gnet`/`guidance` arguments | `teacher/sampler.py` |
| `torch_utils/misc.py`: `InfiniteSampler`, `set_random_seed` | The order of training images and per-step seeding | `teacher/data.py`, `teacher/training.py` |
| `reconstruct_phema.py` | Reference for `teacher/reconstruct.py`, which applies the same solver to this project's snapshot files | `teacher/phema_tools.py` |

**EDM** (Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022)

| Official code | Used for | Where |
|---|---|---|
| `augment.py`: `AugmentPipe` | Non-leaky augmentation with p = 0.15 and EDM's AFHQv2 settings, colour transforms off | `teacher/training.py` |
| `networks.py`: `SongUNet`, `EDMPrecond` | The DDPM++ insurance arm: `SongUNet` used unchanged; `EDMPrecond` copied as `DDPMppPrecond` (null class, mean-centring, precision) and tested against the official class | `teacher/networks.py` |
| `loss.py`: `EDMLoss` | The DDPM++ arm's loss (re-expressed for the stratified sigma; tested equal) | `teacher/losses.py` |

**clean-fid** (Parmar et al., "On Aliased Resizing and Surprising Subtleties in GAN Evaluation", CVPR 2022)
is the course's own FID dependency (`requirements.txt`). The proxy FID calls its `build_feature_extractor`,
`get_folder_features`, `frechet_distance` and `kernel_distance` in "clean" mode, the same path as
`cleanfid.fid.compute_fid`, which `evaluate.py` uses.

No other packages are used beyond the course's `requirements.txt` (torch, torchvision, kagglehub,
Pillow, clean-fid, tqdm) and what they install (numpy, scipy, requests). EDM2's `training_loop.py`
imports `psutil` at module level for its status line; if psutil is missing, `teacher/official.py`
registers an empty stand-in, since `EDM2Loss` and `learning_rate_schedule` never call it. EDM2's
`generate_images.py` imports `click`, so only its `edm_sampler` and `StackedRandomGenerator`
definitions are executed from the file.
