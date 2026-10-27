"""Shared model configuration.

``ModelOneNFE`` is constructed by ``evaluate.py`` without arguments, so the
architecture of the submitted checkpoint has to be fixed here.
"""

NUM_CLASSES = 151
IMG_RESOLUTION = 64

# Official D2O / EDM DDPM++ 64x64 layout (channel_mult 1,2,2,2, four blocks per
# resolution, attention at 16x16) with model_channels reduced from 128 to 96:
# 34.8M parameters, so the generator plus its EMA copy stay below 100M.
GENERATOR_KWARGS = dict(
    img_resolution=IMG_RESOLUTION,
    img_channels=3,
    label_dim=NUM_CLASSES,
    arch="ddpmpp",
    model_channels=96,
    channel_mult=[1, 2, 2, 2],
    num_blocks=4,
    attn_resolutions=[16],
    dropout=0.0,
)

# Noise level fed to the one-step generator, x = G(sigma * z, sigma, c).
# 80 is the official D2O value.
SIGMA_G = 80.0

# Noise levels of the few-step generator, one network evaluation each: it starts
# from pure noise like the one-step generator, then re-noises its own output to
# the next level and denoises again. The intermediate levels are an untuned
# first choice inside the range the EDM was pre-trained on (ln sigma ~ N(-1.2, 1.2)).
MAX_FEW_NFE = 4
SIGMAS_FEW = [80.0, 2.0, 0.8, 0.3]
