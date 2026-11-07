NUM_CLASSES = 151
IMG_RESOLUTION = 64

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

GENERATOR_FAMILY = "edm"

GENERATOR_EDM2_KWARGS = dict(
    img_resolution=IMG_RESOLUTION,
    img_channels=3,
    label_dim=NUM_CLASSES,
    null_class=True,
    sigma_data=0.634817864072234,
    model_channels=112,
    channel_mult=[1, 2, 3, 4],
    num_blocks=3,
    attn_resolutions=[16, 8],
    channels_per_head=56,
    augment_dim=9,
    augment_balance=0.2,
)

SIGMA_G = 80.0

MAX_FEW_NFE = 4
SIGMAS_FEW = [80.0, 2.0, 0.8, 0.3]
