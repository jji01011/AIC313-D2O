import copy
import math

IMG_RESOLUTION = 64
IMG_CHANNELS = 3
NUM_CLASSES = 151
AUGMENT_DIM = 9
BATCH_SIZE = 256


NETWORK_PRESETS = {
    'E112': dict(family='edm2', ref_lr=0.0045, params=95_162_869,
                 unet=dict(model_channels=112, channel_mult=[1, 2, 3, 4], num_blocks=3, attn_resolutions=[16, 8], channels_per_head=56)),
    'E128-1233': dict(family='edm2', ref_lr=0.0042, params=92_836_261,
                 unet=dict(model_channels=128, channel_mult=[1, 2, 3, 3], num_blocks=3, attn_resolutions=[16, 8], channels_per_head=64)),
    'E144-1223': dict(family='edm2', ref_lr=0.0040, params=92_665_029,
                 unet=dict(model_channels=144, channel_mult=[1, 2, 2, 3], num_blocks=3, attn_resolutions=[16, 8], channels_per_head=72)),
    'E96': dict(family='edm2', ref_lr=0.0048, params=69_925_317,
                 unet=dict(model_channels=96, channel_mult=[1, 2, 3, 4], num_blocks=3, attn_resolutions=[16, 8], channels_per_head=48)),
    'E80': dict(family='edm2', ref_lr=0.0052, params=48_568_725,
                 unet=dict(model_channels=80, channel_mult=[1, 2, 3, 4], num_blocks=3, attn_resolutions=[16, 8], channels_per_head=40)),
    'E64': dict(family='edm2', ref_lr=0.0058, params=31_093_093,
                 unet=dict(model_channels=64, channel_mult=[1, 2, 3, 4], num_blocks=3, attn_resolutions=[16, 8], channels_per_head=64)),
    'DDPMpp128': dict(family='edm', ref_lr=2e-4, params=None,
                 unet=dict(model_channels=128, channel_mult=[1, 2, 2, 2], num_blocks=4, attn_resolutions=[16],
                           embedding_type='positional', encoder_type='standard', decoder_type='standard',
                           channel_mult_noise=1, resample_filter=[1, 1])),
    'tiny-edm2': dict(family='edm2', ref_lr=0.01, params=None,
                 unet=dict(model_channels=16, channel_mult=[1, 2, 2], num_blocks=1, attn_resolutions=[16], channels_per_head=16)),
    'tiny-ddpmpp': dict(family='edm', ref_lr=2e-4, params=None,
                 unet=dict(model_channels=16, channel_mult=[1, 2, 2], num_blocks=1, attn_resolutions=[16],
                           embedding_type='positional', encoder_type='standard', decoder_type='standard',
                           channel_mult_noise=1, resample_filter=[1, 1])),
}


RECIPES = {
    'edm2': dict(
        loss='edm2',
        P_mean_base=-0.8, P_std=1.6,
        betas=(0.9, 0.99), eps=1e-8,
        ref_batches=35_000,
        rampup_steps=5_000,
        nan_to_num=dict(nan=0.0, posinf=0.0, neginf=0.0),
        precision='fp16',
    ),
    'edm': dict(
        loss='edm',
        P_mean_base=-1.2, P_std=1.2,
        betas=(0.9, 0.999), eps=1e-8,
        ref_batches=0,
        rampup_steps=5_000,
        nan_to_num=dict(nan=0.0, posinf=1e5, neginf=-1e5),
        precision='bf16',
    ),
}

COMMON_RECIPE = dict(
    batch_size=BATCH_SIZE,
    micro_batch=64,
    label_dropout=0.1,
    augment=dict(p=0.15, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1),
    ema_stds=[0.050, 0.100],
    stratified_sigma=True,
)


def dropout_for_planned_images(planned_images):
    mimg = planned_images / 1e6
    if mimg < 60:
        return 0.10
    if mimg < 120:
        return 0.15
    return 0.20

def size_rule(r_e112):
    if r_e112 >= 230:
        return dict(teacher='E112', arms=['E112@1x', 'E112@2x', 'E128-1233@1x', 'E144-1223@1x', 'DDPMpp128@1x'])
    if r_e112 >= 150:
        return dict(teacher='E112', arms=['E112@1x', 'E112@2x', 'DDPMpp128@1x'])
    if r_e112 >= 100:
        return dict(teacher='E96', arms=['E96@1x', 'E96@2x', 'DDPMpp128@1x'])
    arms = ['E80@1x', 'E80@2x', 'DDPMpp128@1x']
    note = 'below 60 img/s the design also suggests Patch Diffusion on E96 (not implemented)' if r_e112 < 60 else None
    return dict(teacher='E80', arms=arms, note=note)

def steps_for_days(img_per_sec, days, batch_size=BATCH_SIZE):
    return int(img_per_sec * days * 86400 // batch_size)


def p_mean_for(family, sigma_data):
    return RECIPES[family]['P_mean_base'] + math.log(sigma_data / 0.5)

def network_config(preset, dropout=0.0, precision=None, sigma_data=0.5):
    p = NETWORK_PRESETS[preset]
    family = p['family']
    kwargs = dict(img_resolution=IMG_RESOLUTION, img_channels=IMG_CHANNELS, label_dim=NUM_CLASSES, sigma_data=float(sigma_data))
    kwargs.update(copy.deepcopy(p['unet']))
    if family == 'edm2':
        kwargs.update(augment_dim=AUGMENT_DIM, augment_balance=0.2, logvar_channels=128, null_class=True,
                      use_fp16=True, precision=precision or RECIPES['edm2']['precision'], forced_wn='post_step',
                      dropout_res=dropout_table(family, dropout, p['unet']['channel_mult']))
    else:
        kwargs.update(augment_dim=AUGMENT_DIM, null_class=True, use_fp16=True,
                      precision=precision or RECIPES['edm']['precision'], dropout=float(dropout))
    return dict(family=family, preset=preset, kwargs=kwargs)

def dropout_table(family, p, channel_mult):
    if family != 'edm2':
        return float(p)
    resolutions = [IMG_RESOLUTION >> level for level in range(len(channel_mult))]
    return {int(res): (0.0 if res == IMG_RESOLUTION else float(p)) for res in resolutions}

def recipe(preset, ref_lr=None):
    family = NETWORK_PRESETS[preset]['family']
    r = copy.deepcopy(COMMON_RECIPE)
    r.update(copy.deepcopy(RECIPES[family]))
    r['family'] = family
    r['ref_lr'] = float(ref_lr if ref_lr is not None else NETWORK_PRESETS[preset]['ref_lr'])
    return r
