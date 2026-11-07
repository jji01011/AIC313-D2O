import ast
import importlib.util
import os
import sys
import types

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.path.join(PROJECT_ROOT, 'third_party')
EDM2_ROOT = os.path.join(THIRD_PARTY, 'edm2')
EDM_ROOT = os.path.join(THIRD_PARTY, 'edm')

if EDM2_ROOT not in sys.path:
    sys.path.insert(0, EDM2_ROOT)

import dnnlib
from torch_utils import misc as edm2_misc
from torch_utils import persistence


def _load_file(module_name, path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_definitions(path, names, namespace):
    with open(path) as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise ImportError(f'{sorted(missing)} not found in {path}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return [namespace[name] for name in names]

try:
    import psutil
except ImportError:
    sys.modules['psutil'] = types.ModuleType('psutil')

networks_edm2 = _load_file('edm2_training_networks_edm2', os.path.join(EDM2_ROOT, 'training', 'networks_edm2.py'))
phema = _load_file('edm2_training_phema', os.path.join(EDM2_ROOT, 'training', 'phema.py'))
edm2_training_loop = _load_file('edm2_training_training_loop', os.path.join(EDM2_ROOT, 'training', 'training_loop.py'))

edm_augment = _load_file('edm_training_augment', os.path.join(EDM_ROOT, 'training', 'augment.py'))
edm_networks = _load_file('edm_training_networks', os.path.join(EDM_ROOT, 'training', 'networks.py'))
edm_loss = _load_file('edm_training_loss', os.path.join(EDM_ROOT, 'training', 'loss.py'))

MPFourier = networks_edm2.MPFourier
MPConv = networks_edm2.MPConv
Block = networks_edm2.Block
normalize = networks_edm2.normalize
mp_silu = networks_edm2.mp_silu
mp_sum = networks_edm2.mp_sum
mp_cat = networks_edm2.mp_cat
PowerFunctionEMA = phema.PowerFunctionEMA
solve_posthoc_coefficients = phema.solve_posthoc_coefficients
EDM2Loss = edm2_training_loop.EDM2Loss
learning_rate_schedule = edm2_training_loop.learning_rate_schedule
set_random_seed = edm2_misc.set_random_seed
AugmentPipe = edm_augment.AugmentPipe
SongUNet = edm_networks.SongUNet
EDMLoss = edm_loss.EDMLoss
InfiniteSampler = edm2_misc.InfiniteSampler

edm_sampler, StackedRandomGenerator = _load_definitions(
    os.path.join(EDM2_ROOT, 'generate_images.py'), ['edm_sampler', 'StackedRandomGenerator'], dict(np=np, torch=torch))
