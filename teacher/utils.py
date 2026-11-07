import datetime
import json
import os
import re
import socket
import sys
import time

import numpy as np
import torch

KI = 1 << 10
MI = 1 << 20

def parse_nimg(value):
    if isinstance(value, (int, float)):
        return int(value)
    m = re.fullmatch(r'\s*([0-9]*\.?[0-9]+)\s*(Ki|Mi|Gi|k|M|G)?\s*', str(value))
    if not m:
        raise ValueError(f'Cannot parse image count {value!r}')
    scale = dict(Ki=KI, Mi=MI, Gi=1 << 30, k=10 ** 3, M=10 ** 6, G=10 ** 9).get(m.group(2), 1)
    return int(round(float(m.group(1)) * scale))

def format_nimg(n):
    if n < MI:
        return f'{n / KI:.1f} Ki ({int(n):,} img)'
    return f'{n / MI:.3f} Mi ({n / 1e6:.3f} Mimg)'

def crossed(prev, cur, interval):
    return interval > 0 and cur // interval > prev // interval

def to_builtin(obj):
    if isinstance(obj, dict):
        return {to_builtin(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_builtin(v) for v in obj)
    if obj is None or isinstance(obj, (torch.Tensor, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj

def code_fingerprint():
    import glob
    import hashlib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, 'teacher', '**', '*.py'), recursive=True))
    files.append(os.path.join(root, 'third_party', 'MANIFEST.json'))
    h = hashlib.sha256()
    for f in files:
        if os.path.isfile(f):
            h.update(os.path.relpath(f, root).encode())
            with open(f, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()

def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec='seconds')

class Logger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, 'log.txt')

    def print(self, *args):
        msg = ' '.join(str(a) for a in args)
        print(msg, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')

    def record(self, name, obj):
        with open(os.path.join(self.run_dir, f'{name}.jsonl'), 'a') as f:
            f.write(json.dumps(to_builtin(obj)) + '\n')

def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(to_builtin(obj), f, indent=2)
    os.replace(tmp, path)

def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    print(f'Warning: skipping an incomplete record in {path}', flush=True)
    return out

def device_info(device):
    info = dict(host=socket.gethostname(), torch=str(torch.__version__), python=sys.version.split()[0], device=str(device))
    if torch.device(device).type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, sm_count=p.multi_processor_count, total_memory=p.total_memory,
                    cuda=str(torch.version.cuda), cudnn=torch.backends.cudnn.version())
    return info

class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def __call__(self):
        return time.perf_counter() - self.t
