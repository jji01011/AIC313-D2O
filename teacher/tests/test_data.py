import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from teacher import data as data_mod
from teacher import official

SPLIT_DIR = os.path.join(official.PROJECT_ROOT, 'data', 'pokemon-generation-one-22k')

def _tiny_dataset(root, n_per_class=3, classes=('Abra', 'Pikachu')):
    lines = []
    rng = np.random.RandomState(0)
    for c in classes:
        os.makedirs(os.path.join(root, 'PokemonData', c), exist_ok=True)
        for i in range(n_per_class):
            ext = 'png' if i % 2 else 'jpg'
            arr = rng.randint(0, 256, size=(rng.randint(40, 130), rng.randint(40, 130), 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(root, 'PokemonData', c, f'{c}{i}.{ext}'))
            lines.append(f'PokemonData/{c}/{c}{i}.{ext}')
    with open(os.path.join(root, 'train_split.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(root, 'val_split.txt'), 'w') as f:
        f.write('\n'.join(lines[:1]) + '\n')
    with open(os.path.join(root, 'category_to_id.json'), 'w') as f:
        json.dump({c: i for i, c in enumerate(classes)}, f)
    open(os.path.join(root, '.transparency_composited_on_white'), 'w').close()

def test_cache_is_bit_exact():
    with tempfile.TemporaryDirectory() as root:
        _tiny_dataset(root)
        cache = data_mod.build_cache(root, root, num_workers=0, verbose=False)
        ds, _ = data_mod.load_course_dataset(root, root)
        for i in range(len(ds)):
            x, y = ds[i]
            assert torch.equal(data_mod.u8_to_float(cache['images'][i]), x)
            assert int(cache['labels'][i]) == int(y)

def test_png_export_matches_evaluate():
    import evaluate
    images = torch.rand(5, 3, 64, 64) * 2 - 1
    images[0, :, :4] = 1.0
    images[1, :, :4] = -1.0

    class FakeModule:
        def val_dataloader(self):
            return [images]

    with tempfile.TemporaryDirectory() as d:
        evaluate.prepare_reference_set(FakeModule(), os.path.join(d, 'a'))
        os.makedirs(os.path.join(d, 'b'))
        for i, img in enumerate(images):
            data_mod.export_png(img, os.path.join(d, 'b', f'{i:04d}.png'))
        for i in range(len(images)):
            a = open(os.path.join(d, 'a', f'{i:04d}.png'), 'rb').read()
            b = open(os.path.join(d, 'b', f'{i:04d}.png'), 'rb').read()
            assert a == b

def test_splits_structure():
    with open(os.path.join(SPLIT_DIR, 'train_split.txt')) as f:
        paths = [l.strip() for l in f if l.strip()]
    with open(os.path.join(SPLIT_DIR, 'category_to_id.json')) as f:
        mapping = json.load(f)
    labels = np.array([mapping[p.split('/')[1]] for p in paths])
    s = data_mod.draw_splits(labels, len(mapping), seed=0)
    C, H, P, M, probe = map(set, (s['canary'], s['holdout'], s['pilot_train'], s['main_train'], s['probe']))
    assert len(C) == 755 and len(H) == 3020 and len(P) == 14059 and len(M) == 16324
    assert C <= H and not (P & H) and not (M & C) and P <= M and probe <= P
    for k in range(151):
        assert sum(labels[i] == k for i in s['canary']) == 5
        assert sum(labels[i] == k for i in s['holdout']) == 20
    for a, b in s['floor_pairs']:
        assert not set(a) & set(b) and set(a) <= M and set(b) <= M and len(a) == len(b) == 3020
    for r in s['ref_subsets']:
        assert set(r) <= M and len(r) == 3020
    assert data_mod.draw_splits(labels, len(mapping), seed=0) == s

def test_stream_matches_infinite_sampler():
    n = 37
    images = torch.arange(n, dtype=torch.uint8).reshape(n, 1, 1, 1).expand(n, 3, 2, 2).contiguous()
    labels = torch.arange(n)
    stream = data_mod.TrainStream(images, labels, seed=5, device=torch.device('cpu'))
    it = iter(official.InfiniteSampler(range(n), seed=5))
    expect = [next(it) for _ in range(200)]
    got = []
    for pos in range(0, 200, 16):
        _, y = stream.get(pos, min(16, 200 - pos))
        got += y.tolist()
    assert got == expect
    _, y = stream.get(123, 10)
    assert y.tolist() == expect[123:133]

def test_mean_and_sigma_data():
    torch.manual_seed(0)
    u8 = torch.randint(0, 256, (50, 3, 8, 8), dtype=torch.uint8)
    mean, sd = data_mod.mean_and_sigma_data(u8)
    x = data_mod.u8_to_float(u8).double()
    m = x.mean(0)
    m = (m + m.flip(-1)) / 2
    assert torch.allclose(mean.double(), m, atol=1e-6)
    assert abs(sd - float((x - m).pow(2).mean().sqrt())) < 1e-9
