import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Stand-in for ./data/pokemon-generation-one-22k')
    p.add_argument('--split-dir', default='./data/pokemon-generation-one-22k')
    p.add_argument('--limit-per-class', type=int, default=None, help='Write a reduced train_split.txt with at most N images per class')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    root = Path(args.root)
    split_dir = Path(args.split_dir)
    lines = [l.strip() for l in (split_dir / 'train_split.txt').read_text().splitlines() if l.strip()]
    mapping = __import__('json').loads((split_dir / 'category_to_id.json').read_text())
    if args.limit_per_class:
        kept, seen = [], {}
        for line in lines:
            c = Path(line).parent.name
            if seen.get(c, 0) < args.limit_per_class:
                kept.append(line)
                seen[c] = seen.get(c, 0) + 1
        lines = kept
    root.mkdir(parents=True, exist_ok=True)
    (root / 'train_split.txt').write_text('\n'.join(lines) + '\n')
    shutil.copy(split_dir / 'category_to_id.json', root / 'category_to_id.json')
    shutil.copy(split_dir / 'val_split.txt', root / 'val_split.txt')
    rng = np.random.RandomState(args.seed)
    palette = np.random.RandomState(1234).randint(0, 230, size=(len(mapping), 3))
    for i, line in enumerate(lines):
        path = root / line
        path.parent.mkdir(parents=True, exist_ok=True)
        c = mapping[path.parent.name]
        w, h = rng.randint(48, 160), rng.randint(48, 160)
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
        rx, ry = w * rng.uniform(0.15, 0.35), h * rng.uniform(0.15, 0.35)
        col = tuple(int(v) for v in np.clip(palette[c] + rng.randint(-20, 21, 3), 0, 255))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)
        fmt = 'PNG' if path.suffix.lower() == '.png' else 'JPEG'
        img.save(path, format=fmt)
    (root / '.transparency_composited_on_white').touch()
    print(f'wrote {len(lines)} images under {root}')

if __name__ == '__main__':
    main()
