import argparse
import time
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid

from d2o.fid import FIDMonitor, folder_statistics
from evaluate import generate_samples
from model import ModelOneNFE


def main(args):
    device = torch.device(args.device)
    monitor = FIDMonitor(args.data_root, args.split_dir, reference_dir=args.refs[0], device=device)
    refs = [np.load(Path(ref).with_name(Path(ref).name + "_stats.npz")) for ref in args.refs]
    for path in args.ckpts:
        source = torch.load(path, map_location="cpu", weights_only=False)
        model = ModelOneNFE(device=device, sigma_g=source["args"]["sigma_g"], **source["net_kwargs"]).to(device)
        model.load_state_dict(source["state_dict"])
        model.eval().requires_grad_(False)
        scores = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            workdir = Path(args.workdir)
            for old in workdir.glob("*.png"):
                old.unlink()
            generate_samples(model, workdir, monitor.categories, batch_size=64, device=device)
            mu, sigma = folder_statistics(workdir, monitor.feat_model, device)
            scores.append([float(fid.frechet_distance(mu, sigma, ref["mu"], ref["sigma"])) for ref in refs])
        scores = np.array(scores)
        parts = [f"{Path(ref).name} {scores[:, k].mean():.3f} ({', '.join(f'{v:.3f}' for v in scores[:, k])})" for k, ref in enumerate(args.refs)]
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {path} ({source['cur_nimg'] / 1e3:.0f} kimg), seeds {args.seeds}: " + " | ".join(parts)
        print(line, flush=True)
        with open(args.log, "a") as log:
            log.write(line + "\n")
        del model, source
        torch.cuda.empty_cache()


if __name__ == "__main__":
    split = lambda text: [value for value in text.split(",") if value]
    parser = argparse.ArgumentParser(description="One-step checkpoints against several train reference statistics")
    parser.add_argument("--ckpts", type=split, required=True)
    parser.add_argument("--refs", type=split, default=["./results/reference_train", "./results/reference_train_balanced"])
    parser.add_argument("--seeds", type=lambda text: [int(v) for v in text.split(",")], default=[0, 1, 2])
    parser.add_argument("--workdir", default="./results/score_refs_samples")
    parser.add_argument("--log", default="./results/balanced_fid.log")
    parser.add_argument("--data_root", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--split_dir", default="./data/pokemon-generation-one-22k")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
