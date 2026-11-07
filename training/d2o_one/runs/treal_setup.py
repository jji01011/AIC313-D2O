import fcntl
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from cleanfid import fid
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, ".")
from d2o.fid import folder_statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR, REF = "data/cache/teacher58_ref", "results/reference_teacher58"
REAL_DIR = "data/cache/teacher58_real"
TREAL = dict(
    run="e112_treal",
    args=("--g_edm2 checkpoints/edm/ema-0020971520-0.100.pt --freeze_encoder 1 --d_shared 1 --d_width 0.5625 --d_drop_heads 1 "
          "--lr 1e-3 --batch_gpu 16 --dmd_ratio 1.0 --dmd_teacher checkpoints/edm/teacher_unguided-0058720256-0.130.pt "
          "--dmd_fake_lora 6 --dmd_fake_lora_base generator --dmd_fake_lora_dec 1 --dmd_fake_lr 2e-4 --dmd_from_kimg 2000 "
          "--dmd_fake_steps 5 --dmd_guidance 1.0 --extra_real data/cache/teacher58_real --extra_real_p 1.0 --dlr 1e-4 "
          "--fid_reference results/reference_teacher58"),
    piece=50, until=5425, max_until=6125, log="results/tref_calibration.log", score="! bash runs/tref_score.sh",
    init="runs/e112_lorag_f5_g13_lr3b/d2o-state-latest.pt", start=5125)


def log(text):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {text}"
    print(line, flush=True)
    with open("results/treal_setup.log", "a") as file:
        file.write(line + "\n")


def export(images, folder):
    os.makedirs(folder, exist_ok=True)
    for old in glob.glob(os.path.join(folder, "*.png")):
        os.remove(old)
    for count, image in enumerate(images):
        to_pil_image(image).convert("RGB").save(os.path.join(folder, f"{count:05d}.png"))


def main():
    os.chdir(ROOT)
    ref_shards, real_shards = sorted(glob.glob(f"{REF_DIR}/shard-*.pt")), sorted(glob.glob(f"{REAL_DIR}/shard-*.pt"))
    if len(ref_shards) < 5 or len(real_shards) < 17:
        log(f"not complete ({len(ref_shards)} reference shards, {len(real_shards)} real shards): nothing done")
        return 1
    device = torch.device("cuda")
    feat_model = fid.build_feature_extractor("clean", device, use_dataparallel=False)
    if not os.path.exists(f"{REF}_stats.npz"):
        images = torch.cat([torch.load(path, weights_only=False)["images"] for path in ref_shards])
        export(images, REF)
        mu, sigma = folder_statistics(REF, feat_model, device)
        np.savez(f"{REF}_stats.npz", mu=mu, sigma=sigma)
        log(f"{REF}: {len(images)} teacher samples exported, statistics written")
    stats = np.load(f"{REF}_stats.npz")
    for shard in real_shards[:3]:
        images = torch.load(shard, weights_only=False)["images"]
        export(images, "results/treal_floor_samples")
        mu, sigma = folder_statistics("results/treal_floor_samples", feat_model, device)
        log(f"floor: {shard} (3,020 teacher samples, 20 per category) against {REF}: {fid.frechet_distance(mu, sigma, stats['mu'], stats['sigma']):.3f}")

    state = os.environ.get("TREAL_DIRECTOR_STATE", "runs/director_state.json")
    return switch(state) if os.path.exists(state) else 0


def switch(state_path):
    with open(state_path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(state_path) as file:
            state = json.load(file)
        current = state.get("manual")
        if current and current.get("run") == TREAL["run"]:
            log("the director already has the e112_treal entry")
        else:
            if current:
                state.setdefault("manual_after", []).insert(0, dict(current, piece=100))
            state["manual"] = TREAL
            tmp = state_path + ".tmp"
            with open(tmp, "w") as file:
                json.dump(state, file, indent=1)
            os.replace(tmp, state_path)
            log(f"director ({state_path}): manual entry is now e112_treal (from {TREAL['init']} at {TREAL['start']} kimg to "
                f"{TREAL['until']}); after it: {[entry['run'] for entry in state.get('manual_after', [])]}")
    if state_path == "runs/director_state.json":
        with open("results/treal_setup.done", "w") as file:
            file.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
