"""Training entry point for team 16. Each stage runs its own script; arguments after the stage name are passed through.

    python train.py list                         # print the exact recipe of the submitted checkpoints
    python train.py <stage> [script arguments]   # run one stage (cwd and PYTHONPATH set as the script expects)

Stages (details and selection rules in TRAINING.md; all training uses train_split.txt only):
    teacher     E112 EDM2 teacher, from scratch            teacher/train.py               (python -m teacher.train)
    one         one_nfe: D2O one-step generator             training/d2o_one/train_d2o.py
    one-calib   one_nfe: output gain calibration + scoring  training/d2o_one/calibrate_generator.py
    few-traj    few_nfe 1/4: teacher ODE trajectories        training/ectmodel/gen_traj.py
    few-cmt     few_nfe 2/4: Consistency Mid-Training        training/ectmodel/cmt_train.py
    few-ecd     few_nfe 3/4: consistency distillation (ECD)  training/ectmodel/ect_train.py --distill 1
    few-sid     few_nfe 4/4: guided SiD over the 4-call chain, fake score = generator + LoRA   training/ectmodel/sid_few.py

The teacher is used for training only. Checkpoint paths in the recipe are relative to each stage's working directory.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ECT = os.path.join(ROOT, "training", "ectmodel")
D2O = os.path.join(ROOT, "training", "d2o_one")

STAGES = {
    "teacher": (ROOT, ["-m", "teacher.train"]),
    "one": (D2O, ["train_d2o.py"]),
    "one-calib": (D2O, ["calibrate_generator.py"]),
    "few-traj": (ECT, ["gen_traj.py"]),
    "few-cmt": (ECT, ["cmt_train.py"]),
    "few-ecd": (ECT, ["ect_train.py", "--distill", "1"]),
    "few-sid": (ECT, ["sid_few.py"]),
}

RECIPE = """\
python train.py teacher --prep-dir data/teacher_prep --run-dir runs/main_E112 --preset E112 --subset main \\
    --resume runs/pilot_E112_lr1x/training-state-XXXXXXXXXX.pt --allow-data-switch --total-steps <N>

python train.py one --outdir runs/e112_treal_lr3_cont --resume runs/e112_treal_lr3/d2o-state-latest.pt \\
    --g_edm2 checkpoints/edm/ema-0020971520-0.100.pt --freeze 0 --freeze_encoder 1 --d_shared 1 --d_width 0.5625 \\
    --d_drop_heads 1 --xflip 1 --ada_target 0.6 --lr 3e-4 --dlr 3e-5 --batch_gpu 16 --dmd_ratio 1.0 \\
    --dmd_teacher checkpoints/edm/teacher_unguided-0058720256-0.130.pt --dmd_fake_lora 6 --dmd_fake_lora_base generator \\
    --dmd_fake_lora_dec 1 --dmd_fake_lr 2e-4 --dmd_from_kimg 2000 --dmd_fake_steps 5 --dmd_guidance 1.0 \\
    --extra_real data/cache/teacher58_real --extra_real_p 1.0 --fid_reference results/reference_teacher58 \\
    --snap_kimg 25 --keep_snapshots 0 --kimg 5875

python train.py few-traj --out_dir traj_e112 --shards 12
python train.py few-cmt --run_dir runs/cmt_A --traj_dir traj_e112 --total_kimg 3200 --snap_kimg 400       
python train.py few-ecd --run_dir runs/ecd_cmt8 --init_ckpt runs/cmt_A/ema200_000800kimg.ckpt \\
    --start_stage 3 --stage_kimg 1000000 --lr 1e-4 --lr_ref_steps 0 --dropout -1 --total_kimg 6400         
python train.py few-ecd --run_dir runs/ecd_low --init_ckpt runs/ecd_cmt8/ema500_000400kimg.ckpt \\
    --start_stage 3 --stage_kimg 1000000 --lr 2e-5 --lr_ref_steps 0 --dropout -1 --total_kimg 1600 \\
    --snap_kimg 100 --ema_kimgs 200 500                                                                    
python train.py few-sid --run_dir runs/gbase12_lr1e4 --init runs/ecd_low/ema200_000400kimg.ckpt \\
    --teacher <E112 teacher_ag_main.pt> --cfg 1.5 --sigmas 80,3,1.5,0.4 --glr 3e-6 --lr 1e-4 --batch 64 \\
    --micro_batch 8 --fake_only_kimg 20 --ema_kimg 20 --snap_kimg 5 --total_kimg 80 --save_state 0 \\
    --fake_lora 12 --fake_lora_base generator
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if sys.argv[1] == "list":
        print(RECIPE)
        return
    if sys.argv[1] not in STAGES:
        sys.exit(f"unknown stage {sys.argv[1]!r}; one of: list, {', '.join(STAGES)}")
    cwd, cmd = STAGES[sys.argv[1]]
    env = dict(os.environ)
    env.setdefault("EDM_TEACHER_CODE", ROOT)  
    env["PYTHONPATH"] = os.pathsep.join(p for p in (ROOT, env.get("PYTHONPATH", "")) if p)  
    sys.exit(subprocess.call([sys.executable, *cmd, *sys.argv[2:]], cwd=cwd, env=env))


if __name__ == "__main__":
    main()
