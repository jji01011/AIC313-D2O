# One-NFE (D2O) 인계

2026-10-27 기준 one-NFE 작업 전체입니다. 코드와 로그는 이 저장소에, checkpoint는 Release `handoff-2026-10-27`에 있습니다.

- **기술 설명**: [docs/FASTGEN_TECHNICAL_HANDOFF.md](../docs/FASTGEN_TECHNICAL_HANDOFF.md) (구조, 원본 논문과의 차이, 100M 계산, 재현 명령)
- **실험 목록**: [docs/FASTGEN_EXPERIMENT_INVENTORY.csv](../docs/FASTGEN_EXPERIMENT_INVENTORY.csv)
- **작업 일지**: [STATUS.md](../STATUS.md) (결정과 이유, 시간순. 맨 위가 최신)

## checkpoint 받기

```bash
GH_TOKEN=<이 저장소를 읽을 수 있는 토큰> bash handoff/fetch_checkpoints.sh
```

파일마다 원래 경로에 받고 md5를 확인합니다. 목록과 md5는 [checkpoints.txt](checkpoints.txt)에 있습니다. 브라우저로 Release 페이지에서 받아도 되는데, 그때는 파일 이름의 `__`를 `/`로 바꾼 경로에 두면 됩니다 (`edm__*`는 `checkpoints/edm/` 아래).

| 파일 | 크기 | 용도 |
|---|---|---|
| `runs/e112_treal_lr3/calib-005675.ckpt` | 381 MB | 공식 FID 최저 **21.36** (3회 평균, 보고용으로만 측정) |
| `runs/e112_treal_lr3/calib-005825.ckpt` | 381 MB | 미리 정한 규칙(teacher 기준 최저 10.460)으로 고른 후보, 공식 21.74 |
| `runs/e112_treal_lr3/d2o-state-latest.pt` | 1.33 GB | 위 줄을 5825 kimg에서 이어 학습할 상태 (생성기, EMA, 판별자, fake score, optimizer) |
| `checkpoints/edm/teacher_unguided-0058720256-0.130.pt` | 381 MB | teacher T58 (DM 항의 real score, 판별자용 표본의 출처) |
| `checkpoints/edm/ema-0058720256-0.100.pt` | 190 MB | T58과 같은 학습의 EMA 0.10 파일 |
| `checkpoints/edm/ema-0020971520-0.100.pt` | 190 MB | E112 출발점 (`--g_edm2`, `--dmd_teacher` 인자에 쓰임) |
| `checkpoints/edm/ema-0016777216-0.100.pt` | 190 MB | 처음 받은 E112 |
| `data/cache/teacher58_real.tar` | 631 MB | 판별자의 "진짜" 이미지로 쓰는 T58 표본 51,340장. 스크립트가 `data/cache/teacher58_real/`에 풉니다 |

두 후보의 공식 FID 차이(0.38)는 측정 흔들림(±0.3~0.5) 안입니다. **`checkpoints/one_nfe.ckpt`는 아직 만들지 않았습니다** (어느 후보를 낼지 미정).
checkpoint는 출력 gain 1.10을 가중치에 접어 넣은 것이라 `ModelOneNFE`로 그대로 로드해서 평가하면 됩니다.

## 바로 실행하기

환경은 `requirements.txt` 그대로입니다 (torch, torchvision, clean-fid, kagglehub, Pillow, tqdm). clean-fid 0.1.35는 SciPy 1.18 이상에서 깨지므로 `scipy<1.18`.
모든 명령은 저장소 루트에서 실행합니다.

```bash
python download_dataset.py                       # 데이터셋 (처음 한 번)
GH_TOKEN=<토큰> bash handoff/fetch_checkpoints.sh   # checkpoint 8개
```

**후보 평가 (공식 evaluate.py, val 기준이라 보고용으로만)**. `one_nfe.ckpt`를 만들지 않고 경로를 직접 넘깁니다.

```bash
python evaluate.py --model_checkpoint runs/e112_treal_lr3/calib-005675.ckpt --evaluate_mode one_nfe
```

생성 이미지는 `results/evaluation/`에 남습니다. 처음 실행할 때 val 기준 이미지를 `results/reference/`에 만듭니다. 측정마다 ±0.3~0.5 흔들립니다.

**이어서 학습 (5825 kimg 상태에서)**. 두 후보가 있는 `runs/e112_treal_lr3/`를 건드리지 않도록 새 폴더에 씁니다. 50 kimg에 A100 20 GB 조각 기준 약 75분, peak VRAM 12.8 GB입니다.

```bash
python train_d2o.py --outdir runs/e112_treal_lr3_cont --resume runs/e112_treal_lr3/d2o-state-latest.pt \
    --g_edm2 checkpoints/edm/ema-0020971520-0.100.pt --freeze 0 --freeze_encoder 1 --d_shared 1 --d_width 0.5625 --d_drop_heads 1 \
    --xflip 1 --ada_target 0.6 --lr 3e-4 --dlr 3e-5 --batch_gpu 16 --dmd_ratio 1.0 --dmd_teacher checkpoints/edm/teacher_unguided-0058720256-0.130.pt \
    --dmd_fake_lora 6 --dmd_fake_lora_base generator --dmd_fake_lora_dec 1 --dmd_fake_lr 2e-4 --dmd_from_kimg 2000 \
    --dmd_fake_steps 5 --dmd_guidance 1.0 --extra_real data/cache/teacher58_real --extra_real_p 1.0 \
    --fid_reference results/reference_teacher58 --snap_kimg 25 --keep_snapshots 0 --kimg 5875
```

인자는 `e112_treal_lr3`를 돌린 줄(`runs/director.log` 1064행)에 `runs/chain_final.sh`가 덧붙이던 공통 인자(`--freeze 0 --xflip 1 --ada_target 0.6 --snap_kimg 25 --keep_snapshots 0`)를 더한 것입니다. chain은 `--edm runs/edm_c96/edm-005000.pt`도 붙였지만 `--g_edm2`가 있으면 읽지 않으므로 뺐습니다 (`train_d2o.py:137`).
`--extra_real`은 상태에 저장되지 않아서 이어 갈 때마다 다시 넣어야 합니다. 같은 `--outdir`로 더 가려면 `--resume runs/e112_treal_lr3_cont/d2o-state-latest.pt`와 늘린 `--kimg`를 줍니다.

**고르기 (teacher 기준 점수, 이 줄의 선택 규칙)**. gain 1.0/1.1/1.2 × 시드 3개로 보정해 `calib-<kimg>.ckpt`를 쓰고 `results/tref_calibration.log`에 적습니다.

```bash
PY=python bash runs/tref_score.sh e112_treal_lr3_cont 5875
```

주의: 이 스크립트는 같은 run 폴더에서 그 run의 최저가 아닌 calib 파일을 지웁니다 (공식 최저 파일은 남김). 줄 전체의 최저가 되면 `evaluate.py`를 세 번 돌려 `results/official_fid.log`에 적습니다 (보고용).

**확인한 것 (2026-10-27, 이 저장소를 새로 clone하고 Release에서 받은 파일로)**: 8개 파일 md5 일치.
`evaluate.py`로 calib-005675 공식 21.25 (기록 21.26~21.45). `train_d2o.py` 위 명령으로 5825 → 5827 재개 (99,992,471 파라미터, nonfinite 0, peak 12.7 GB).
`tref_score.sh`로 상태에서 5825를 다시 보정하면 teacher 기준 10.460이 그대로 나오고, 만들어진 `calib-005825.ckpt`가 Release 파일과 md5까지 같습니다.

`runs/chain_final.sh`, `runs/director.py`, `runs/watchdog.sh`, `runs/status.sh`는 원래 서버의 cron과 경로에 묶인 자동화라 다른 기계에서는 그대로 돌지 않습니다.

## 저장소에 없는 것

- 데이터셋: `python download_dataset.py`로 받습니다.
- FID용 샘플 폴더(`runs/*/fid_samples/`, `results/*/`)와 train 특징 파일 두 개: 점수 스크립트가 다시 만듭니다.
- 다른 줄의 checkpoint와 재개 상태 (원래 서버에 있음, 필요하면 요청).

## 규정 메모

- 모델 선택은 train 쪽 기준으로만 했고 val(`evaluate.py`)은 고른 뒤 보고용으로만 돌렸습니다. 이어서 작업할 때도 같은 규칙을 지켜야 합니다 (어기면 0점 조항).
- 현재 줄의 학습 구성은 생성기 + 판별자 + fake score = 99,992,471 파라미터입니다 (EMA 복사본은 따로 세지 않는 팀 기준, 2026-10-13 TA 답변 반영). 자세한 내용은 기술 문서 6절.
