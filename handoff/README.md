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

## 저장소에 없는 것

- 데이터셋: `python download_dataset.py`로 받습니다.
- FID용 샘플 폴더(`runs/*/fid_samples/`, `results/*/`)와 train 특징 파일 두 개: 점수 스크립트가 다시 만듭니다.
- 다른 줄의 checkpoint와 재개 상태 (원래 서버에 있음, 필요하면 요청).

## 규정 메모

- 모델 선택은 train 쪽 기준으로만 했고 val(`evaluate.py`)은 고른 뒤 보고용으로만 돌렸습니다. 이어서 작업할 때도 같은 규칙을 지켜야 합니다 (어기면 0점 조항).
- 현재 줄의 학습 구성은 생성기 + 판별자 + fake score = 99,992,471 파라미터입니다 (EMA 복사본은 따로 세지 않는 팀 기준, 2026-10-13 TA 답변 반영). 자세한 내용은 기술 문서 6절.
