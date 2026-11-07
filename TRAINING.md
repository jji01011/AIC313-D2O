# 현재 최고 제출본 학습 기록

공식 `evaluate.py`(validation split, 3,020장) 기준 결과입니다.

| 모드 | 모델 | 파라미터 | NFE | FID | TA 기준 |
|---|---|---|---|---|---|
| `one_nfe` | 팀원 D2O one-step 생성기 (E112 EDM2 denoiser를 GAN 목적함수로 미세조정, `calib-005825`) + 저주파 층화 latent (고정 seed) | 95,158,709 | 1 | **21.36** | 39.15 |
| `few_nfe` | E112 구조 consistency 모델 (CMT → ECD → guided SiD, fake score = 학생 + LoRA, 4-step) + 층화 noise (고정 seed) | 95,175,157 | 4 | **21.32** | 30.90 |

- 측정: 아래 명령(`--batch_size 32`, 기본값)으로 쟀습니다. `one_nfe`는 2026-11-03에 21.364였고, 2026-11-06에 이 폴더에서 다시 재도 21.364였습니다. `few_nfe`는 2026-11-07에 21.323입니다(1회 측정, 보고용). 두 모델 모두 noise를 샘플링 순서로 고정해서 batch 크기와 상관없이 같은 noise를 씁니다(`one_nfe`는 `--batch_size 16`에서 21.346으로, GPU 연산 차이 정도만 다릅니다).
- 이전 `few_nfe` 제출본과 후보(MeanFlow 26.56, SiD 20.62, ECD + 층화 22.22)는 2-b절, 2-c절, 2-d절에 남겨 두었습니다.
- 이전 `one_nfe` 제출본들(R3GAN `ho_v3mf_ann` 23.53, R3GAN v1 28.53)은 0-a절과 1-b절에 남겨 두었습니다. 직전 제출 폴더 전체는 `../submission_backup_20261029/`에 있습니다.

```bash
cd team_16
python evaluate.py --model_checkpoint ./checkpoints/one_nfe.ckpt --evaluate_mode one_nfe
python evaluate.py --model_checkpoint ./checkpoints/few_nfe.ckpt --evaluate_mode few_nfe
```

두 모드는 서로 다른 모델입니다. 모든 모델은 `train_split.txt`만으로 처음부터 학습했고, 사전학습 모델은 쓰지 않았습니다.
- `one_nfe`는 팀원이 train split만으로 처음부터 학습한 EDM2 E112 denoiser에서 시작한 D2O 생성기입니다. 학습에는 같은 방식으로 학습한 teacher T58(95.2M, 고정)을 썼습니다.
- `few_nfe`는 팀원이 train split만으로 처음부터 학습한 EDM2 teacher(E112, 95.2M)를 증류한 모델입니다.
- 대회 규칙상 teacher는 학습에만 쓸 수 있고 100M 이하여야 합니다. 모든 teacher가 이를 지키며, 샘플링할 때는 제출 모델만 실행합니다.

---

## 0. 현재 `one_nfe` (2026-11-03): 팀원 D2O 생성기 + 저주파 층화 latent (val FID 21.36)

**출처**: 팀원 저장소 `jji01011/AIC313-D2O`(branch `d2o-handoff`, commit `32f0704`)의 인계본입니다. 가중치는 Release `handoff-2026-10-27`의 `e112_treal_lr3__calib-005825.ckpt`(md5 `ec1ca78a06754c685e76db510f74d919`)를 그대로 `checkpoints/one_nfe.ckpt`로 썼습니다.
학습 방법, 하이퍼파라미터, 재현 명령은 그 저장소의 `handoff/README.md`와 `docs/FASTGEN_TECHNICAL_HANDOFF.md`에 있습니다. 요약하면 다음과 같습니다.
- **생성기**: 팀원 E112 EDM2 denoiser(95.2M, train split으로 처음부터 학습)를 노이즈 수준 80에서 한 번 부르는 one-step 생성기로 미세조정했습니다(D2O, Zheng & Yang, ICML 2025).
- **목적함수**: D2O의 GAN loss(판별자가 생성기의 고정 encoder를 특징 추출기로 공유)와 DMD2식 distribution matching 항입니다. real score는 teacher T58(고정), fake score는 생성기에 붙인 LoRA입니다.
- **판별자의 "진짜" 입력**: teacher T58의 표본 51,340장입니다. T58은 train split만으로 처음부터 학습했습니다.
- **체크포인트 선택(팀원 규칙, 미리 정함)**: teacher 표본 기준 FID 최저인 5,825 kimg(10.460). 출력 gain 1.10은 가중치에 접어 넣었습니다.
- **학습 최대 VRAM**: 12.8 GB. 학습 framework(생성기, 판별자 고유 부분, LoRA)의 parameter는 99,992,471입니다.

**latent 설계와 noise seed (train split만 사용)**: evaluate.py는 seed를 고정하지 않아 같은 가중치도 val FID가 측정마다 흔들립니다(독립 latent로 21.49~21.82, 3회 평균 21.69). 그래서 이미지마다 쓰는 latent를 고정하고, 클래스마다 20장의 latent 저주파 성분을 층화했습니다.
- **저주파 층화**: 각 latent의 8×8 블록 평균(×8 하면 3×8×8 = 192개 값, 각각 N(0, 1))을 채널마다 정규직교 8×8 DCT로 돌린 192개 계수를, 한 클래스의 20장에 대해 Latin hypercube로 다시 뽑습니다. 계수마다 확률 1/20 구간에서 하나씩 뽑고, 구간을 어느 이미지에 줄지는 무작위입니다. 각 계수는 여전히 N(0, 1)이고, 고주파 나머지는 seed 표 그대로입니다. 1 NFE 그대로이며 학습은 없습니다.
- **효과 (같은 seed 0~7끼리 독립 latent와 비교)**: held-out FID −0.18, E112 teacher 표본 10k 기준 FID −0.19이고, 8개 seed 모두에서 둘 다 낮아졌습니다. val(보고용)은 −0.22입니다. 짝 이미지끼리 저주파를 공유하게 하면 held-out이 +0.75 나빠지므로, 저주파 성분이 이미지 구성을 정합니다. 층화 해상도(4×4, 16×16, 64×64), 구간 중앙값 LHS, Sobol 점은 차이가 없었고 (z, −z) 짝은 효과가 절반 이하였습니다.
- **seed 선택**: seed 0~7 중 held-out FID가 가장 낮은 seed를 골랐습니다. val은 고르는 데 쓰지 않았습니다.

| seed | held-out FID (선택 기준) | teacher 표본 FID | val FID (보고용) |
|---|---|---|---|
| 0 | 19.74 | 11.03 | 21.33 |
| 1 | 19.76 | 10.94 | 21.59 |
| 2 | 19.76 | 10.99 | 21.63 |
| 3 | 19.68 | 10.91 | 21.95 |
| **4 (선택)** | **19.60** | **10.91** | **21.36** |
| 5 | 19.66 | 11.03 | 21.82 |
| 6 | 19.73 | 10.92 | 21.74 |
| 7 | 19.73 | 11.10 | 21.54 |
| 평균 | 19.71 | 10.98 | 21.62 |

- 이전 제출본(층화 없음, seed 0~3 중 seed 1)은 held-out 19.71, teacher 표본 11.04, val 21.28이었습니다. 층화로 seed 사이의 held-out 편차가 0.116에서 0.056으로 줄었습니다.
- 같은 목적함수로 5,825 kimg에서 더 학습하는 것(lr 1e-4, 3e-4, DM 항 2배, 각 50~75 kimg)은 세 경우 모두 EMA의 held-out과 teacher 표본 FID를 조금씩 나쁘게 해서 쓰지 않았습니다.

**샘플링 (`ModelOneNFE`, 정확히 1 NFE)**: `x = G(80 z, 80, c)`. 샘플링 순서상 j번째 이미지의 seed 표 noise는 CPU `torch.Generator`를 `4 * 10**7 + j`로 seed해서 뽑고, 연속한 20장(evaluate.py 순서로 한 클래스)마다 저주파 계수를 `4 * 10**7 + 5 * 10**6 + 묶음 번호`로 seed한 난수로 층화합니다.

**few-step으로 쓰려는 시도(기록용)**: 생성 → 노이즈 다시 섞기 → 다시 생성(노이즈 수준 2.5~0.3, 2~4 NFE)은 FID 70~110으로 실패했습니다. 생성기가 노이즈 수준 80에서만 미세조정됐기 때문입니다.

## 0-a. 이전 `one_nfe` (2026-10-24 ~ 10-28 제출본, val FID 23.53): `ho_v3mf` + 학습률 감쇠 마무리

1절의 `ho_v3mf` 1,400만 장 학습 상태에서 이어서 학습했습니다(`runs/ho_v3mf_ann`).
- **바뀐 점**: 200만 장 동안 학습률을 5e-5에서 0으로 선형으로 줄였습니다(`--anneal_start_kimg 14000 --anneal_end_kimg 16000`).
- **나머지**: 데이터, 합성 데이터, 손실, 증강, EMA는 그대로입니다.
- **held-out FID (선택 기준)**: 1,400만 장 25.76 → 1,500만 장 25.18 → 1,575만 장 24.72 → **1,600만 장 24.44**
  - E112 실행(`ho_v3e112`, 24.58)보다도 낮습니다. held-out은 E112 쪽에 유리하게 기울어 있는데도 이겼으므로, 제출본을 다시 이 모델로 바꿨습니다.
- **보고용 val FID**: 1,500만 장 23.04, 1,600만 장 23.84입니다. 제출 폴더에서 공식 `evaluate.py`로 다시 재면 23.53입니다.

```bash
mkdir -p runs/ho_v3mf_ann && cp runs/ho_v3mf/train_state_014000kimg.pt runs/ho_v3mf_ann/train_state.pt
python train.py <1절의 ho_v3mf 인자> --total_kimg 16000 --anneal_start_kimg 14000 --anneal_end_kimg 16000 \
    --fine_from_kimg 14000 --run_dir runs/ho_v3mf_ann
# 제출 가중치 = runs/ho_v3mf_ann/G_ema_016000kimg.ckpt
```

## 0-b. E112 teacher 실험 (2026-10-17 ~ 10-23, 한때 제출본, val FID 24.71)

1절과 같은 GAN 설정입니다. 합성 데이터만 MeanFlow 대신 팀원의 EDM2 teacher로 만들었습니다.
- **teacher**: E112. 파라미터 95,162,869(학습 대상) + 12,288(고정)입니다.
  - train split만으로 처음부터 학습했습니다(98시간, 5,870만 장, 학습 최대 VRAM 19.0 GB).
  - 학습에만 쓰고 샘플링에는 쓰지 않습니다.
- **합성 데이터**: autoguidance(w = 1.3, Heun 32-step)로 71,575장을 만들었습니다(`ganmodel/runs/teacher_data/gen_edm2.py`).
  - 세 teacher 버전 중 held-out FID가 가장 낮은 것을 골랐습니다: unguided 18.21, CFG 1.4 17.63, autoguidance 1.3 17.47.
- **학습과 선택**: held-in 데이터로 1,400만 장까지 학습했습니다(`runs/ho_v3e112`, 기록 `runs/fid_ho_v3e112.tsv`).
  - held-out FID 최저는 1,375만 장의 24.58로, MeanFlow 데이터로 학습한 `ho_v3mf`(25.76)보다 낮습니다.
  - 미리 정한 규칙(held-out이 0.5 이상 낮으면 교체)에 따라 제출본을 바꿨습니다.
- **주의**: 보고용 val FID는 24.71로, `ho_v3mf`(23.28)보다 높습니다.
  - 두 teacher 모두 held-out 이미지까지 학습했고, E112 쪽이 학습 이미지를 더 잘 재현합니다(암기 점수 0.213 대 0.224).
  - 그래서 held-out 비교가 E112 쪽에 유리하게 기울었을 수 있습니다.
- **진행 중**: 같은 설정을 전체 train과 E112 합성 데이터 85,395장으로 학습하고 있습니다(`runs/full_v3e112`, 1,400만 장에서 멈추도록 미리 정함).

## 1. `one_nfe`: R3GAN + MeanFlow 합성 데이터 (2026-10-12 ~ 10-19 제출본, val FID 23.28)

### 무엇이 바뀌었나 (이전 v1 대비)
1. **teacher 합성 데이터**: MeanFlow 4-step 샘플 71,575장(학습 이미지 1장당 5장, 클래스 비율 유지)을 미리 만들었습니다(`ganmodel/runs/teacher_data/gen_meanflow.py`).
   - GAN의 "진짜" 배치 중 50%를 이 합성 이미지에서 뽑습니다(`--extra_data`, `--extra_ratio 0.5`).
   - 데이터가 적어 판별기가 과적합하던 문제가 사라졌습니다. v1은 900만 장 이후 무너졌지만, 이번 실행은 1,400만 장까지 계속 좋아졌습니다.
2. **증강과 정규화를 공식 R3GAN 기본값으로 맞춤**
   - 색 증강 확률을 기하 증강의 0.5배로 줄였습니다(`--aug_color_mult 0.5`).
   - γ는 2.0 → 0.2로 바꿨습니다(FFHQ-64 기본값).
3. **선택 기준을 바꿈**
   - train 안에는 중복 이미지가 많아(18%가 정확한 사본을 가짐) train FID가 믿을 만하지 않았습니다.
   - 그래서 중복 묶음 단위로 train에서 떼어 낸 held-out 세트(`ganmodel/runs/heldout/`)로 고릅니다.
     - 2,764장을 학습에서 빼고, 서로 다른 이미지 2,265장을 FID 기준으로 씁니다.
     - 진짜 이미지의 held-out FID 하한은 18.1입니다.
   - validation은 보고에만 썼습니다.
4. **속도**: `torch.compile`을 R1/R2가 없는 스텝에만 적용했습니다. 초당 126장이 142장이 됐습니다.

생성기와 판별기 구조, 손실 함수, R1/R2, lazy regularization, bf16은 1-b절(v1)과 같습니다.

### 하이퍼파라미터
| 항목 | 값 |
|---|---|
| 학습 데이터 | held-in 14,315장 + MeanFlow 합성 71,575장 (배치의 50%) |
| 배치 / Optimizer | 32 / Adam β1 = 0, β2 0.9 → 0.99 |
| 학습률 | 2e-4 → 5e-5 |
| γ (R1, R2) | 2.0 → 0.2, lazy ×16 |
| ADA p | 0 → 0.5 (색 증강은 ×0.5) |
| 생성기 EMA | 반감기 0 → 50만 장 |
| 스케줄 | 모두 1,200만 장에 걸친 코사인, 그 뒤로는 최종값 유지 |
| seed | 1 |
| 최대 VRAM | 16.83 GB (`torch.cuda.max_memory_allocated`) |
| 속도 | 초당 약 140장 (A100D-20C vGPU 1장) |

### 재현 명령 (`ganmodel/`)
```bash
cd meanflowmodel   # 1) teacher 합성 데이터 (MeanFlow = meanflowmodel/runs/meanflow_final/ema_0040000.ckpt, 이전 few_nfe)
python ../ganmodel/runs/teacher_data/gen_meanflow.py 5 ../ganmodel/runs/teacher_data/mf4_x5.pt
cd ../ganmodel     # 2) held-out 분할 (중복 묶음 단위, train split만 사용)
python runs/heldout/make_heldout.py 0.1 15
python train.py --batch_size 32 --compile 1 --exclude runs/heldout/heldout_split.pt \
    --total_kimg 14000 --decay_kimg 12000 --gamma_base 2.0 --gamma_final 0.2 \
    --aug_final 0.5 --aug_color full --aug_color_mult 0.5 --ema_kimg 500 --reg_interval 16 \
    --extra_data runs/teacher_data/mf4_x5.pt --extra_ratio 0.5 \
    --log_kimg 20 --sample_kimg 500 --snapshot_kimg 1000 --fine_snapshot_kimg 250 --fine_from_kimg 5000 \
    --keep_states 1 --keep_states_from 5000 --seed 1 --run_dir runs/ho_v3mf
# 같은 명령에 --total_kimg 20000 --keep_states 0 을 주면 이어서 학습합니다(진행 중).
# 제출 가중치 = runs/ho_v3mf/G_ema_014000kimg.ckpt (held-out FID 최저 25.76)  →  submission/checkpoints/one_nfe.ckpt
```

### 스냅샷별 FID (기록: `ganmodel/runs/fid_ho_v3mf.tsv`)
| 학습량 | 100만 | 200만 | 300만 | 400만 | 500만 | 600만 | 700만 | 800만 | 900만 | 1,000만 | 1,100만 | 1,200만 | 1,300만 | **1,400만** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| held-out FID (선택 기준) | 44.75 | 36.37 | 34.00 | 33.65 | 32.33 | 31.66 | 31.28 | 30.40 | 29.23 | 28.21 | 27.28 | 26.48 | 25.87 | **25.76** |
| validation FID (보고용) | 47.54 | 37.04 | 32.85 | 30.89 | 29.21 | 27.84 | 27.24 | 25.88 | 24.75 | 24.27 | 23.56 | 23.34 | 23.44 | **23.34** |
| 암기 점수 (높을수록 덜 베낌) | 0.214 | 0.216 | 0.220 | 0.223 | 0.223 | 0.224 | 0.223 | 0.225 | 0.224 | 0.224 | 0.226 | 0.226 | 0.224 | 0.224 |

**비교 실험** (모두 같은 held-in 데이터와 held-out 기준)
- 합성 데이터 없이 같은 설정으로 학습한 `ho_v3`는 held-out FID가 100만/200만/300만 장에서 51.73/43.96/40.23이었습니다.
- 같은 학습량에서 합성 데이터를 쓴 쪽이 6~8 낮습니다.

**이전 제출본과의 비교**
- 이전 제출본(v1 900만 장)의 held-out FID는 29.06이었습니다.
- v1은 held-out 이미지까지 학습한 모델이라 유리한 조건이었는데도 이 결과라서, 제출본을 교체했습니다.

### 샘플링
v1과 같습니다. `z ~ N(0, I)`(64차원)와 클래스 라벨로 생성기를 1회 실행하고, 결과를 [-1, 1]로 자릅니다. 생성기 구조와 설정이 같아서 `model.py`는 바뀌지 않았습니다.

---

## 1-b. 이전 `one_nfe` 제출본: R3GAN v1 (2026-10-10 ~ 10-12, val FID 28.53)

### 방법
R3GAN(Huang et al., NeurIPS 2024)을 공식 코드(github.com/brownvc/R3GAN) 구조 그대로 순수 PyTorch로 다시 구현했습니다. 공식 코드의 NVIDIA 전용 CUDA 연산(`bias_act`, `upfirdn2d`)은 공식 코드에 함께 들어 있는 참조 구현으로 대체했습니다.

- **생성기**: 64차원 노이즈와 64차원 클래스 임베딩을 이어 붙여 입력으로 받습니다.
  - 4×4 basis에서 시작해 [1,2,1] 필터로 보간 업샘플링하며 64×64까지 키웁니다.
  - 각 해상도에서 ResNet 1-3-1 bottleneck 블록 2개를 쓰고, 3×3 합성곱은 grouped 방식입니다.
  - 채널 수: (768, 768, 768, 768, 384)
  - Cardinality: (96, 96, 96, 96, 48)
  - Expansion: 2
  - 정규화 층은 없고, fix-up 초기화를 씁니다.
- **판별기**: 생성기를 거꾸로 뒤집은 구조입니다(23,086,464 파라미터). 마지막 특징을 클래스 임베딩(768차원)에 투영하는 projection 방식으로 클래스 조건을 넣습니다.
- **손실**: 상대적 GAN 손실을 씁니다.
  - 판별기: `softplus(-(D(real) - D(fake)))`
  - 생성기: `softplus(-(D(fake) - D(real)))`
  - 진짜와 가짜 쌍은 같은 클래스 라벨을 씁니다.
- **정규화**: 판별기에 zero-centered R1(진짜)과 R2(가짜) gradient penalty를 겁니다. 가중치는 γ/2이고, 증강된 입력에 대해 계산합니다.
- **ADA 증강**(Karras et al., 2020): 판별기 입력(진짜와 가짜 모두)에 미분 가능하게 적용합니다.
  - 기하: xflip, rotate90, 정수 이동, 확대·축소, 회전, 비등방 확대, 소수 이동
  - 색: 밝기, 대비, 휘도 반전, 색상 회전, 채도
  - cutout
  - 데이터셋 자체에도 좌우 반전을 적용합니다.

### 공식 설정과 다른 점 (추가 라이브러리 없이 돌리기 위한 것)
- **bf16 autocast**: 공식 코드도 고해상도 단계를 fp16으로 돌립니다.
- **Lazy regularization**: R1·R2를 **16스텝마다** 계산하고 가중치를 ×16으로 키웁니다(StyleGAN2 방식).
  - 이유: 전용 CUDA 연산 없이는 grouped 합성곱의 2차 미분이 일반 스텝보다 약 10배 느립니다.
  - 효과: 학습 속도가 초당 36장에서 127장이 됐습니다.

### 하이퍼파라미터
코사인 스케줄은 공식 `cosine_decay_with_warmup`과 같고, 1,200만 장에 걸쳐 진행됩니다.

| 항목 | 값 |
|---|---|
| 배치 | 32 (공식 FFHQ-64 설정의 GPU당 배치와 같음) |
| Optimizer | Adam, β1 = 0, β2: 0.9 → 0.99 (코사인) |
| 학습률 | 2e-4 → 5e-5 (코사인) |
| γ (R1, R2) | 1.0 → 0.1 (코사인), lazy ×16 |
| ADA 확률 p | 0 → 0.5 (코사인) |
| 생성기 EMA | 반감기를 0에서 50만 장까지 늘림 (코사인) |
| 900만 장 시점의 값 | lr 7.2e-5, γ 0.232, β2 0.977, p 0.427, EMA 반감기 42.7만 장 |
| 학습량 | 900만 장 시점 스냅샷을 선택 (1,400만 장까지 학습) |
| 시간 | 900만 장까지 19.8시간 (A100D-20C vGPU 1장, 초당 약 126장) |
| 최대 VRAM | 16.14 GB (`torch.cuda.max_memory_allocated`) |

### 재현 명령 (`ganmodel/`)
```bash
cd ganmodel
python train.py --batch_size 32 --total_kimg 14000 --decay_kimg 12000 \
    --gamma_base 1.0 --gamma_final 0.1 --aug_final 0.5 --ema_kimg 500 \
    --reg_interval 16 --log_kimg 20 --sample_kimg 500 --snapshot_kimg 1000 \
    --run_dir runs/r3gan
# 제출 가중치 = runs/r3gan/G_ema_009000kimg.ckpt  →  submission/checkpoints/one_nfe.ckpt
```
나중에 추가한 옵션의 기본값(`--aug_color full`, `--keep_states 0`)은 이 실행과 같은 동작입니다. 실제 실행 인자는 `runs/r3gan/train_state.pt`의 `args`에 저장돼 있습니다.

### 스냅샷별 FID와 선택 이유
| 학습량 | 100만 | 200만 | 300만 | 400만 | 500만 | 600만 | 700만 | 800만 | **900만** | 1,000만 | 1,100만 | 1,200만 | 1,300만 | 1,400만 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train 기준 FID | 46.48 | 40.14 | 35.89 | 33.20 | 31.85 | 30.79 | 29.57 | 28.64 | **27.23** | 28.09 | 32.41 | 42.20 | 53.84 | 68.51 |
| validation FID (보고용) | 52.00 | 44.07 | 37.54 | 34.43 | 33.59 | 31.67 | 29.89 | 29.46 | **28.18** | 30.74 | 38.10 | 48.42 | 62.24 | 77.23 |

- **선택 기준**: 900만 장 스냅샷을 **train split 기준 FID**(클래스당 20장, `meanflowmodel/runs/train_reference`)가 가장 낮다는 이유로 골랐습니다. validation은 선택에 쓰지 않았습니다.
- **900만 장 이후가 무너진 이유**: 증강 확률 p가 0.4를 넘고 γ가 0.25 아래로 내려간 구간입니다.
  - 색 증강(색상 회전, 채도, 휘도 반전)이 생성 결과로 새어 나왔습니다. 샘플에 무늬 배경과 과한 채도가 나타났습니다.
  - 판별기가 생성기를 압도했습니다. 마지막 로그의 점수는 진짜 3.54, 가짜 0.33이었습니다.
- **재학습(v2) 결과: 채택하지 않음**
  - 바꾼 설정: 색 증강은 밝기·대비만 남김(`--aug_color mild`), p 최대 0.3, γ 최종 0.4, 1,000만 장에서 감쇠 종료(`--total_kimg 12000 --decay_kimg 10000`).
  - 1,200만 장까지 무너지지 않았지만, train FID 최저가 1,100만 장의 27.47로 v1 900만 장의 27.23보다 높았습니다. 그래서 v1을 유지합니다.
  - v2 val FID(보고용)는 900만 장 이후 30.66 ~ 34.34였습니다.
  - 암기 점수(`ganmodel/memo.py`, 높을수록 덜 베낌)는 v1 900만 장 0.2078, v2 1,100만 장 0.2108로 둘 다 암기 문제는 없었습니다.
  - 기록: `ganmodel/runs/fid_r3gan_v2.tsv`, `ganmodel/runs/memo_v1_v2.tsv`

### 샘플링 (`submission/model.py`의 `ModelOneNFE`)
`z ~ N(0, I)` (64차원)와 클래스 라벨로 생성기를 1회 실행하고, 결과를 [-1, 1]로 자릅니다. 정확히 1 NFE입니다.

---

## 2. 현재 `few_nfe` (2026-11-07): ECD 모델 + guided SiD (fake score = 학생 + LoRA) + 층화 noise (val FID 21.32)

2-d절 모델(2-a절 `ecd_low` EMA200 400k)에서 시작해, 4-step 샘플러 전체를 생성기로 놓고 guided Score identity Distillation(SiD)으로 7만 장 더 학습했습니다. 실제 이미지는 쓰지 않고 E112 teacher의 score만 씁니다.
- **왜**: ECD는 한 번의 호출이 teacher 궤적의 끝을 맞히도록 학습하므로, 4번 이어 부르는 샘플러 출력의 분포는 직접 학습되지 않습니다. SiD는 그 최종 출력 분포를 teacher 분포에 맞춥니다.
- **손실**: 생성 결과 y에 노이즈를 섞은 x_s = y + s·n에서, teacher score f_φ(CFG 1.5, `--cfg 1.5`)와 fake score f_ψ의 차이로 생성기를 학습합니다(SiD, α = 1). gradient는 4번의 호출 모두를 지나갑니다(`--grad_steps 4`).
- **fake score와 100M 제한**: SiD 원 논문처럼 teacher 전체 복사본을 fake score로 쓰면 학습 framework가 약 190M이 됩니다. 그래서 fake score를 **학생 생성기 자신 + LoRA(rank 12, 3,606,420개, `--fake_lora 12 --fake_lora_base generator`)** 로 만들었습니다. 팀원 D2O의 fake score(생성기 + LoRA)와 같은 구성입니다.
  - fake score는 학생과 같은 parameter 텐서를 씁니다(복사본 없음, LoRA 가지는 fake score 통과 때만 켜짐).
  - fake score를 통과할 때는 학생 가중치를 상수로 둡니다(`requires_grad` 끔, eval 모드). 그래서 fake score 손실은 LoRA만 학습하고, 생성기 손실은 SiD 그대로 fake score의 입력을 통해서만 학생에 전달됩니다. 시작 시점의 fake score는 학생과 출력이 같습니다.
  - 공유 텐서는 한 번만 센다는 TA 답변(2026-10-13)에 따라 학습 framework는 학생 95,175,157 + LoRA 3,606,420 = **98,781,577개**입니다. teacher는 학습되지 않는 별도 모델입니다.
- **base 비교**: 같은 설정에서 LoRA를 고정된 teacher 위에 붙인 실행도 했습니다(2026-11-06, 직전 제출본, val 21.31). 학생 base와 teacher base는 모든 스냅샷에서 held-out 차이가 0.03 이하였고(아래 표), 전체 복사본 실행(18.58)과도 같은 수준입니다. teacher 위 LoRA는 teacher 가중치가 학생 framework에 들어가는지 TA 답변으로 정해지지 않아, 규정이 분명한 학생 base를 제출했습니다.
- **하이퍼파라미터**: 생성기 lr 3e-6, fake score lr 1e-4, Adam(β 0, 0.999), batch 64(micro-batch 8), 처음 2만 장은 fake score만 학습, EMA 반감기 2만 장, 학습 σ (80, 3, 1.5, 0.4). 최대 VRAM 9.9 GB, RTX 3090 1장에서 7만 장까지 2.1 h(예열 포함).
- **선택 (train split의 held-out FID만 사용, 미리 정한 규칙)**: fake score lr 1e-4와 3e-5 두 run을 8만 장까지 학습하고, 3만 5천 장부터 5천 장마다 4-seed held-out 평균을 쟀습니다. 가장 낮은 것은 lr 1e-4의 7만 장(18.562)이었습니다. lr 3e-5의 최저도 7만 장(18.565)으로 차이는 잡음 수준입니다. 두 run 모두 7만 장 뒤에는 다시 나빠졌습니다.

| fake score, lr (held-out, seed 0~3 평균) | 35k | 45k | 55k | 60k | 65k | **70k** | 75k | 80k |
|---|---|---|---|---|---|---|---|---|
| **학생 + LoRA, 1e-4 (제출)** | 18.760 | 18.694 | 18.616 | 18.584 | 18.572 | **18.562** | 18.581 | 18.606 |
| 학생 + LoRA, 3e-5 | 18.760 | 18.698 | 18.615 | 18.585 | 18.570 | 18.565 | 18.577 | 18.609 |
| teacher + LoRA, 1e-4 (직전 제출본) | 18.759 | 18.691 | 18.619 | 18.586 | 18.575 | 18.573 | 18.591 | 18.630 |
| teacher + LoRA, 3e-5 | 18.756 | 18.697 | 18.620 | 18.587 | 18.579 | 18.581 | 18.602 | 18.637 |

- **샘플러**: σ (80, 5, 1.8, 0.4)와 층화 noise(2-d절과 같은 방법)입니다. σ는 같은 SiD 설정의 전체 복사본 모델에서 held-out으로 고른 값입니다.
- **seed 선택**: seed 0~7 중 held-out FID가 가장 낮은 seed 3입니다. val은 seed를 고른 뒤 보고용으로만 쟀습니다.

| seed | held-out FID (선택 기준) | teacher 표본 FID | val FID (보고용) |
|---|---|---|---|
| 0 | 18.569 | 9.556 | 21.68 |
| 1 | 18.577 | 9.455 | 21.54 |
| 2 | 18.722 | 9.601 | 21.27 |
| **3 (선택)** | **18.382** | **9.530** | **21.32** |
| 4 | 18.696 | 9.557 | 21.07 |
| 5 | 18.531 | 9.549 | 21.54 |
| 6 | 18.734 | 9.539 | 21.65 |
| 7 | 18.494 | 9.480 | 21.09 |
| 평균 | 18.588 | 9.533 | 21.39 |

- 시작점인 2-d절 모델(같은 층화, σ 80, 3, 1.5, 0.4)은 held-out 18.76~19.07, val 21.51~22.40이었습니다.

**샘플링 (`ModelFewNFE`, 정확히 4 NFE)**: `x = f(mean + 80 z₀, 80) → f(x + 5 z₁, 5) → f(x + 1.8 z₂, 1.8) → f(x + 0.4 z₃, 0.4) → clamp(-1, 1)`.

**재현 명령**: `python train.py list`(2-a절의 궤적, CMT, ECD 두 단계 뒤에 아래 명령)입니다. 원본 스크립트는 `training/ectmodel/gbase_queue_1107.sh`(두 run), `gbase_post_1107.sh`(선택 절차)이고, 로그는 `training/ectmodel/logs/gbase12_lr*/`에 있습니다. teacher base 비교 실행은 `lora_queue_1106.sh`, `logs/lora12_lr*/`입니다.
```bash
python train.py few-sid --run_dir runs/gbase12_lr1e4 --init runs/ecd_low/ema200_000400kimg.ckpt \
    --teacher <E112 teacher_ag_main.pt> --cfg 1.5 --sigmas 80,3,1.5,0.4 --glr 3e-6 --lr 1e-4 --batch 64 \
    --micro_batch 8 --fake_only_kimg 20 --ema_kimg 20 --snap_kimg 5 --total_kimg 80 --save_state 0 \
    --fake_lora 12 --fake_lora_base generator
# 제출 가중치 = runs/gbase12_lr1e4/ema500_000070kimg.ckpt (md5 8f62ec637c5ecac4c838eaecea6a7a55)  →  checkpoints/few_nfe.ckpt
```

## 2-d. 이전 `few_nfe` 후보 (2026-11-03): 2-a절 ECD 모델 + 층화 noise, σ (80, 3, 1.5, 0.4) (val FID 22.22)

2절 모델의 SiD 시작점입니다.

가중치는 2-a절의 `ecd_low` EMA200 400k 그대로입니다. 2-c절의 SiD 75k는 val FID를 보고 고른 것이고 train split 지표(held-out, teacher 표본)는 이 모델보다 나빠서 다시 이 모델로 돌아왔습니다.
샘플러만 바꿨고, 모두 train split의 held-out FID로 골랐습니다(seed 0~3 평균 비교, val은 보고용).
- **noise 층화**: 4개 noise 각각의 8×8 블록 평균 성분을 `one_nfe`와 같은 방법(0절)으로 클래스마다 20장에 걸쳐 층화합니다. k번째 noise는 seed `FEW_NOISE_SEED + 100 k`를 씁니다. 첫 noise만 층화했을 때 held-out −0.01, teacher 표본 FID −0.18이었고, 네 noise 모두 층화하면 held-out이 0.03 더 낮았습니다.
- **σ 스케줄**: (80, 4, 1.5, 0.5) 19.01 → (80, 3, 1.5, 0.4) 18.97 (teacher 표본 9.86 → 9.78). (80, 3, 1.2, 0.4), (80, 3, 1.0, 0.3)은 같은 수준이고, (80, 2, 0.8, 0.3), (80, 6, 2.5, 1.0)은 나빴습니다.
- **해 보았지만 쓰지 않은 것**: 다음 단계 noise에 이전 단계의 noise를 섞기(비율 0.3에서 held-out +0.4), 출력 gain 0.97~1.06(1.03이 −0.005로 차이 없음).
- **seed 선택**: seed 0~7 중 held-out FID가 가장 낮은 seed입니다.

| seed | held-out FID (선택 기준) | teacher 표본 FID | val FID (보고용) |
|---|---|---|---|
| 0 | 18.97 | 9.79 | 22.40 |
| 1 | 18.93 | 9.74 | 22.04 |
| 2 | 19.01 | 9.80 | 21.51 |
| 3 | 18.82 | 9.80 | 22.20 |
| 4 | 19.02 | 9.79 | 21.77 |
| **5 (선택)** | **18.76** | **9.76** | **22.22** |
| 6 | 19.07 | 9.72 | 22.26 |
| 7 | 18.86 | 9.73 | 21.83 |

- 2-a절 그대로(층화 없음, σ 80, 4, 1.5, 0.5, 고정하지 않은 noise)는 held-out 19.04~19.33, teacher 표본 9.84~9.92, val 22.15~22.57이었습니다.

**샘플링 (`ModelFewNFE`, 정확히 4 NFE)**: `x = f(mean + 80 z₀, 80) → f(x + 3 z₁, 3) → f(x + 1.5 z₂, 1.5) → f(x + 0.4 z₃, 0.4) → clamp(-1, 1)`. `z₀..z₃`는 위의 층화 noise입니다.

## 2-c. 이전 `few_nfe` (2026-10-29 ~ 11-02 제출본): 2-a절 모델 + SiD 후학습 (val FID 20.62)

2-a절 모델(`ecd_low` EMA200 400k)에서 시작해 Score identity Distillation(SiD, Zhou et al., ICML 2024)으로 7만 5천 장 더 학습했습니다(`runs/sid_B`).
- **방법**: 4-step 샘플러를 그대로 생성기로 씁니다. 손실은 고정된 E112 teacher score와, 생성 결과로 학습하는 fake score(초기값 teacher)의 차이입니다. 실제 이미지는 쓰지 않습니다.
- **코드와 하이퍼파라미터**: `training/ectmodel/sid_few.py`(원본 `ectmodel/sid_few.py`). 학습률은 생성기와 fake score 모두 3e-5, batch 64, fake score 단독 워밍업 2만 장, EMA 반감기 2만 장입니다. 최대 VRAM은 17.2 GB입니다.
- **선택 (train split 규칙을 따르지 않음)**: 75k 스냅샷은 팀 결정으로 val FID를 보고 골랐습니다.
  - train split 지표는 2-a절 모델보다 나빴습니다(held-out 19.10 → 20.06, teacher 표본 기준 FID 10.01 → 11.39).
  - 그런데 val은 25k마다 22.09, 21.19, 20.39, 20.42, 20.42, 20.68로 나아졌습니다.
- **샘플링**: 2-a절과 같습니다(`ModelFewNFE`, 노이즈 수준 80, 4.0, 1.5, 0.5, 정확히 4 NFE).

## 2-a. 이전 `few_nfe` (2026-10-27 ~ 10-28 제출본): E112 teacher 증류 consistency 모델 (val FID 22.57)

### 구조
팀원 teacher와 같은 EDM2 U-Net(E112) 구조입니다.
- 채널 112, 배수 (1,2,3,4), 해상도마다 블록 3개
- attention은 16×16과 8×8에서 씁니다.
- 가중치 정규화를 강제합니다(forced weight normalization).
- 파라미터는 95,162,869(학습 대상) + 12,288(고정 평균 이미지)입니다.

네트워크 코드는 팀원 코드(`teacher/networks.py`, `teacher/official.py`)와, 그것이 불러오는 NVlabs EDM2/EDM 공식 코드(`third_party/`, 수정 없음, CC BY-NC-SA 4.0, `THIRD_PARTY.md`)를 제출 폴더에 포함했습니다. 코스 의존성(numpy, torch)만 씁니다.

### 학습 (train split만 사용, teacher는 학습에만 사용)
1. **초기화**: E112 teacher(unguided, 5,870만 장 학습, val 20.50)의 가중치를 복사했습니다.
2. **teacher 궤적 생성** (`training/ectmodel/gen_traj.py`)
   - teacher의 공식 sampler(EDM2 Heun 32-step, 63 NFE)로 궤적 49,152개(4,096개 × 12 shard)를 만들었습니다. 제출한 CMT 80만 장 스냅샷은 이 49,152개로 학습했습니다(학습 로그 첫 줄 `trajectories 49152`). 80만 장에서 재시작할 때 shard 4개를 더해 65,536개로 이어 학습했지만, 그 구간(80만 장 이후)은 제출 모델에 쓰지 않았습니다. 공식 `edm_sampler`와 끝점 차이는 최대 0.004입니다.
   - 클래스는 train 라벨 분포에서 뽑았습니다.
   - 궤적마다 σ = 80 → 0 사이의 상태 17개를 저장했습니다.
3. **CMT 중간학습** (Consistency Mid-Training, Hu et al. 2025; `cmt_train.py`), 80만 장
   - 궤적 위의 상태 x_{t_i}에서 그 궤적의 끝 이미지를 바로 맞히도록 회귀합니다.
   - 거리: 논문은 LPIPS를 쓰지만 사전학습 모델을 쓸 수 없어서 pseudo-Huber(c = 0.06)를 썼습니다.
   - lr 7e-4 선형 감쇠, 배치 128, Adam(0.9, 0.99)
4. **ECD 후학습** (Consistency distillation, ECT의 ImageNet-64 설정; `ect_train.py --distill 1`)
   - 손실: x_t = y + t·ε(t ~ LogNormal(−0.8, 1.6))에서 teacher의 Euler 한 스텝으로 x_r을 만들고, f(x_t, t) ≈ sg[f(x_r, r)]이 되도록 학습합니다.
     - 간격은 r = t(1 − (1 + 8σ(−t))/256)이고, 손실은 pseudo-Huber × (t²+σd²)/(tσd)²입니다.
   - 먼저 lr 1e-4로 40만 장(CMT 80만 장 스냅샷의 EMA 반감기 20만에서 시작)을 학습했습니다. 이 단계는 40만 장 이후 다단계 품질이 나빠졌습니다.
   - 그래서 40만 장 스냅샷(EMA 반감기 50만)에서 lr 2e-5로 40만 장을 더 학습했습니다.
   - **제출 가중치는 이 두 번째 단계의 EMA(반감기 20만 장) 40만 장 스냅샷**입니다.
5. 학습 최대 VRAM은 13.6 GiB(`torch.cuda.max_memory_allocated`)이고, 모든 단계에 18~19.5 GiB 할당 상한을 걸었습니다.

### 선택 (train split의 held-out FID만 사용)
| 후보 (4-step) | held-out FID |
|---|---|
| 이전 제출본 MeanFlow | 26.78 |
| ECD 기본 학습 (CMT 없음, 240만 장) | 32.59 |
| CMT 80만 장 | 25.37 |
| CMT → ECD(lr 1e-4) 40만 장 | 20.82 |
| **CMT → ECD → ECD(lr 2e-5) 40만 장, EMA 20만 (제출)** | **19.10** |
| 같은 단계 50만 장 | 19.36 |

- σ 스케줄도 held-out으로 골랐습니다. 탐색한 후보는 (80, 2.0, 0.8, 0.3), (80, 4.0, 1.5, 0.5), (80, 5, 2.0, 0.7), (80, 6, 2.5, 1.0) 등입니다.
- 주의: teacher가 held-out 이미지까지 학습했기 때문에, E112에서 나온 모델은 held-out에서 실제보다 유리하게 나옵니다(teacher: held-out 18.21, val 20.50).
  - 그래도 보고용 val FID도 이전 제출본보다 낮아서(22.57 대 26.56), 두 지표가 같은 방향입니다.

### 샘플링 (`ModelFewNFE`, 정확히 4 NFE)
```
x = f(mean + 80·z, 80) → x = f(x + 4.0·z₁, 4.0) → x = f(x + 1.5·z₂, 1.5) → x = f(x + 0.5·z₃, 0.5) → clamp(-1, 1)
```

### 재현 명령 (`training/ectmodel/`, Vast 2× RTX 3090에서 실행)
```bash
python gen_traj.py --out_dir traj_e112 --shards 12                 # teacher 궤적 49,152개 (80만 장 스냅샷이 쓴 궤적)
python cmt_train.py --run_dir runs/cmt_A --traj_dir traj_e112 --total_kimg 3200 --snap_kimg 400   # 800k 스냅샷 사용
python ect_train.py --run_dir runs/ecd_cmt8 --distill 1 --init_ckpt runs/cmt_A/ema200_000800kimg.ckpt \
    --start_stage 3 --stage_kimg 1000000 --lr 1e-4 --lr_ref_steps 0 --dropout -1 --total_kimg 6400   # 400k 스냅샷 사용
python ect_train.py --run_dir runs/ecd_low --distill 1 --init_ckpt runs/ecd_cmt8/ema500_000400kimg.ckpt \
    --start_stage 3 --stage_kimg 1000000 --lr 2e-5 --lr_ref_steps 0 --dropout -1 --total_kimg 1600 \
    --snap_kimg 100 --ema_kimgs 200 500
# 제출 가중치 = runs/ecd_low/ema200_000400kimg.ckpt  →  submission/checkpoints/few_nfe.ckpt
```
- CMT run은 800k에서 궤적을 늘려 다시 시작한 뒤 품질이 나빠졌습니다. 그래서 800k 스냅샷만 썼습니다.

---

## 2-b. 이전 `few_nfe`: MeanFlow (4 steps, val FID 26.56; 현재는 `one_nfe` 합성 데이터의 teacher)

A/B 실험 기록 등 더 자세한 내용은 `meanflowmodel/TRAINING.md`(영문)와 `meanflowmodel/runs/` 아래 tsv 파일에 있습니다.

### 방법
MeanFlow(Geng et al., 2025)입니다. 평균 속도 u(z_t, r, t)를 학습하고, 타깃은 MeanFlow 항등식 `u = v − (t − r)·du/dt`로 계산합니다. du/dt는 forward-mode JVP로 구합니다.

- **백본**: ADM/DDPM++ 스타일 UNet입니다.
  - 채널 128, mult (1,2,3,3), 레벨당 블록 3개
  - 어텐션은 16×16과 8×8에서 씁니다.
  - AdaGN으로 (t, h = t − r, 클래스, 증강 라벨) 조건을 넣습니다.
  - JVP를 컴파일하기 위해 GroupNorm은 기본 연산으로 직접 구현했습니다.
- **시간 임베딩**: t와 h를 **×1 그대로** sinusoidal 임베딩에 넣습니다. ×1000을 쓰면 JVP 타깃이 크게 흔들려 학습이 실패합니다.
- **(r, t) 샘플링**: logit-normal(−0.4, 1)에서 뽑고, 배치의 25%를 r ≠ t로 둡니다.
  - 2단계부터는 r ≠ t 샘플의 절반을 **실제 추론 구간**에서 뽑습니다(`--fixed_frac 0.5`).
  - 3~5단계 구간: 1스텝 구간 (0, 1)은 가중치 3, ρ=0.5 4스텝 구간 4개는 각 가중치 1입니다.
- **손실 가중치**: adaptive loss weight를 쓰고, p = 1입니다.
- **CFG**: 학습 타깃에 미리 반영합니다. 실효 강도 ω/(1−κ) = 2.0이고(ω=1, κ=0.5), 클래스 dropout은 10%입니다.
- **규제**: 3~5단계에서 dropout 0.2와 non-leaky 기하 증강(p = 0.15, 증강 라벨 조건, `meanflowmodel/augment.py`)을 씁니다. 1~2단계는 dropout 0.1, 증강 없음입니다.
- **최적화**: Adam(0.9, 0.999), lr 1e-4, grad clip 1.0, 배치 64, 데이터 좌우 반전을 씁니다.
- **EMA**: 0.9999 가중치를 제출합니다.
- **학습량**: 5단계 합계 30만 스텝(약 1,920만 장)입니다. 최대 VRAM은 9.5 GB입니다.

### 단계별 명령 (`meanflowmodel/`)
```bash
REG="--augment_p 0.15 --dropout 0.2 --fixed_frac 0.5 \
     --fixed_intervals 0:1:3,0.866:1:1,0.707:0.866:1,0.5:0.707:1,0:0.5:1"
# 1. 기본 학습 200k
python train.py --steps 200000 --decay_start 150000 --run_dir runs/meanflow
# 2. 추론 구간 fine-tune F1 10k
python train.py --init_from runs/meanflow/train_state.pt --steps 10000 --lr 5e-5 --warmup 500 \
    --fixed_frac 0.5 --run_dir runs/ft/F1 --checkpoint_dir runs/ft/F1/ckpt
# 3. 규제 fine-tune 40k
python train.py --init_from runs/ft/F1/train_state.pt --steps 40000 --lr 1e-4 --warmup 1000 \
    $REG --ema_decays 0.9999 0.999 --run_dir runs/meanflow_reg --checkpoint_dir runs/meanflow_reg/ckpt
# 4. 이어서 G0 10k
python train.py --init_from runs/meanflow_reg/train_state.pt --steps 10000 --lr 1e-4 --warmup 500 \
    $REG --ema_decays 0.999 0.9999 --run_dir runs/ft2/G0 --checkpoint_dir runs/ft2/G0/ckpt
# 5. 마무리 60k (30k부터 lr 선형 감소, 5k마다 EMA 스냅샷)
python train.py --init_from runs/ft2/G0/train_state.pt --steps 60000 --lr 1e-4 --warmup 500 \
    --decay_start 30000 $REG --ema_decays 0.9999 0.999 --snapshot_every 5000 \
    --run_dir runs/meanflow_final --checkpoint_dir checkpoints
```

### 체크포인트 선택 (train split 지표만 사용, `runs/eval/select_final.tsv`)
| 후보 (5단계) | train FID, 4 NFE | 암기 점수 (높을수록 덜 베낌) |
|---|---|---|
| **40k** | **25.19** | **0.2368** |
| 45k | 24.40 | 0.2341 |
| 50k | 24.25 | 0.2328 |
| 55k | 24.20 | 0.2317 |
| 60k | 24.11 | 0.2306 |

- 암기 점수는 4 NFE 샘플마다 같은 클래스에서 가장 가까운 학습 이미지까지의 픽셀 RMSE를 구한 뒤, 그 중앙값을 낸 값입니다.
- 40k보다 늦은 스냅샷은 train FID가 더 낮았습니다. 하지만 학습할수록 암기 점수가 계속 내려갔습니다. 즉 학습 이미지를 베끼는 쪽으로 가고 있었습니다.
- 그래서 암기 점수가 암기하지 않는 기준 모델 수준에 머무는 스냅샷 중 train FID가 가장 좋은 **40k**를 골랐습니다. train FID는 학습 이미지와 비교하는 지표라서, 베끼기만 해도 점수가 좋아지기 때문입니다.

### 샘플링 (`ModelFewNFE`)
시간 격자는 t = 1 → 0.866 → 0.707 → 0.5 → 0이고(`t_i = (1 − i/4)^0.5`), `z_r = z_t − (t − r)·u(z_t, r, t)`로 이동합니다. 정확히 4 NFE입니다.

ρ = 0.5는 train 기준 FID로 {0.25, 0.35, 0.5, 0.75, 1, 1.5, 2} 중에서 골랐습니다.

### 제출 가중치
`meanflowmodel/runs/meanflow_final/ema_0040000.ckpt` (이전 제출본에서는 `checkpoints/few_nfe.ckpt`, 지금은 `../submission_meanflow_backup/checkpoints/few_nfe.ckpt`)

---

## 3. 제출 폴더 구성 (`team_16/`)

```text
team_16/
├── train.py              # 학습 진입점: python train.py list (전체 재현 명령), python train.py <단계> ...
├── model.py              # Model 기본 클래스는 원본 그대로; ModelOneNFE = D2O 생성기 1-step, ModelFewNFE = EDM2 consistency 4-step
├── d2o/                  # 팀원 D2O 코드 중 생성기 정의 (__init__, config, edm2; 저장소 commit 32f0704 그대로)
├── gan_networks.py       # = ganmodel/networks.py (이전 one_nfe와 training/ectmodel/adv_cmt.py용)
├── teacher/              # 팀원 EDM2 teacher 코드 전체 (샘플링은 __init__, networks, official만 사용; 학습은 python -m teacher.train)
├── third_party/          # NVlabs EDM2/EDM 공식 코드 (수정 없음), THIRD_PARTY.md 참고
├── dataset.py, download_dataset.py, evaluate.py, src/, requirements.txt   # 원본 그대로
├── training/             # 학습 코드 (training/README.md): d2o_one/ (one_nfe), ectmodel/ (few_nfe), teacher_e112/ (teacher 문서)
└── checkpoints/
    ├── one_nfe.ckpt      # 팀원 D2O 생성기 EMA, e112_treal_lr3 calib-005825 (Release handoff-2026-10-27 파일 그대로)
    └── few_nfe.ckpt      # E112 구조 consistency 모델, guided SiD (fake score = 학생 + LoRA) EMA @ 70 kimg (runs/gbase12_lr1e4/ema500_000070kimg.ckpt)
```

**학습 코드**: `training/` 폴더에 넣었습니다(`training/README.md`).
- `training/d2o_one/`: 현재 `one_nfe`(D2O) 학습 코드 (팀원 저장소 그대로)
- `training/ectmodel/`: 현재 `few_nfe` 학습 코드 (궤적, CMT, ECD, SiD)
- `training/ganmodel/`: R3GAN 학습, 합성 데이터 생성, held-out 분할, 평가 스크립트
- `training/meanflowmodel/`: MeanFlow 학습 코드
- 이전 `one_nfe`(v1)는 `ganmodel/runs/r3gan/G_ema_009000kimg.ckpt`에 그대로 있습니다.

## 4. 규칙 준수 확인
- 모델별 파라미터는 R3GAN 생성기 22.5M, 판별기 23.1M, MeanFlow 92.5M, few 모델 95.2M(+12,288 고정), E112 teacher 95.2M으로, 모두 100M 이하입니다.
  - MeanFlow는 `one_nfe` 학습에서 teacher로도 쓰입니다(합성 데이터 생성). 규칙상 teacher와 student는 따로 세며, 각각 100M 이하여야 합니다.
  - 팀원 E112 teacher(실험에 사용): 95.2M(+12,288 고정), 학습 최대 VRAM 19.0 GB, train split만 사용.
  - teacher는 샘플링 때 실행하지 않습니다.
  - 현재 `one_nfe`(D2O): 제출 모델 95,158,709, 학습 framework(생성기, 판별자 고유 부분, fake score LoRA) 99,992,471, teacher T58 95,158,709(고정, 별도 모델).
  - 현재 `few_nfe`(SiD 단계): 제출 모델 95,175,157, 학습 framework(생성기, 생성기와 텐서를 공유하는 fake score의 LoRA) 98,781,577, E112 teacher(고정, 별도 모델). 이전 단계(CMT, ECD)는 생성기 하나만 학습합니다.
  - EMA 사본을 같은 모델의 parameter로 따로 셀지는 해석이 갈립니다. 따로 세면 D2O 학습 framework는 100M을 넘습니다(팀원 인계 문서 6.3절). 이 점은 EMA를 쓰는 `few_nfe`와 E112 teacher도 같습니다.
- 학습 최대 VRAM은 R3GAN 16.8 GiB(`ho_v3mf`), MeanFlow 9.5 GB, few 모델(CMT 11.3 GiB, ECD 13.6 GiB, SiD 9.9 GB), D2O 12.8 GB로, 모두 20 GB 미만입니다.
- 사전학습 모델은 쓰지 않았습니다. Inception은 FID 계산에만 쓰이며, 공식 평가와 같은 용도입니다.
- `train_split.txt`만 학습에 썼습니다. 설정과 체크포인트 선택은 train split에서 떼어 낸 held-out FID로만 했습니다(v1은 train FID). 현재 `one_nfe`의 체크포인트는 팀원이 미리 정한 teacher 표본 기준 FID로, latent 설계와 noise seed는 held-out FID로 골랐습니다.
- validation FID는 보고용으로만 썼습니다. 예외였던 SiD 스냅샷(2-c절, val FID로 고름)은 지금 제출본에 쓰지 않습니다. 현재 `few_nfe`의 스냅샷, 샘플러, noise seed는 모두 held-out FID로 미리 정한 규칙에 따라 골랐습니다.
- 고정 파일(`dataset.py`, `evaluate.py`, `src/`, split 파일)과 `Model` 기본 클래스는 수정하지 않았습니다.
- NFE는 `one_nfe` 1회(생성기 forward 1회), `few_nfe` 4회(네트워크 forward 4회)입니다.
- 추가 코드: `teacher/`, `d2o/`(팀원 코드)와 `third_party/`(NVlabs 공식 코드, CC BY-NC-SA 4.0)는 네트워크 정의만 불러오며, numpy와 torch 외 의존성이 없습니다.
- 추가 라이브러리는 설치하지 않았습니다. `requirements.txt`의 패키지와 PyTorch만 씁니다.

## 5. 참고 문헌
- Y. Zheng, T. Yang. *Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation* (D2O). ICML 2025. arXiv 2506.09376. 코드: github.com/Zyriix/D2O (commit `8857b2d`, 팀원이 `d2o/networks.py`로 포팅)
- T. Yin et al. *Improved Distribution Matching Distillation for Fast Image Synthesis* (DMD2). NeurIPS 2024. 코드: github.com/tianweiy/DMD2 (D2O 학습의 distribution matching 항)
- E. J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. (D2O 학습의 fake score, 직접 구현)
- Y. Huang, A. Gokaslan, V. Kuleshov, J. Tompkin. *The GAN is dead; long live the GAN! A Modern GAN Baseline*. NeurIPS 2024. 코드: github.com/brownvc/R3GAN
- T. Karras et al. *Training Generative Adversarial Networks with Limited Data* (ADA). NeurIPS 2020.
- T. Karras et al. *Analyzing and Improving the Image Quality of StyleGAN* (lazy regularization). CVPR 2020.
- Z. Geng, M. Deng, X. Bai, J. Z. Kolter, K. He. *Mean Flows for One-step Generative Modeling*. 2025.
- T. Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models* (non-leaky augmentation). NeurIPS 2022.
- P. Dhariwal, A. Nichol. *Diffusion Models Beat GANs on Image Synthesis* (ADM UNet). NeurIPS 2021.
- Z. Geng et al. *Consistency Models Made Easy* (ECT/ECD). ICLR 2025.
- Z. Hu et al. *Consistency Mid-Training*. arXiv 2509.24526, 2025.
- M. Zhou, H. Zheng, Z. Wang, M. Yin, H. Huang. *Score identity Distillation* (SiD). ICML 2024. / M. Zhou, Z. Wang, H. Zheng, H. Huang. *Guided Score identity Distillation for Data-Free One-Step Text-to-Image Generation*. ICLR 2025. / M. Zhou, Y. Gu, Z. Wang. *Few-Step Diffusion via Score identity Distillation*. arXiv 2505.12674. (few_nfe 마지막 단계, 직접 구현)
- J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop. (SiD의 teacher score)
- T. Karras et al. *Analyzing and Improving the Training Dynamics of Diffusion Models* (EDM2, E112 teacher의 구조). CVPR 2024. / *Guiding a Diffusion Model with a Bad Version of Itself* (autoguidance). NeurIPS 2024.
- M. Shahbazi et al. *Collapse by Conditioning: Training Class-conditional GANs with Limited Data*. ICLR 2022. (코드에 옵션으로 구현했으나 이번 제출본에는 쓰지 않음)
- H.-Y. Tseng et al. *Regularizing Generative Adversarial Networks under Limited Data* (LeCam). CVPR 2021. (옵션으로 구현, 미사용)
