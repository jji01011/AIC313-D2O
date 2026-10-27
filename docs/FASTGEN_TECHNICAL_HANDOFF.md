# FastGen One-NFE Technical Handoff (D2O 기반 one-step 생성기)

> 이 문서는 팀원과 공동연구자에게 현재 one-NFE 구현을 넘기기 위한 기술 문서입니다.
> 기존 설명이나 대화 기록을 사실로 가정하지 않았습니다. 근거는 실제 코드(`파일:라인`), checkpoint 메타데이터, 학습 로그, CPU에서 다시 실행한 parameter count입니다.
> 사람이 남긴 기록(`STATUS.md`)은 "왜 그렇게 했는가"를 설명할 때만 쓰고, 그때마다 출처를 적었습니다.
> 확인하지 못한 내용은 `확인 불가`로 표시했습니다.

---

## 0. 문서 요약

### 0.1 조사 정보

| 항목 | 내용 |
|---|---|
| 조사 날짜 | 2026-10-25 (UTC 10:10부터) |
| 조사 대상 저장소 | `/home/jiin/AIC313-D2O` |
| Git branch / HEAD | `main` / `3ccf3261495bd144bf844b7f298d10406b53117a` ("update readme") |
| HEAD의 정체 | 과제 원본 저장소(`origin` = `https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen`)의 commit입니다. **우리 구현은 하나도 commit되어 있지 않습니다.** |
| commit되지 않은 변경 | `model.py` 수정(M). 나머지는 전부 untracked: `d2o/`, `train_d2o.py`, `train_edm.py`, `calibrate_generator.py`, `check_d2o.py`, `eval_edm.py`, `diagnose_samples.py`, `diagnostics/`, `runs/`(스크립트와 결과), `results/`, `checkpoints/`, `STATUS.md` |
| 다른 remote | `junseok` = `https://github.com/junseokl/FastGen.git` (팀원 저장소, 비공개라 이 서버에서는 읽지 못함). 같은 저장소의 zip 두 개가 루트에 있음: `FastGen-main.zip`(zip comment의 commit `49e25296da5361cf9322bf37c4b31b3d249d583f`), `FastGen-main (1).zip`(`c7f3a3a7306ec6be93e3180f31389a7d307beb84`). 내용은 팀원의 teacher 학습 코드(`edm_teacher/`)입니다. |
| 조사 범위 | one-NFE(`ModelOneNFE`)만. few-NFE는 팀원 담당이고 이 저장소에는 학습 기록이 없습니다 (`model.py`의 `ModelFewNFE`는 구현만 있고 학습한 적 없음). |
| 조사 중 수정한 것 | 없음. 학습 프로세스, checkpoint, 로그, config, 코드는 건드리지 않았습니다. CPU에서 parameter 수와 tensor shape만 다시 셌습니다 (부록 A). |

### 0.2 현재 모델 한 단락 요약

팀원이 train split만으로 처음부터 학습한 **EDM2 U-Net(preset E112, 95.16M)**을 출발점으로 씁니다. 이 네트워크를 **σ=80 한 번 호출로 이미지를 내는 one-step 생성기**로 미세조정했습니다.
학습 중에는 embedding과 encoder(29.4M)를 고정하고 decoder(65.7M)만 학습합니다.
목적함수는 두 가지를 더한 것입니다.
- **D2O식 GAN loss**: 판별자가 별도 backbone 없이 생성기 자신의 고정 encoder를 특징 추출기로 공유합니다.
- **DMD2식 distribution matching 항**: real score는 고정된 teacher T58이고, fake score는 생성기 가중치를 그대로 읽으면서 decoder에 붙인 LoRA(rank 6)만 학습합니다.

지금 최고 성능 줄(`e112_treal*`)에서는 판별자가 보는 "진짜" 이미지가 train 이미지가 아니라 teacher T58의 표본입니다.
샘플링은 `ModelOneNFE.sample()`이 backbone을 **딱 한 번** 실행하고, 출력 gain 1.10은 가중치에 미리 접어 넣었습니다.

### 0.3 가장 좋은 확인된 성능

| 구분 | checkpoint | 선택 기준 값 | 공식 FID (evaluate.py, val, 3회) |
|---|---|---|---|
| **공식 FID 최저 (보고용으로 측정된 값 중)** | `runs/e112_treal_lr3/calib-005675.ckpt` | teacher 기준 10.498 | **21.36** (21.45, 21.36, 21.26) |
| **미리 정한 선택 규칙의 후보** (그 줄의 teacher 기준 최저) | `runs/e112_treal_lr3/calib-005825.ckpt` | teacher 기준 **10.460** | 21.74 (21.74, 22.07, 21.41) |
| 비교: held-out 줄의 후보 | `runs/e112_h20/calib-003150.ckpt` | H 기준 16.954 | 21.57 |
| 비교: train 기준 최저 | `runs/e112_lorag_f5_g13_lr3b/calib-006825.ckpt` | train 기준 11.303 | 22.79 |
| teacher T58 (63 NFE, 팀원 방식 샘플링) | `checkpoints/edm/teacher_unguided-0058720256-0.130.pt` | train 기준 10.095 | 20.32 (표본 한 세트) |
| 실제 train 이미지 (클래스당 20장) | - | train 기준 6.5 | 20.89 (20.48~21.41, 무작위 10회) |
| TA 1-NFE 기준 | - | - | 39.15 |

- 근거: `results/official_fid.log`, `results/tref_calibration.log`, `results/teacher_handoff_eval.log`. 실제 train 이미지의 10회 값은 로그 파일이 없고 `STATUS.md:1342`의 기록만 있습니다 (`diagnostics/coverage_probe_2026-10-10/floor.py`로 측정).
- 공식 FID는 `evaluate.py`가 seed를 고정하지 않아 같은 checkpoint도 측정마다 ±0.3~0.5 흔들립니다. 그래서 21.36과 21.74의 차이(0.38)는 측정 흔들림 범위 안입니다.
- **제출 파일 `checkpoints/one_nfe.ckpt`는 아직 없습니다** (중간 마감 10-31 23:59 KST). 줄 사이의 선택 규칙도 아직 정하지 않았습니다 (9.2절).

### 0.4 주요 구성요소별 parameter 수 (현재 줄, CPU에서 다시 셈)

| 구성요소 | 등록 parameter | 학습 / 고정 | 비고 |
|---|---|---|---|
| Generator `G` (`EDM2Precond`, E112) | 95,158,709 | 학습 65,733,606 (decoder + out) / 고정 29,425,103 (embedding + encoder) | 샘플링에 쓰는 유일한 네트워크 (EMA 사본으로) |
| Discriminator 고유 부분 | 3,610,032 | 학습 3,355,908 (head 3개) / 고정 254,124 (random projection) | encoder는 `G`의 것을 빌려 쓰고 등록하지 않음 |
| Fake score LoRA 가지 | 1,223,730 | 학습 | base는 `G` 자신 |
| **학습 framework 합계 (고유 텐서)** | **99,992,471** | 학습 70,313,244 / 고정 29,679,227 | 100M 한도 안 (팀 기준) |
| EMA generator (`ModelOneNFE`) | 95,158,709 | 학습 안 함 (EMA로 갱신) | 팀 기준에서는 같은 모델로 보고 따로 세지 않음. 따로 세면 195,151,180 (6.3절) |
| Teacher T58 (real score) | 95,158,709 | 고정 | README상 별도 모델 (student와 합산하지 않음) |
| 제출 모델 (`ModelOneNFE`, evaluate.py가 셈) | 95,158,709 | - | `parameter_summary` = total 95,158,709 |

### 0.5 주요 참고 논문과 원본 코드

| 방법 | 논문 | 공식 코드 (로컬 clone, commit) | 우리 쪽 관계 |
|---|---|---|---|
| D2O | https://arxiv.org/abs/2506.09376 (ICML 2025, 이전 판 https://arxiv.org/abs/2405.20750) | https://github.com/Zyriix/D2O (`/home/jiin/D2O`, `8857b2dd09c51384974948e1a190773e95f3c09b`) | 네트워크, 판별자 head, 학습 규칙을 포팅해 변형 |
| DMD2 | https://arxiv.org/abs/2405.14867 (NeurIPS 2024) | https://github.com/tianweiy/DMD2 (`/home/jiin/dmd2-official`, `8d8fa55633d47cfb81bbc7a892e7248f9518763f`) | DM loss와 fake score loss를 **기억으로 재구현**한 뒤 공식 코드와 대조 (복사 아님) |
| EDM2 | https://arxiv.org/abs/2312.02696 (CVPR 2024) | https://github.com/NVlabs/edm2 (`/home/jiin/edm2-official`, `4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc`) | 네트워크 블록 포팅 (`d2o/edm2.py`). teacher E112는 팀원이 이 코드로 학습 |
| EDM | https://arxiv.org/abs/2206.00364 (NeurIPS 2022) | https://github.com/NVlabs/edm (로컬 clone 없음. 팀원 코드에 commit `008a4e5316c8e3bfe61a62f874bddba254295afb`로 포함) | D2O 경유 포팅 (SongUNet, EDMPrecond, Heun sampler, EDM loss) |
| LoRA | https://arxiv.org/abs/2106.09685 | 사용 안 함 | 아이디어만 사용, 직접 구현 (`d2o/lora.py`) |

전체 표는 2절에 있습니다.

### 0.6 확인하지 못한 사항 (요약, 자세한 것은 9.5절)

1. **코드 버전을 commit으로 고정할 수 없습니다.** 우리 코드가 전부 commit되지 않았습니다. 실험 도중 `train_d2o.py`가 여러 번 바뀌었고 (`diagnostics/backup_*`, `runs/*.before-*`에 이전 판이 있음), 각 run이 시작될 때의 정확한 코드 판은 `확인 불가`입니다. 지금 파일은 새 옵션의 기본값이 전부 꺼져 있어서, 이전 run들의 인자로 실행하면 같은 경로를 탈 것으로 보입니다. 다만 다시 실행해 확인하지는 않았습니다.
2. **EMA 복사본의 산정 여부를 TA가 명시적으로 답한 기록이 없습니다.** 팀 기준(2026-10-09 사용자 결정)은 "EMA 사본은 같은 모델"입니다.
3. **처음부터 한 번에 다시 학습해 같은 숫자가 나오는지 확인하지 않았습니다.** 현재 모델은 7단계 lineage를 50~100 kimg 조각으로 이어 붙인 결과입니다. 조각마다 RNG를 seed 0으로 다시 시작합니다 (`train_d2o.py:112`).
   중간 단계의 상태 파일도 대부분 남아 있지 않습니다. 예를 들어 P7(`e112_treal_lr3`)이 출발한 `e112_treal` 5525 상태는 그 뒤 5725까지 덮어써졌습니다 (4.1절).
4. **`e112_treal*` checkpoint의 train 이미지 근접도(암기 점검)는 재지 않았습니다.** 이전 줄은 쟀고 near-copy 0.00~0.03%였습니다. teacher T58 자신은 0.40%입니다.
5. few-NFE 모델, 팀원의 DMD2 few-step 코드, 팀원 GitHub은 이 서버에서 볼 수 없습니다.
6. TA 답변 원문은 사용자가 대화에 붙여 넣은 것을 `STATUS.md`에 옮겨 둔 것만 있습니다 (원 메시지는 보지 못함).

---

## 1. Current Architecture — 현재 모델 전체 구조

### 1.1 큰 그림

현재 최고 성능 줄(`e112_treal_lr3`)과 그 직전 줄들의 학습 framework는 아래와 같습니다. 판별자의 "진짜" 이미지만 줄마다 다릅니다.

```text
====================================== 학습 시 (e112_treal_lr3, 1 step = 64장) ======================================

 z ~ N(0, I) [N,3,64,64] ──×80──┐
 class c ∈ {0..150}  ──one-hot──┤
                                ▼
 ┌──────────────────────── Generator G_θ = EDM2Precond (E112 구조, 95,158,709) ────────────────────────┐
 │  x_c = x − mean_image                                                                             │
 │  emb = MPFourier(c_noise) → emb_noise ⊕ emb_label(one-hot 152칸)     ┐                              │
 │  encoder unet.enc.* (64→32→16→8)                                     ├─ 고정 29,425,103 (eval 유지) │
 │  decoder unet.dec.* + out_conv  ← skip                               ── 학습 65,733,606             │
 │  out = mean_image + c_skip·x_c + c_out·F(x)                                                       │
 └───────────────┬───────────────────────────────────────────┬──────────────────────────────────────┘
                 │ x_fake = G(80z, 80, c) [N,3,64,64]          │ 같은 nn.Parameter 객체 (복사 없음)
                 │                                           │
     ┌───────────┴──────────────┐              ┌─────────────┴───────────────────────────────┐
     ▼                          ▼              ▼                                             ▼
 Discriminator D_φ           DM 항          Fake score F_ψ(x_t,σ,c)                   (D의 특징 추출기)
 x + σ_D·ε, σ_D~logN(-2,1)   x_t = x+σε     = G의 가중치 (상수, eval, grad 없음)          G의 고정 encoder
   → G.features(): 4개 해상도  σ∈Karras    + decoder/out conv 77개에 LoRA rank 6          features {64,32,16,8}
   → CCM/CSM 랜덤 projection   [0.002,80]    (1,223,730, 학습)
     (고정 254,124)            p_real = x − T(x_t)
   → head 0,2,3 (학습        p_fake = x − F_ψ(x_t)
     3,355,908)              grad = (p_real−p_fake)/mean|p_real|
   → logits [N,75]
       ▲
 "진짜" 입력 = teacher T58 표본 51,340장 (e112_treal*), 이전 줄은 train 이미지

 Real score T = Teacher T58 (EDM2Precond, 95,158,709, 전부 고정, 별도 모델)
 EMA:  Ĝ ← lerp(G, Ĝ, β)  (ModelOneNFE, 95,158,709, 매 step, 학습 안 함)

=========================================== 샘플링 시 (evaluate.py) ===========================================

 z, c ──▶ Ĝ(80·z, σ=80, one-hot(c)) 한 번 ──▶ clamp(−1, 1)        ← 판별자, fake score, teacher 전부 없음
          (gain 1.10이 out_gain과 mean_image에 이미 들어 있음)
```

코드 위치:
- 생성기 구성: `train_d2o.py:137-143` (`--g_edm2`), `train_d2o.py:194` (`apply_freeze`)
- 판별자 구성: `train_d2o.py:173-191`
- fake score 구성: `train_d2o.py:220-241`
- teacher: `train_d2o.py:222`
- EMA: `train_d2o.py:154-161`, `train_d2o.py:732`
- 샘플링: `model.py:97-112`

### 1.2 구성요소별 상세

#### (1) Generator G — `EDM2Precond` (`d2o/edm2.py:287-359`)

| 항목 | 내용 |
|---|---|
| 역할 | one-step 생성기. 샘플링 때 실제로 쓰이는 유일한 네트워크입니다 (EMA 사본으로). |
| backbone | EDM2의 magnitude-preserving U-Net. `model_channels` 112, `channel_mult` [1,2,3,4], 해상도당 block 3개, attention은 16x16과 8x8, `channels_per_head` 56, `sigma_data` 0.6348 (`d2o/config.py:31-44`, checkpoint의 `net_kwargs`와 같음) |
| 입력 | `x` [N,3,64,64] (= 80·z), `sigma` [N] (= 80), `class_labels` [N,151] one-hot (내부에서 152칸으로 확장. 마지막 칸은 null class: `d2o/edm2.py:260-264`) |
| 출력 | [N,3,64,64]. `mean_image + c_skip·(x − mean_image) + c_out·F(x)` (`d2o/edm2.py:356-359`). σ=80에서는 c_skip ≈ 6.3e-5라 사실상 `mean_image + 0.635·F(x)` |
| checkpoint | 초기값 `checkpoints/edm/ema-0020971520-0.100.pt` (팀원 E112, 2,097만 장 시점 EMA). 현재 상태 `runs/e112_treal_lr3/d2o-state-latest.pt`의 `net` |
| 학습 / 고정 | `--freeze_encoder 1`: 이름이 `unet.emb_`, `unet.enc.`로 시작하는 parameter를 고정합니다 (`d2o/training_utils.py:24`, `:27-40`). `encoder_frozen=True`면 `train()`을 불러도 embedding과 encoder는 eval 모드로 남습니다 (`d2o/edm2.py:316-321`). 그래서 forced weight normalization(`d2o/edm2.py:84-86`)이 고정 가중치를 다시 쓰지 않습니다. |
| 공유 | 판별자의 특징 추출기와 fake score의 base가 이 객체 자체입니다 (5절). |
| 사용 시점 | 학습과 샘플링 (샘플링은 EMA 사본) |
| parameter | 95,158,709 (학습 65,733,606, 고정 29,425,103) |

체크포인트 파일에는 있지만 생성기에 넣지 않은 것이 둘 있습니다: uncertainty head(`logvar_*`, 학습 전용)와 augmentation embedding(`emb_augment`, 입력이 0이라 상수 배율로 대체).
그래서 팀원 파일의 95,162,869보다 4,160개 적습니다 (`d2o/edm2.py:227-229`, `:312-313`).

**dropout은 쓰지 않습니다.** 팀원의 teacher 학습 설정에는 `dropout_res` {64: 0, 32: 0.15, 16: 0.15, 8: 0.15}가 있고 calib checkpoint의 `net_kwargs`에도 그대로 남아 있습니다.
하지만 우리 포팅은 `Block`을 만들 때 dropout 인자를 넘기지 않아 기본값 0이 됩니다 (`d2o/edm2.py:101`, `:133-134`, 블록 생성 `:181-212`). 그래서 생성기 미세조정과 fake score pass 모두 dropout 없이 돕니다.

#### (2) EMA Generator — `ModelOneNFE` (`model.py:79-127`)

| 항목 | 내용 |
|---|---|
| 역할 | 평가, 스냅샷, 제출에 쓰는 생성기. raw 생성기보다 훨씬 좋습니다 (우리 34.8M 생성기에서 raw 31.5 대 EMA 24.4, `STATUS.md` 2026-10-08 기록). |
| 갱신 | 매 step `p_ema ← p_net.lerp(p_ema, β)`, β = 0.5^(batch / halflife), halflife = min(500 kimg, 0.05 × cur_nimg) (`d2o/training_utils.py:44-51`, 호출 `train_d2o.py:732`). 5,500 kimg 시점의 halflife는 약 275 kimg입니다. |
| 학습 | gradient 없음 (`requires_grad_(False)`, `train_d2o.py:161`) |
| 저장 | 25 kimg마다 `d2o-*.ckpt`(=`{"state_dict": ModelOneNFE.state_dict()}`)를 쓰고 FID 측정 뒤 지웁니다. 최저만 `best.ckpt`로 남습니다 (`train_d2o.py:528-547`). |
| parameter | 95,158,709 |

#### (3) Discriminator D — `ProjectedDiscriminator` + `EDM2FeatureNetwork(shared=True)` (`d2o/discriminator.py:175-246`)

| 항목 | 내용 |
|---|---|
| 역할 | 진짜 이미지와 생성 이미지를 가리는 GAN 판별자 |
| 구조 | ① 입력에 노이즈: σ_D = exp(N(−2.0, 1.0)), [0.02, 80]으로 clamp, x + σ_D·ε (`d2o/discriminator.py:234-243`). ② **생성기 G의 고정 encoder**로 해상도 64/32/16/8의 특징을 뽑음 (`G.features`, `d2o/edm2.py:281-284`). ③ 고정된 무작위 1x1 conv(CCM)와 cross-scale fusion(CSM) (`d2o/discriminator.py:195-196`). ④ 해상도마다 projection-conditioned head (`SingleDiscCond`, `d2o/discriminator.py:65-88`, `DownBlock` `:52-59`): spectral-norm conv 4x4/2 + BatchNorm + LeakyReLU로 8x8까지 줄인 뒤 4x4 conv로 5x5 map, 클래스 embedding(151×128 → 64)과 내적 |
| 입력 / 출력 shape (CPU에서 확인) | encoder 특징 {64: [N,112,64,64], 32: [N,224,32,32], 16: [N,336,16,16], 8: [N,448,8,8]} → projection 뒤 {'0': [N,36,128,128], '1': [N,36,64,64], '2': [N,72,32,32], '3': [N,144,16,16]} → head 0, 2, 3이 각각 [N,1,5,5] → logits [N,75] |
| 폭 | `--d_width 0.5625`: cout = round(64 × 0.5625) = 36, head 채널도 0.5625배 (`train_d2o.py:180`, `:183-186`) |
| head 제거 | `--d_drop_heads 1`: 32x32 특징 단계의 head를 지웠습니다 (`train_d2o.py:190-191`). 1950 kimg에서 LoRA 자리를 만들려고 학습된 판별자에서 구별력이 가장 약한 head를 뺐습니다 (`STATUS.md` 2026-10-11 09:25 측정표). 남은 head는 상태에서 그대로 읽습니다 (`train_d2o.py:257-267`). |
| 학습 / 고정 | head만 학습 (3,355,908). CCM/CSM은 등록되지만 `requires_grad_(False)` (254,124, `:199`). encoder는 등록조차 하지 않습니다 (`object.__setattr__`, `:187-190`). |
| 사용 시점 | 학습만 |
| parameter | 등록 3,610,032 |

D2O 공식 판별자는 ImageNet 사전학습 특징 추출기(DeiT/EfficientNet 또는 VGG16-BN/EfficientNet, `/home/jiin/D2O/train.py:234-241`)를 씁니다.
과제 규정상 사전학습 모델을 쓸 수 없어서, 그 자리를 우리가 학습한(또는 팀원이 train split으로 학습한) diffusion encoder로 바꿨습니다 (`d2o/discriminator.py:1-17`).
diffusion encoder는 노이즈가 섞인 입력으로 학습됐으므로 특징을 뽑기 전에 노이즈를 넣습니다. 공식 D2O에는 이 단계가 없습니다.

#### (4) Fake Score F — `LoRAFakeScore(base=G, rank=6, shared=True, decoder_only=True)` (`d2o/lora.py:29-83`)

| 항목 | 내용 |
|---|---|
| 역할 | DMD2의 "fake score": 지금 생성기 분포의 denoiser입니다. teacher(real score)와의 차이가 생성기를 미는 방향이 됩니다. |
| 구조 | G의 네트워크를 그대로 쓰고, decoder(`dec.*`)와 `out_conv`의 4차원 convolution 77개에 저차원 가지를 forward hook으로 붙입니다. 가지 = `conv(conv(x, down[6,Cin,k,k]), up[Cout,6,1,1])` (`:43-60`) |
| 입력 / 출력 | `x_t` [N,3,64,64], `σ` [N] (Karras 격자 0.002~80), class one-hot → denoised [N,3,64,64] |
| 초기값 | `up`이 0이라 처음에는 G와 정확히 같은 출력 (CPU에서 확인). 상태 파일의 `fake_score`에서 이어받습니다 (`train_d2o.py:275-277`). |
| 학습 / 고정 | LoRA 가지만 학습. base(G)의 가중치는 fake score pass 동안 `requires_grad=False`, eval 모드 (`:66-74`) |
| 사용 시점 | 학습만 (DM 항 계산, fake score loss) |
| parameter | 1,223,730 (down/up 77쌍) |

#### (5) Real Score / Teacher T — `EDM2Precond.from_checkpoint(T58)` (`train_d2o.py:222`)

| 항목 | 내용 |
|---|---|
| 역할 | 데이터 분포의 denoiser (DMD2의 real score). 현재 줄에서는 판별자가 보는 "진짜" 표본의 출처이기도 합니다. |
| checkpoint | `checkpoints/edm/teacher_unguided-0058720256-0.130.pt` = 팀원 hand-off의 `teacher_unguided.pt` (5,872만 장, post-hoc EMA σ_rel 0.13). `_lr3b`, `h20`까지의 줄은 E112 21.0M(`ema-0020971520-0.100.pt`)을 썼습니다. |
| 학습 / 고정 | 전부 고정 (`eval().requires_grad_(False)`) |
| 계산 | float32로 실행 (`force_fp32=True`, `train_d2o.py:334`). `--dmd_guidance w ≠ 1`이면 0.25 < σ ≤ 4 구간에서 null class로 한 번 더 돌려 `null + w·(cond − null)` (`:335-338`). 현재 줄은 w = 1.0 (꺼짐), `_g13`/`_lr3b`/`h20`은 1.3 |
| 공유 | 생성기와 공유하는 텐서 0개 (CPU에서 확인) |
| parameter | 95,158,709, 별도 모델 |

#### (6) Teacher 표본 데이터 (현재 줄의 "진짜" 이미지)

- `runs/gen_teacher58.sh`가 `diagnostics/teammate_edm2/generate_teacher_samples.py`로 만들었습니다. 설정은 Heun 32단계, 시작점 `mean_image + 80·noise`, 클래스당 20장입니다.
  - `data/cache/teacher58_real/`: 17 shard × 3,020 = 51,340장, seed 1000부터. 판별자가 진짜로 봅니다.
  - `data/cache/teacher58_ref/`: 5 shard, 15,100장, seed 5000부터. 판별자는 보지 않고 선택 기준(`results/reference_teacher58`)으로만 씁니다.
- 학습 루프는 진짜 batch를 꺼낸 뒤 확률 `--extra_real_p 1.0`으로 전부 teacher 표본으로 바꿉니다 (`train_d2o.py:296-304`). 그래서 현재 줄에서는 판별자, R1, 생성기 label의 class 분포가 전부 teacher 표본 쪽(클래스 균등)을 따릅니다.

#### (7) 출력 보정 (gain) — `calibrate_generator.py`

- 학습 뒤 EMA 생성기에 gain g를 곱해 넣습니다: `out_gain ← out_gain·g`, `mean_image ← mean_image·g` (`calibrate_generator.py:100-111`). g는 1.0/1.1/1.2 중 선택 기준 FID(시드 3개 평균)가 가장 낮은 값이고, 현재 후보는 1.10입니다.
- 네트워크 실행 횟수는 그대로 1입니다. checkpoint에서 확인한 값: `out_gain` 0.1537(E112 초기값 0.1764), `mean_image` 평균 0.19712 = 초기값 0.17920 × 1.1.
- EDM2 생성기에는 smoothing이 해로워서 쓰지 않습니다 (`STATUS.md` 2026-10-09 17:27 측정).

### 1.3 Generator와 Fake Score의 가중치 공유

- `LoRAFakeScore.__init__`은 base를 `self._base = (base,)` 튜플에 넣습니다 (`d2o/lora.py:37`). `nn.Module` 속성이 아니라서 base의 parameter가 fake score에 **등록되지 않습니다**. 즉 `fake_score.parameters()`에는 LoRA 가지 154개 텐서만 나옵니다.
- fake score의 forward는 `base(x, σ, c)`, 즉 **G 객체를 그대로 호출**합니다 (`:77`). 그래서 G의 decoder가 한 step 학습되면 그 다음 fake score pass는 바뀐 G 위에서 계산됩니다. fake score는 "현재 G + LoRA 보정"입니다.
- CPU 확인 결과: `fake_score._base[0] is G` → True, fake score parameter와 G parameter의 id 교집합 0개, 세 모듈의 고유 텐서 합 99,992,471 (부록 A).

### 1.4 Discriminator가 Generator의 encoder를 재사용하는 방식

- `EDM2FeatureNetwork(net, shared=True)`가 `object.__setattr__(self, "edm", encoder)`로 G를 붙입니다 (`d2o/discriminator.py:187-190`). submodule로 등록되지 않으므로 판별자 쪽에서는 세어지지도, 저장되지도, optimizer에 들어가지도 않습니다. CPU 확인 결과 `discriminator.feature_network.edm is G` → True입니다.
- 판별자는 `self.edm.features(x, σ, c)`로 G의 embedding과 encoder 부분만 실행합니다 (`:204-211`, `d2o/edm2.py:254-284`). decoder는 실행하지 않습니다.
- encoder가 고정(`--freeze_encoder 1`, `--d_shared 1`)이라 판별자의 "눈"은 학습 내내 teacher E112의 encoder 그대로입니다. 두 옵션은 서로를 요구합니다 (`train_d2o.py:179`).
- 시험만 하고 버린 변형 `--d_shared 2`(`e112_live`)는 encoder까지 학습하면서, 판별자가 매 step의 encoder를 상수로 읽었습니다 (`d2o/edm2.py:323-345`).

### 1.5 diffusion 네트워크에서 one-step 생성기로: forward가 어떻게 달라졌나

| | 원래 diffusion denoiser (E112) | 현재 one-step 생성기 |
|---|---|---|
| 호출 | 여러 σ에서 반복 (Heun 32단계 = 63 NFE) | **σ = 80 한 번** |
| 입력 | `x_σ = x_0 + σ·ε` | `80·z` (순수 노이즈, `model.py:111`) |
| 출력 의미 | E[x_0 \| x_σ]의 추정 (σ=80에서는 흐린 클래스 평균, 학습 전 FID 300.9) | 그 자체가 표본 |
| 학습되는 부분 | 전부 (팀원) | decoder + out_conv만 |
| 같은 가중치의 다른 forward | - | fake score pass: LoRA 켬, eval 모드, σ ∈ [0.002, 80], base로 gradient 없음 |
| preconditioning | EDM (c_skip, c_out, c_in, c_noise = ln σ / 4) | 같음. D2O의 ε-shift(`D2OPrecond`, `d2o/networks.py:418-428`)는 쓰지 않음. 우리 34.8M 줄에만 사용 |

`ModelOneNFE.sample()`의 실제 동작은 다음과 같습니다 (`model.py:97-112`).
1. `latents = randn(shape)`
2. category를 one-hot으로 바꿈
3. `backbone(latents·80, 80, one-hot, force_fp32=True)` **한 번** 실행
4. `clamp(−1, 1)`

CPU에서 hook으로 확인한 결과 backbone 1회, `out_conv` 1회, encoder 모듈 16개와 decoder 모듈 21개가 각각 1회씩 실행됩니다.

---

## 2. Papers & GitHub Provenance — 논문 및 코드 출처

URL 검증 방법: D2O, DMD, DMD2, EDM2, ECT, autoguidance의 arXiv 번호는 로컬 clone의 README에서 확인했습니다. R3GAN, Diffusion2GAN, SiD, guidance interval은 웹 검색으로 확인했습니다. 나머지(EDM, LoRA, ProlificDreamer, Projected GAN, StyleGAN-XL, StyleGAN2-ADA, R1, CFG, CTM, IMLE, clean-fid, relativistic GAN)는 조사자가 알고 있는 번호이며 이번에 따로 검증하지 않았습니다.

### 2.1 출처 표

| 논문 / 방법 | 공식 논문 URL | 공식 GitHub URL | 원 논문의 핵심 방법 | 우리 구현에 실제 사용한 부분 | 수정한 부분 | 관련 코드 |
|---|---|---|---|---|---|---|
| **D2O** (Zheng & Yang, "Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation", ICML 2025) | https://arxiv.org/abs/2506.09376 (원판 https://arxiv.org/abs/2405.20750) | https://github.com/Zyriix/D2O (clone `8857b2d`, 2026-10-07) | 사전학습 diffusion 모델을 σ=80 고정 입력의 one-step 생성기로 바꾸고 **GAN loss만으로** 미세조정. distillation loss 없음. D2O-F(일부 고정). ImageNet 특징 위의 projected 판별자 | σ_g = 80 고정 입력, NS GAN loss, lazy R1(interval 16), Adam β=(0, 0.99), 판별자 lr의 lazy 보정, EMA 규칙(halflife + ramp), 이름 기반 freeze 규칙. 34.8M 줄에서는 네트워크(`SongUNet`, `D2OPrecond`) 포팅 그대로 | 판별자 특징 추출기: ImageNet 네트워크 → diffusion encoder (노이즈 입력). 현재 줄의 생성기는 EDM2(E112). encoder 고정 + 판별자와 공유. DMD2 항 추가. loss를 head 합이 아니라 평균으로 계산 (3.2절). 배치 64, 학습률 다름 | `train_d2o.py:1-8`, `:88-97`, `:204-210`, `:707-728`; `d2o/networks.py:1-14`, `:418-428`; `d2o/training_utils.py:9-51` |
| **Projected GAN / StyleGAN-XL** (D2O의 `pg_modules` 경유) | https://arxiv.org/abs/2111.01007, https://arxiv.org/abs/2202.00273 | https://github.com/autonomousvision/projected-gan, https://github.com/autonomousvision/stylegan-xl | 고정 특징 네트워크 + 랜덤 CCM/CSM projection + 다중 스케일 head, projection conditioning | `DownBlock`, `SingleDiscCond`, `MultiScaleD`, `FeatureFusionBlock`, CCM/CSM 구조 (D2O 코드에서 포팅) | 특징 네트워크를 diffusion encoder로 교체, `width` 배율, head 제거 옵션, EDM2 encoder용 `EDM2FeatureNetwork` | `d2o/discriminator.py:1-17`, `:27-116`, `:175-246` |
| **StyleGAN2-ADA** (D2O의 `training/augment.py` 경유) | https://arxiv.org/abs/2006.06676 | https://github.com/NVlabs/stylegan2-ada-pytorch | 적응형 판별자 증강 | `AugmentPipe`를 plain PyTorch로 포팅, ADA 목표 0.6 | geometric warp를 `grid_sample`로 대체 (custom CUDA op 없이). 현재 줄에서는 증강 확률이 0에 머묾 (max `augment_p` 0.0000, `runs/e112_treal_lr3/log.jsonl`) | `d2o/augment.py:1-8`; `train_d2o.py:212-215`, `:735-738` |
| **EDM** (Karras et al., NeurIPS 2022) | https://arxiv.org/abs/2206.00364 | https://github.com/NVlabs/edm | preconditioning, EDM loss, Heun sampler, DDPM++ | 우리 EDM 사전학습(`train_edm.py`, EDM loss), `SongUNet`/`EDMPrecond`(D2O 경유 포팅), Heun sampler(teacher 표본 생성에 사용) | dnnlib 제거, attention을 autograd op로 (R1 이중 역전파용) | `train_edm.py:1-30`; `d2o/networks.py`; `d2o/sampling.py:22-40` (edm_sampler) |
| **EDM2** (Karras et al., CVPR 2024) | https://arxiv.org/abs/2312.02696 | https://github.com/NVlabs/edm2 (clone `4bf8162`, 2026-10-09) | magnitude-preserving U-Net, forced weight normalization, post-hoc EMA | 네트워크 블록(`normalize`~`Block`, `UNet`)을 이름 그대로 포팅해 E112 checkpoint를 이름으로 읽음 | 팀원 형식에 맞춤: `mean_image` 중심화, null class 152칸, augment embedding을 상수 배율로. encoder만 쓰는 `EDM2Encoder`, `features_fixed`, `encoder_frozen` | `d2o/edm2.py:1-18`, `:25-213`, `:216-359` |
| **E112 teacher** (팀원 학습, EDM2 preset E112) | - | 팀원 저장소 (비공개). 코드 사본: `FastGen-main*.zip`, `/home/jiin/FastGen-teammate/teacher_code_2026-10-13/` | EDM2 공식 코드를 그대로 vendoring해 train split만으로 학습 (`THIRD_PARTY.md`: edm2 `4bf8162`, edm `008a4e5`) | 생성기 초기값(21.0M snapshot), teacher(T58), teacher 표본 | 학습은 팀원이 함. 우리는 읽기만 함 | `checkpoints/edm/*.pt`; `diagnostics/teammate_edm2/` |
| **DMD** (Yin et al., CVPR 2024) | https://arxiv.org/abs/2311.18828 | 공식 코드 공개 여부 `확인 불가` (사용하지 않음) | real/fake score 차이로 생성기 분포를 맞추는 distribution matching + regression loss | DM gradient의 기본 아이디어 (DMD2판으로 사용) | regression loss는 쓰지 않음 | - |
| **DMD2** (Yin et al., "Improved Distribution Matching Distillation for Fast Image Synthesis", NeurIPS 2024) | https://arxiv.org/abs/2405.14867 | https://github.com/tianweiy/DMD2 (clone `8d8fa55`, 2026-10-11) | regression loss 제거, two-time-scale update(fake score 5회 : 생성기 1회), fake score 위 GAN 분류기, backward simulation | `compute_distribution_matching_loss`와 `compute_loss_fake`의 식 (Karras 1000단계 격자, DM은 2~98% 구간, `grad=(p_real−p_fake)/mean|p_real|`, `0.5·MSE(x, sg(x−grad))`, fake loss 가중 `σ⁻²+σ_data⁻²`), fake score 5회 갱신 | **코드를 복사하지 않고 기억으로 재구현**한 뒤 (2026-10-10, `train_d2o.py:48`의 주석) clone이 생긴 2026-10-11에 함수 단위로 대조 (`STATUS.md` 2026-10-11 07:20: "틀린 곳 없음"). fake score = LoRA. GAN은 D2O 판별자. DM gradient를 GAN gradient norm에 맞춤. optimizer Adam(0, 0.99). backward simulation은 one-step에서는 해당 없음 | `train_d2o.py:45-61`, `:217-248`, `:325-362`, `:622-626`, `:648-657`, `:694-704` |
| **LoRA** (Hu et al., ICLR 2022) | https://arxiv.org/abs/2106.09685 | https://github.com/microsoft/LoRA (사용 안 함) | 고정 가중치 옆에 저차원 학습 가지 | down/up 저차원 가지, up 0 초기화 | conv 가지(down은 원 conv와 같은 kernel, up은 1x1). base가 teacher 또는 **생성기 자신**. hook으로 켜고 끔 | `d2o/lora.py` 전체 |
| **ProlificDreamer / VSD** (Wang et al., NeurIPS 2023) | https://arxiv.org/abs/2305.16213 | https://github.com/thu-ml/prolificdreamer (사용 안 함) | fake score(variational score)를 LoRA로 parameterize | 아이디어(선례)만 참고 | - | `d2o/lora.py:1-20` (docstring) |
| **Guidance interval** (Kynkäänniemi et al., NeurIPS 2024) | https://arxiv.org/abs/2404.07724 | - | guidance를 중간 노이즈 구간에만 | teacher(real score)에 0.25 < σ ≤ 4에서 w=1.3 (`_g13`, `_lr3b`, `h20`). 현재 treal 줄은 끔 | 구간과 세기는 우리 teacher에서 sweep해서 고름 | `train_d2o.py:335-338`, `:825-827`; `diagnostics/teammate_edm2/guidance_sweep.py` |
| **Classifier-free guidance** (Ho & Salimans) | https://arxiv.org/abs/2207.12598 | - | cond와 uncond(null) 예측의 외삽 | null class 예측과의 외삽 | 위 구간에만 | 위와 같음 |
| **CTM** (Kim et al., ICLR 2024) | https://arxiv.org/abs/2310.02279 | (사용 안 함) | consistency loss와 GAN loss의 적응형 가중 | **보조 항 gradient를 GAN gradient norm의 ratio배로 맞추는 방식**의 근거로 참고 | 전체 parameter에 대한 norm 비율로 단순화 | `train_d2o.py:630-657` |
| **R1 regularization** (Mescheder et al., 2018) | https://arxiv.org/abs/1801.04406 | - | 진짜 이미지에서의 판별자 gradient penalty | lazy R1 (D2O/StyleGAN2 방식) | γ 0.02, head 평균 logit 기준 | `train_d2o.py:707-715` |
| **clean-fid** (Parmar et al., CVPR 2022) | https://arxiv.org/abs/2104.11222 | https://github.com/GaParmar/clean-fid | FID 계산 | 과제가 제공한 의존성. 학습 중 FID 모니터와 선택 기준에 사용 | evaluate.py의 함수를 train/held-out/teacher 기준 통계에 적용 | `d2o/fid.py`; `evaluate.py:37-47` |
| **ECT** (Geng et al., "Consistency Models Made Easy") | https://arxiv.org/abs/2406.14548 | https://github.com/locuslab/ect (clone `4311059`, 2026-10-10) | 사전학습 diffusion을 consistency 모델로 tuning | `ECMLoss`를 이식 (실험 후 제거) | GAN gradient에 norm 맞춤으로 더함 | `train_d2o.py:23-29`, `:450-500` (기본값 꺼짐) |
| **IMLE** (Li & Malik, 2018) | https://arxiv.org/abs/1809.09087 | - | 실제 이미지마다 가장 가까운 표본을 당김 | coverage 항 (실험 후 제거) | 후보 32장, Hungarian 매칭 | `train_d2o.py:16-21`, `:405-448` |
| **Diffusion2GAN** (Kang et al., ECCV 2024) | https://arxiv.org/abs/2405.05967 | - | teacher의 (노이즈, 이미지) 쌍으로 조건부 GAN 회귀 | paired regression 항 (실험 후 제거) | LPIPS 대신 픽셀 L1 + 고정 encoder 특징 (사전학습 LPIPS 금지) | `train_d2o.py:38-43`, `:376-403` |
| **R3GAN** (Huang et al., NeurIPS 2024) / **Relativistic GAN** (Jolicoeur-Martineau) | https://arxiv.org/abs/2501.05441, https://arxiv.org/abs/1807.00734 | https://github.com/brownvc/R3GAN (사용 안 함) | relativistic pairing loss + R1 + R2 | `--loss_type rp --r2_gamma` (실험 후 미사용) | 논문 코드 없이 조사자 지식으로 구현 | `train_d2o.py:31-36`, `:88-97`, `:314-323`, `:716-726` |
| **SiD** (Zhou et al., ICML 2024) | https://arxiv.org/abs/2404.04057 | - | data-free one-step distillation. 생성기 입력 σ 2.5 제안 | σ_g 2.5 시험의 근거만 (실험 후 폐기) | - | `train_d2o.py:775` (`--sigma_g`) |
| **Autoguidance** (Karras et al., NeurIPS 2024) | https://arxiv.org/abs/2406.02507 | edm2 저장소 안 | 약한 모델 기준 guidance | teacher 탐색에서만 측정 (미사용) | - | `diagnostics/teammate_edm2/autoguidance_sweep.py` |

### 2.2 출처 종류별 분류

1. **원 코드나 공식을 거의 그대로 가져온 부분** (이름과 계산이 같은 포팅)
   - EDM2 네트워크 블록 (`d2o/edm2.py:25-213`). 공개 EDM2 출력과 최대 차이 6e-8~1e-7 (`STATUS.md` 2026-10-09 16:50 검증 기록).
   - D2O/EDM의 `SongUNet`, `EDMPrecond`, `D2OPrecond` (`d2o/networks.py`). 34.8M 줄 전용.
   - D2O `pg_modules`의 판별자 head와 projection (`d2o/discriminator.py:27-116`).
   - D2O `AugmentPipe` (`d2o/augment.py`).
   - D2O 학습 규칙(Adam β, lazy R1과 판별자 lr 보정, EMA, freeze 규칙).
   - EDM loss (`train_edm.py:23-29`), EDM Heun sampler (`d2o/sampling.py`).
2. **원 논문을 참고해 직접 다시 쓴 부분**
   - DMD2의 DM loss와 fake score loss (기억으로 재구현 후 대조).
   - ECT의 `ECMLoss` 이식, R3GAN loss, IMLE, KD. 전부 실험 후 꺼 둠.
3. **여러 논문을 조합한 부분**
   - 현재 목적함수 = D2O GAN + DMD2 DM 항 + CTM식 gradient norm 맞춤 + guidance interval(일부 줄) + LoRA(VSD 선례).
4. **우리 프로젝트에서 독자적으로 만든 부분**
   - 판별자가 **생성기 자신의 고정 encoder**를 특징 추출기로 공유하는 구성 (`--d_shared 1`).
   - fake score를 **생성기 자신 + decoder LoRA**로 만드는 구성 (`--dmd_fake_lora_base generator --dmd_fake_lora_dec 1`).
   - 학습 도중 판별자 head 하나를 빼고 그 자리에 LoRA를 넣은 것 (`--d_drop_heads`).
   - 판별자의 진짜 이미지를 teacher 표본으로 바꾸고, held-out teacher 표본으로 고르는 teacher 중심 선택 (`--extra_real`, `results/reference_teacher58`).
   - held-out train 이미지로 고르는 선택 (`--holdout`, `--balanced`).
   - 사후 출력 보정 (`calibrate_generator.py`: gain, smoothing, offset 등을 가중치에 접어 넣음).
5. **실험했지만 최종적으로 쓰지 않은 부분**
   - IMLE, ECT, teacher 쌍 KD, relativistic loss + R2, σ_g 2.5, 판별자 backbone 학습, `--reuse_fake`, D2O-F, 고정 증강, 배치 128, 판별자 노이즈 −3.5/−4.0, E112 인코더 판별자(tdisc), teacher 표본 섞기(tmix), 우리 EDM fake score(dn30_dmd), 전체 복사 fake score(e112_dmd, 한도 초과), teacher 위 LoRA(e112_lora/lora2), autoguidance, EMA 외삽, smoothing(EDM2), 채도, label 배율, offset.
   - 코드는 대부분 기본값 꺼짐으로 남아 있습니다.

**"논문을 참고했다"와 "논문 코드를 가져왔다"의 구분**:
- 가져온 코드(포팅): D2O 네트워크·판별자·증강, EDM2 블록, EDM sampler와 loss.
- 참고만 한 것: DMD2(식만, 코드 복사 없음), LoRA, VSD, CTM, guidance interval, CFG, R3GAN, IMLE, Diffusion2GAN, SiD.
- 이식한 코드: ECT의 `ECMLoss`(공식 파일을 보고 옮김, 현재 미사용).
- 팀원 코드: teacher 학습은 팀원이 EDM2 공식 코드를 vendoring해서 했고, 우리는 그 결과 checkpoint만 씁니다. 팀원 코드는 실행하지 않았습니다.

---

## 3. Original vs Ours — 원본 논문과 구현 차이

### 3.1 비교표 (D2O, DMD2, 우리 현재 줄 `e112_treal_lr3`)

| 항목 | Original D2O (공식 코드 기준) | Original DMD2 (EDM ImageNet 설정) | Our Implementation | 변경한 이유 |
|---|---|---|---|---|
| Generator | 공개 EDM 사전학습(DDPM++/NCSN++) 전체, `D2OPrecond`(ε=2e-3 shift), σ=80 입력 | teacher 복사본(EDM/ADM), conditioning σ 80 | E112(EDM2, 95.16M)에서 시작, **encoder 고정, decoder만 학습**, EDM preconditioning(shift 없음), σ=80 | 사전학습 모델 금지 → 팀원이 train split으로 학습한 E112 사용. 100M 한도 안에 넣으려고 encoder 고정 + 공유 |
| Discriminator | ImageNet 사전학습 특징 네트워크 2개 + 랜덤 projection + head, 깨끗한 이미지를 224로 보간 | fake score bottleneck 위의 작은 분류 head (`gan_classifier`), diffusion-GAN 노이즈 | **생성기 자신의 고정 encoder**(공유, 노이즈 입력 σ_D~logN(−2,1)) + 랜덤 projection(폭 0.5625) + head 3개 | ImageNet 가중치 금지. 별도 encoder를 들면 100M 초과 (`e112_full` 137.6M) |
| Fake Score | 없음 (GAN만) | teacher의 **전체 복사본**을 denoising loss로 학습 | **생성기 가중치(상수) + decoder conv 77개의 LoRA rank 6** (1.22M만 학습) | 전체 복사본은 framework 합계 195M로 초과 (TA: 같이 학습하면 합산) |
| Real Score | 없음 | 고정 teacher | 고정 teacher T58 (별도 모델), guidance 끔 (이전 줄 1.3, 0.25<σ≤4) | 같음 |
| Teacher 사용 | 초기값으로만 (선택적 instance distillation `teacher_type`이 있으나 기본 꺼짐) | 초기값 + real score | 생성기 초기값은 E112 21.0M, real score와 판별자 "진짜" 표본은 T58 | train 기준 FID가 val을 따라가지 않게 되어, teacher 분포를 닮게 하는 방향 선택 (8절) |
| Loss | NS GAN (head 합), R1 γ 1e-4 (`train.sh`의 모든 설정. 코드 기본값 0.01) | DM loss + (선택) GAN 분류 loss 가중 3e-3, 판별자 쪽 1e-2 | NS GAN (**head 평균**) + DM 항, DM gradient를 **GAN gradient norm에 맞춤**(ratio 1.0), lazy R1 γ 0.02 | GAN이 주, DM이 보조. 두 gradient 크기를 같게 둠 |
| Optimizer | Adam β(0, 0.99), lr 8e-6~1e-4, dlr 4e-5~1e-4 (`/home/jiin/D2O/train.sh`) | AdamW β(0.9, 0.999), lr 2e-6, grad clip 10 | Adam β(0, 0.99): G 3e-4, D 3e-5×16/17, fake score 2e-4 (200 step ramp) | D2O 설정 유지. EDM2 가중치(원소 크기 1)라 lr을 20배쯤 키움 |
| EMA | halflife 0.5 Mimg (`train.sh`의 두 설정은 50 Mimg), warmup 0.05 (`/home/jiin/D2O/train.py:95-96`) | EDM 학습 코드(`/home/jiin/dmd2-official/main/edm/train_edm.py`)에는 EMA 갱신이 없음. `/home/jiin/dmd2-official/main/utils.py:138`의 `EMA` 클래스는 정의만 있고 EDM 경로에서 쓰지 않음 | halflife 500 kimg, ramp 0.05 (D2O 기본값과 같음) | D2O 설정 유지 |
| Training schedule | 수십 Mimg, batch 256~512, 다중 GPU | 수백만 iter, batch 40×7 GPU, fake 5 : G 1 | batch 64 (16×4 누적), 1 G step : 1 D step : **fake score 5 step**, 50 kimg 조각, 누적 5,825 kimg | 20 GB vGPU 한 장 (peak 13.98 GB) |
| Sampling | G 1회 (σ=80) | G 1회 (σ=80) | G 1회 (σ=80) + 가중치에 접어 넣은 gain 1.10 | gain이 공식 FID에서도 효과 (우리 생성기 36.51 → 33.23) |

### 3.2 중요한 변경 상세

**(a) GAN loss의 head 집계.**
- 공식 D2O는 head마다 loss를 구해 **합**합니다 (`/home/jiin/D2O/training/loss.py:239`의 `sum([F.softplus(-glogit).mean() for glogit in g_logits])`, 판별자 쪽 `:273`, `:282`). R1도 head logit의 합으로 계산합니다 (`:303`).
- 우리는 logit을 [N, 75]로 이어 붙인 뒤 **평균**합니다 (`train_d2o.py:89`, `:96`). R1도 `logits.mean(dim=1)` 기준입니다 (`:712`).
- 그래서 같은 γ라도 실효 세기가 공식과 다릅니다. Adam은 gradient 크기에 둔감하지만 R1과 GAN 항의 상대 비율은 달라집니다. 이 차이를 줄인 ablation은 하지 않았습니다.

**(b) DMD2 fake score를 LoRA로 바꾼 것.**
- 기능적으로 같은 부분: fake score loss의 식, 노이즈 격자, 가중, 갱신 횟수(5)는 DMD2 공식 코드와 같습니다. 생성기 표본은 `detach`합니다.
- 기능적으로 다른 부분:
  1. 표현력이 decoder 77개 conv의 rank 6 가지로 제한됩니다.
  2. base가 teacher가 아니라 **지금 생성기**입니다. 생성기는 σ=80에서만 GAN으로 학습되므로 다른 σ에서는 denoiser 능력이 무뎌져 있습니다. 측정 기록상 teacher 대비 denoising loss가 1.5~7.8배였습니다 (`STATUS.md` 2026-10-11 07:40). LoRA가 이 차이를 메워야 합니다.
  3. 생성기가 바뀌면 fake score도 그만큼 같이 바뀝니다 (공유). 그래서 fake score가 생성기 분포를 따라가는 것이 부분적으로 자동입니다.
  4. LoRA 가지에는 EDM2 MPConv의 weight normalization과 gain이 적용되지 않습니다 (`d2o/lora.py:58-60`. 원 conv 출력에 정규화되지 않은 가지 출력을 더함). `out_conv`는 원래 `out_gain`이 곱해지는데 가지 쪽에는 곱해지지 않습니다.
  5. fake score optimizer state는 상태 파일에 저장하지 않습니다. 재개 때마다 lr이 200 step 동안 다시 ramp합니다 (`train_d2o.py:275-276`, `:346-355`).
- 효과 확인 (같은 출발점, DM 75~85 kimg 구간):
  - 전체 복사본 `e112_dmd` 1850: train 기준 13.143 / 공식 23.95
  - teacher 위 LoRA `e112_lora2`: 13.334 / 24.53
  - 생성기 위 LoRA `e112_lorag` 2075: 13.435 / 24.82
  - 375 kimg 뒤 `e112_lorag` 2375의 공식 22.78은 `e112_dmd` 2000의 22.77과 같았습니다.

**(c) DM gradient의 가중.**
- DMD2는 DM loss가 주 loss이고 GAN 분류 loss를 3e-3으로 작게 더합니다.
- 우리는 GAN gradient 전체 norm을 기준으로 DM gradient의 norm을 맞춥니다 (`train_d2o.py:630-657`):
  `g_total = ∇L_GAN + ratio · (‖∇L_GAN‖ / ‖∇L_DM‖) · ∇L_DM`, ratio = 1.0
- 로그의 `dmd_grad_ratio`(맞추기 전 ‖∇L_DM‖/‖∇L_GAN‖)는 현재 줄 `e112_treal_lr3`에서 중앙값 0.088, 5~95% 구간 0.06~0.13입니다 (2 kimg 기록 156개). 그 앞 `e112_treal`에서는 중앙값 0.048, 0.02~0.09였습니다.
- 즉 원래 DM gradient는 GAN gradient의 1/10 안팎인데, 그것을 GAN과 같은 크기로 키워서 씁니다.

**(d) 판별자의 "진짜" 데이터.**
- D2O와 DMD2 모두 판별자는 실제 데이터를 봅니다.
- 현재 줄은 실제 train 이미지 대신 **teacher T58의 표본만** 봅니다 (`--extra_real_p 1.0`).
- teacher는 train split만으로 학습됐으므로 데이터 출처는 train split입니다. 다만 "GAN이 실제 데이터를 본다"는 D2O의 핵심 가정에서는 벗어났습니다.

**(e) teacher 샘플링 시작점.**
- E112 계열은 `x − mean_image`로 학습되어, prior가 `mean_image + 80·ε`입니다.
- 2026-10-13 이전의 teacher FID(11.28 / 22.67 등)는 `80·ε`에서 시작해 표본이 어두웠습니다. 올바른 시작점으로 재면 21.0M teacher가 공식 20.52입니다 (`results/teacher_handoff_eval.log`).
- 학생(one-step)은 학습과 같은 `80·z` 입력이라 영향이 없습니다. DM 항도 노이즈를 더한 학생 표본에서 teacher를 부르므로 영향이 없습니다.

---

## 4. Training Pipeline — 실제 학습 과정

### 4.1 학습 단계 (현재 모델의 lineage)

현재 후보 checkpoint는 아래 단계를 차례로 이어 학습한 결과입니다.
- 각 단계는 이전 단계의 `d2o-state-latest.pt`에서 `--resume`으로 시작합니다.
- 경로와 kimg는 `runs/chain.log`에서 확인했습니다.
- 공통 인자는 `runs/chain_final.sh:32`의 BASE(`--edm runs/edm_c96/edm-005000.pt --freeze 0 --xflip 1 --ada_target 0.6 --lr 5e-5 --dlr 1e-4 --snap_kimg 25 --keep_snapshots 0`)이고, 그 뒤 인자가 덮어씁니다. `--edm`은 `--g_edm2` 경로에서는 쓰이지 않습니다 (`train_d2o.py:137-139`, `:166`).

| 단계 | run | kimg | 바뀐 것 (그 단계에서) | 학습 framework (고유) | 전환 조건 |
|---|---|---|---|---|---|
| P0 | `e112_fit` | 0 → 1950 | E112에서 시작. encoder 고정, 판별자 공유 encoder + 폭 0.5625 + head 4개. lr 1e-3, dlr 1e-4. GAN만 | 99,963,661 | 1500 kimg 뒤 평평 (1950에서 디스크 사고로 멈춤) |
| P1 | `e112_lorag` | 1950 → 2000 | head 1 제거(판별자 optimizer 새로 시작), LoRA fake score 생성. **fake score만 준비 학습, DM 항 꺼짐** (`--dmd_from_kimg 2000`) | 99,992,471 | director: `loss_fake` 마지막 10개 평균 ≤ 0.50 (실제 0.351) |
| P2 | `e112_lorag` | 2000 → 2875 | DM 항 켬 (teacher E112 21.0M, ratio 1.0), fake score 3회/step | 99,992,471 | 2600 이후 평평해서 세션이 닫음 (`closed`) |
| P3 | `e112_lorag_f5` | 2875 → 3425 | fake score 5회/step | 99,992,471 | 마지막 두 지점에 0.02 이상 새 최저 없음 |
| P4 | `e112_lorag_f5_g13` | 3425 → 3675 | teacher guidance 1.3 (0.25<σ≤4) | 99,992,471 | 같은 규칙 |
| P5 | `e112_lorag_f5_g13_lr3b` | 3675 → 5125 | lr 3e-4, dlr 3e-5 | 99,992,471 | 사람 결정: train 기준이 val을 따라가지 않음 → teacher 중심 줄로 |
| P6 | `e112_treal` | 5125 → 5525 | 진짜 이미지 = T58 표본, DM teacher = T58, guidance 끔, lr 1e-3, dlr 1e-4, 선택 기준 = teacher 표본 | 99,992,471 | teacher 기준 개선 < 0.02 |
| P7 | `e112_treal_lr3` | 5525 → 5825 | lr 3e-4, dlr 3e-5 | 99,992,471 | 같은 규칙 (5825에서 멈춤) |

- P7이 끝난 뒤 `e112_treal` 자체도 같은 설정(lr 1e-3)으로 5525 → 5725까지 더 학습됐습니다 (2026-10-22 16:25~21:25 UTC, `runs/chain.log`). 이 연장은 현재 후보의 lineage에 들어가지 않습니다.
- 그래서 지금 `runs/e112_treal/d2o-state-latest.pt`는 P7이 출발한 5525 상태가 아니라 5725 상태입니다. 5525 상태는 남아 있지 않습니다.

- 단계 전환은 `runs/director.py`가 로그를 읽고 규칙대로 큐에 줄을 덧붙였습니다 (`runs/director.py:1-63`, `MARGIN = 0.02` `:74`). 사람이 직접 넣은 항목(`manual`, `close`, `closed`)도 섞여 있습니다.
- 각 단계의 `train.log` 첫 줄들에 "Generator + discriminator + fake score: 99,992,471 parameters"가 찍혀 있습니다 (P1~P7).

### 4.2 한 번의 optimizer step (현재 줄 기준)

`--batch 64 --batch_gpu 16`이므로 한 step은 16장씩 4 round입니다 (`train_d2o.py:115-116`).

```text
data = 4 × (real 16장, label)      ← train 이미지를 꺼낸 뒤 전부 T58 표본으로 교체 (next_real, :296-304)

[1] Generator step (:604-659)
    판별자 parameter requires_grad=False (:605-606)
    round마다:
      x_fake = G(80·z, 80, c_real)                                    (run_G, :281-294)
      L_GAN = mean softplus(−D(x_fake, c))      → backward (retain_graph)   (:618-620)
      L_DM  = dmd_loss(x_fake, c)               → autograd.grad로 따로 모음  (:622-626)
    g = ∇L_GAN + 1.0·(‖∇L_GAN‖/‖∇L_DM‖)·∇L_DM   (전체 학습 parameter 기준 norm, :630-657)
    nan/inf 정리 후 Adam step (:658-659)

[2] Discriminator step (:661-693)
    round마다:
      x_fake' = G(80·z', 80, c)  (no_grad, 새 노이즈)                    (:673-676)
      L_D = mean softplus(D(x_fake')) + mean softplus(−D(x_real))       (:681-683)
      L_F = fake_score_loss(x_fake')  → backward (LoRA만)                (:684-687)
    판별자 Adam step (:693), fake score Adam step 1회 (:697)

[3] Fake score 추가 4 step (:698-704)
    매번 4 round × 새 x_fake'' (no_grad) → L_F → LoRA Adam step

[4] Lazy R1 (16 step마다, :707-728)
    x_real (증강 없음, fp32)에 대해 (γ/2)·16·E‖∇_x mean_k D_k(x_real)‖², γ=0.02 → 판별자 step

[5] EMA 갱신 (:732), ADA 확률 조정 (:735-738), 2 kimg마다 로그, 25 kimg마다 평가와 상태 저장 (:741-761)
```

### 4.3 loss 식 (코드와 대응)

기호: G = 생성기, D = 판별자 logit [N, 75], T = teacher, F = fake score, sg = stop-gradient, σ_g = 80.

- **생성기 GAN loss** (`train_d2o.py:89`, `:618-619`)
  L_GAN(θ) = (1/75N) Σ_{i,k} softplus(−D_k(G_θ(σ_g z_i, σ_g, c_i), c_i))
- **판별자 loss** (`:96`, `:682`)
  L_D(φ) = mean softplus(D(x_fake)) + mean softplus(−D(x_real))
- **판별자 입력** (`d2o/discriminator.py:234-243`, `:204-211`)
  D(x, c) = heads(CSM(CCM(E_G(x + σ_D ε, σ_D, c))), c), ln σ_D ~ N(−2.0, 1.0), σ_D ∈ [0.02, 80]
- **R1** (`:707-715`)
  L_R1 = (γ/2) · r1_interval · E_x ‖∇_x (1/75 Σ_k D_k(x, c))‖², γ = 0.02, interval 16
- **Distribution matching** (`:325-344`)
  - σ = s_j, j ~ U{20, …, 980}, s = Karras 격자 1000단계 (0.002 → 80, ρ = 7, 오름차순, `:243-244`)
  - x_t = sg(x) + σ ε
  - p_real = x − T(x_t, σ, c), p_fake = x − F_ψ(x_t, σ, c) (둘 다 no_grad, float32)
  - g = (p_real − p_fake) / mean_{CHW} |p_real|
  - L_DM(θ) = ½ · MSE(x, sg(x − g)), x = G_θ(...)
  - 그래서 ∂L_DM/∂x ∝ g이고, 경사하강은 x를 T(x_t) 쪽으로, F(x_t)에서 먼 쪽으로 옮깁니다.
- **Fake score loss** (`:357-362`)
  L_F(ψ) = mean[(σ⁻² + σ_data⁻²) · (F_ψ(sg(x) + σε, σ, c) − sg(x))²], σ는 격자 전체에서 균등 (j ~ U{0..999}), σ_data = 0.6348
- **보조 항 결합** (`:630-657`)
  ∇θ_total = ∇θ L_GAN + Σ_aux ratio_aux · (‖∇θ L_GAN‖ / ‖∇θ L_aux‖) · ∇θ L_aux. 현재 aux = DM 하나, ratio 1.0
- **사용하지 않는 추가 loss**: consistency(ECT), IMLE, KD, R2, relativistic loss는 코드에 있지만 현재 줄의 인자로 전부 0이거나 꺼짐입니다 (`runs/e112_treal_lr3/calib-005825.ckpt`의 `args`: `ect_ratio 0.0`, `imle_ratio 0.0`, `kd_ratio 0.0`, `r2_gamma 0.0`, `loss_type ns`).

### 4.4 어떤 parameter가 gradient를 받는가

| 단계 | gradient를 받는 것 | 받지 않는 것 (방법) |
|---|---|---|
| Generator step | G의 decoder + out_conv (65.7M) | G의 encoder와 embedding (`requires_grad=False`, `apply_freeze`), 판별자 전체 (`:605-606`), teacher (`requires_grad_(False)`, no_grad), fake score (no_grad 안에서 실행) |
| Discriminator step | 판별자 head 3개 (3.36M) | CCM/CSM (고정), G (가짜는 no_grad로 생성, encoder는 고정이라 특징 경로에도 parameter gradient 없음) |
| Fake score step | LoRA down/up (1.22M) | G의 모든 가중치 (fake score pass 동안 `requires_grad=False`, eval, `d2o/lora.py:66-74`), 생성기 표본은 `detach` |
| R1 | 판별자 head | 위와 같음 |
| EMA | 없음 (no_grad의 lerp) | - |

판별자 경로에서 encoder parameter는 고정이지만 **입력 이미지 쪽 gradient는 흐릅니다**. 그래서 생성기는 판별자의 판단을 encoder 특징을 거쳐 받습니다.

### 4.5 Update frequency (코드와 인자에서 확인한 값)

| 네트워크 | step당 갱신 | 근거 |
|---|---|---|
| Generator | 1회 | `train_d2o.py:659` |
| Discriminator | 1회 + 16 step마다 R1 1회 | `:693`, `:707` |
| Fake score (LoRA) | **5회** (`--dmd_fake_steps 5`; P2는 3회) | `:697-704`, checkpoint `args.dmd_fake_steps = 5` |
| EMA | 1회 | `:732` |

### 4.6 노이즈 sampling과 schedule

| 용도 | 분포 | 근거 |
|---|---|---|
| 생성기 입력 | z ~ N(0, I), σ_g = 80 고정 | `:284`, `--sigma_g 80` |
| 판별자 입력 노이즈 | ln σ_D ~ N(−2.0, 1.0), [0.02, 80] | `d2o/discriminator.py:234-237`, `--d_sigma_mean -2.0` |
| DM 항 | Karras 격자 1000단계의 2%~98% 구간에서 균등 | `:243-244`, `:330-331` |
| fake score loss | Karras 격자 전체에서 균등 | `:360` |
| R1 | 노이즈 없음 (D 내부에서 σ_D는 그대로 섞임) | `:711` |

### 4.7 Sampling (평가와 제출)

| 항목 | 값 |
|---|---|
| 로드되는 모듈 | `ModelOneNFE` 하나 (backbone = `EDM2Precond`). `evaluate.py`가 인자 없이 `ModelOneNFE()`를 만들면 처음엔 34.8M `D2OPrecond`가 생기지만, `load_state_dict`가 `backbone.unet.out_gain` 키를 보고 EDM2로 바꿔 끼웁니다 (`model.py:115-127`) |
| buffer | `sigma_g` (80), `backbone.mean_image`, `emb_fourier.freqs/phases` |
| forward 횟수 | **1회** (CPU hook 확인: backbone 1, out_conv 1) |
| 판별자, fake score, teacher | 로드하지 않음 |
| 출력 | `clamp(−1, 1)` → `evaluate.py`가 PNG로 저장 (`evaluate.py:78-82`) |

---

## 5. LoRA 및 Parameter Sharing 상세 분석

### 5.1 왜 LoRA인가

- DMD2의 fake score는 teacher 크기의 네트워크를 하나 더 학습합니다. E112로 그렇게 하면 framework 합계가 195,122,370입니다 (`e112_dmd`).
- TA 답(2026-10-11, `STATUS.md` 원문 옮김)은 "generator와 fake score network는 합쳐서 100M 안"이었습니다. 그래서 fake score가 새로 들 수 있는 parameter는 판별자 head 하나를 뺀 자리(약 120만)뿐이었습니다.
- 첫 시도(`e112_lora`, `e112_lora2`)는 LoRA를 **teacher 위에** 붙였습니다. 그런데 TA가 "frozen parameters 포함"을 근거로 든 이상, fake score의 base인 teacher 가중치가 student 쪽에 합산될 위험이 남았습니다.
- 그래서 LoRA를 **생성기 자신 위에** 붙였습니다 (`e112_lorag`). fake score가 쓰는 가중치가 전부 이미 생성기로 세어진 텐서이기 때문입니다.

### 5.2 어디에 붙였나

- 대상: G의 `unet` 아래 모듈 중 이름이 `dec.`로 시작하거나 `out_conv`인 것 가운데, 가중치가 4차원인 `MPConv` (`d2o/lora.py:43-52`).
- encoder는 고정이라 teacher encoder 그대로이므로 decoder만 붙였습니다 (`--dmd_fake_lora_dec 1`).
- 개수: **77개** (첫 두 개 `dec.8x8_in0.conv_res0`, `dec.8x8_in0.conv_res1`, 마지막 두 개 `dec.64x64_block3.conv_skip`, `out_conv`).
  - 종류별 (CPU에서 셈): `conv_res0` 21, `conv_res1` 21, `conv_skip` 16, attention의 1x1 conv `attn_qkv` 9와 `attn_proj` 9, `out_conv` 1.
  - attention의 1x1 conv는 가중치가 4차원이라 포함됩니다.
  - decoder 블록마다 있는 `emb_linear` 21개는 가중치가 2차원이라 **제외**됩니다 (`d2o/lora.py:46`의 `ndim == 4` 조건).
- rank 6, scale 항은 없음 (α/r 같은 배율 없이 `up`의 크기로 결정).
- shape 예: `down` [6, 448, 3, 3], `up` [448, 6, 1, 1]; 1x1 conv에는 `down` [6, 448, 1, 1], `up` [1344, 6, 1, 1].
- 합계 1,223,730.

### 5.3 원래 weight와 결합되는 방식

forward hook입니다 (`d2o/lora.py:54-61`):
```python
def hook(module, inputs, output):
    if not self.active:
        return None                      # 생성기 자신의 pass: 원래 출력 그대로
    x = inputs[0]
    y = conv2d(x, down, padding=k // 2)  # [N, 6, H, W]
    return output + conv2d(y, up)        # 원래 MPConv 출력 + 저차원 보정
```
- 원래 `MPConv`의 출력(가중치 정규화와 gain 적용된 conv)에 정규화 없는 저차원 conv 출력을 더합니다.
- `out_conv`는 원래 `out_gain`이 곱해진 출력에 가지 출력을 더합니다.
- 가중치를 합쳐(merge) 저장하지 않습니다. 제출 checkpoint에는 LoRA가 없습니다.

### 5.4 forward 순서와 상태

**생성기 pass (LoRA 꺼짐)**: `fake.active == False` → hook이 `None`을 반환해 원래 출력 그대로 → 일반 G(80z, 80, c). decoder는 train 모드, encoder는 eval 모드.

**fake score pass** (`LoRAFakeScore.forward`, `d2o/lora.py:63-83`):
1. base(G)의 모듈 중 train 모드인 것을 기억해 전부 `training=False`로 둡니다. 이렇게 해야 forced weight normalization이 아직 backward가 남은 생성기 가중치를 덮어쓰지 않습니다.
2. G의 `requires_grad=True`인 parameter를 기억해 `False`로 둡니다.
3. `active = True`, `G(x_t, σ, c)` 실행 → 77개 hook이 가지 출력을 더합니다.
4. `finally`에서 `active = False`, 모듈 모드와 `requires_grad`를 되돌립니다.

CPU에서 확인한 결과, fake score pass 뒤에도 G는 train, encoder는 eval, decoder parameter는 `requires_grad=True`, encoder는 `False`로 정확히 돌아옵니다.

### 5.5 gradient와 optimizer

- `f_params = list(fake_score.parameters())` = LoRA 154개 텐서뿐입니다 (`train_d2o.py:240`).
- optimizer는 `Adam(f_params, lr 2e-4, β(0, 0.99))`이고 (`:241`), 프로세스마다 처음 200 step 동안 lr을 선형으로 올립니다 (`:352`).
- DM 항에서는 fake score가 no_grad 안에서 계산되므로 LoRA로 gradient가 가지 않습니다 (`:328-343`).
- fake score loss에서는 `x.detach()`이고 G parameter는 잠시 `requires_grad=False`이므로, gradient가 LoRA에만 갑니다.

### 5.6 생성기가 바뀌면 fake score는?

- fake score = (현재 G) + LoRA입니다. G의 decoder가 생성기 step에서 바뀌면 fake score의 base가 같이 바뀝니다.
- 장점: fake score가 생성기 분포를 따라가는 일의 일부가 공짜로 됩니다.
- 위험: 생성기의 σ=80 GAN 학습이 다른 σ의 denoiser 능력을 계속 무디게 하므로 LoRA가 계속 보정해야 합니다.
- 기록된 수치: `loss_fake`는 준비 구간 시작 0.676에서 50 kimg 뒤 0.354로 내려갔습니다. teacher 바탕 LoRA는 0.31~0.34였습니다. 현재 줄의 마지막 50개 평균은 0.29입니다 (`runs/e112_treal_lr3/log.jsonl`).

### 5.7 독립 fake-score network 방식과의 차이

| | 독립 fake score (`e112_dmd`) | teacher 위 LoRA (`e112_lora2`) | **생성기 위 LoRA (현재)** |
|---|---|---|---|
| base | teacher 복사본 (학습) | teacher (고정, 공유) | 생성기 (학습 중, 상수로 읽음) |
| 학습 parameter | 95,158,709 | 1,202,140 | 1,223,730 |
| framework 합계 (고유) | 195,122,370 | 99,970,881 (teacher를 세지 않을 때) | 99,992,471 |
| 규정 | 초과 | 미확정 | 안 (TA 2026-10-13: 같은 텐서는 한 번) |
| 출발점의 fake score 품질 | teacher와 같음 | teacher와 같음 | 생성기 → teacher보다 2~8배 나쁜 denoiser |
| 결과 (같은 출발점) | 13.04 / 22.77 (2000) | 13.33 / 24.53 (2050) | 12.31 / 22.78 (2375), 11.89 / 22.40 (2875) |

### 5.8 진짜 같은 `nn.Parameter`인가, 값 복사인가

**같은 객체 참조입니다.** CPU에서 `train_d2o.py`와 같은 순서로 구성해 확인했습니다 (부록 A).
- `discriminator.feature_network.edm is net` → True. `edm`은 판별자의 named_children에 없습니다.
- `fake_score._base[0] is net` → True.
- 판별자 parameter와 G parameter의 id 교집합 0개. fake score parameter와 G parameter의 id 교집합 0개 (G의 텐서는 어느 쪽에도 등록되지 않음).
- 세 모듈의 고유 텐서 합: id 기준 99,992,471 = data_ptr 기준 99,992,471 (408개 텐서). 겹쳐 세면 290,309,889라는 기록이 `STATUS.md` 2026-10-13 07:21에 있습니다 (G를 세 번 세는 경우).
- teacher와 EMA 생성기는 G와 공유하는 텐서가 0개인 별도 객체입니다.

---

## 6. Parameter Budget — 100M 제한 검증

### 6.1 counting 기준

| 기준 | 정의 | 코드 |
|---|---|---|
| Total registered | 모듈마다 `src.utils.count_parameters(module, trainable_only=False)` (= `sum(p.numel() for p in module.parameters())`, buffer 제외) | `src/utils.py:7-11` |
| Unique | framework의 모든 모듈 parameter를 `id(p)`와 `p.data_ptr()`로 중복 제거한 합 | 부록 A |
| Trainable / Frozen | `apply_freeze` 뒤의 `requires_grad` | `d2o/training_utils.py:27-40` |
| TA 규칙 적용 | ① 같은 텐서는 한 번 ② training phase마다 그 phase의 구성 ③ 공유 가중치도 forward 횟수대로 NFE | `STATUS.md` 2026-10-13 07:21 (TA 원문) |

### 6.2 현재 구조 (P1~P7)

| 항목 | 값 |
|---|---|
| Generator registered | 95,158,709 (trainable 65,733,606 / frozen 29,425,103) |
| Discriminator registered | 3,610,032 (trainable 3,355,908 / frozen 254,124) |
| Fake score registered | 1,223,730 (trainable 전부) |
| **Total registered = Unique** | **99,992,471** (공유 텐서는 어느 모듈에도 두 번 등록되지 않음) |
| Trainable (unique) | 70,313,244 |
| Frozen (unique) | 29,679,227 (G encoder 29,425,103 + 판별자 projection 254,124) |
| 별도 teacher | 95,158,709 (T58, 전부 frozen, 텐서 공유 0) |
| EMA copy | 95,158,709 (텐서 공유 0) |
| Sampling 시 | 95,158,709 (`ModelOneNFE`, `evaluate.py`가 세는 값) |
| 한도까지 여유 | 7,529 |

**99,992,471은 현재 코드와 일치합니다.** 세 군데에서 같은 값입니다: CPU 재계산, 각 단계 `train.log`의 시작 출력(`train_d2o.py:245-248`), `runs/director.py` docstring.

### 6.3 기준별 판정

| 해석 | 계산 | 결과 |
|---|---|---|
| TA 규칙 ①② + teacher는 별도 모델 + EMA는 같은 모델 (팀 기준) | 99,992,471 | **안 (100M 이하)** |
| 위 + EMA 사본도 framework에 산정 | 99,992,471 + 95,158,709 = 195,151,180 | 초과 |
| teacher는 별도 모델 (README: "teacher와 student는 합산하지 않는다") | T58 단독 95,158,709 | 안 |
| 제출 모델 (`evaluate.py`가 실제로 세는 값) | 95,158,709 | 안 |

- EMA 사본: README는 "EMA copies"를 모델 합계에 포함되는 예로 듭니다. 팀은 2026-10-09에 "EMA는 고르는 하나만 남는 같은 모델"로 보기로 했습니다 (사용자 결정, `STATUS.md`).
- TA의 두 답변(2026-10-11, 2026-10-13)은 EMA를 직접 다루지 않았습니다.
- 팀원 teacher 학습도 네트워크 1개 + EMA 사본 2개를 같이 들었습니다 (`STATUS.md` 2026-10-09, 팀원 `config.json`의 `ema_stds`).

### 6.4 학습 phase별 최대 parameter (현재 lineage)

| phase | 구성 | 고유 parameter |
|---|---|---|
| P0 `e112_fit` (0~1950) | G + 판별자(head 4개) | 99,963,661 |
| P1~P7 (1950~5825) | G + 판별자(head 3개) + LoRA | 99,992,471 |
| **최대** | | **99,992,471** |

head 1(1,194,920)을 빼고 LoRA(1,223,730)를 넣었습니다. 두 구성을 다 더하면 약 1억 119만이지만, TA 규칙 ②(phase별 적용)에 따라 문제가 없습니다.

### 6.5 다른 구성들의 parameter (CPU 재계산, 부록 A 출력)

| run | 구성 | 고유 parameter | 판정 (팀 기준) | EMA 포함 |
|---|---|---|---|---|
| `d2o_one_*` | 34.8M 생성기 + 판별자(우리 EDM encoder 사본 + head) | 59,751,203 | 안 | 94,549,958 |
| `d2o_one_dn30_tdisc` | 34.8M + E112 encoder 16x16까지 | 58,065,198 | 안 | 92,863,953 |
| `d2o_one_dn30_tdisc8` | 34.8M + E112 encoder 전체 | 77,258,738 | 안 | 112,057,493 |
| `dn30_dmd` | 34.8M + 판별자 + 우리 EDM fake score | 94,549,958 | 안 | 129,348,713 |
| `e112_full` | E112 전체 학습 + 판별자(E112 encoder 사본) | 137,618,692 | **초과** | 232,777,401 |
| `e112_w56` | 위, 폭 0.5625 | 129,388,764 | **초과** | 224,547,473 |
| `e112_fit` / `e112_live` 등 | G + 판별자 head 4개 | 99,963,661 | 안 | 195,122,370 |
| `e112_dmd` | G + 판별자 + teacher 크기 fake score | 195,122,370 | **초과** | 290,281,079 |
| `e112_lora` | G + 판별자(폭 0.46875) + teacher 위 LoRA rank 4 | 99,923,465 | 미확정 (teacher를 base로 공유) | 195,082,174 |
| `e112_lora2` | G + 판별자(head 3) + teacher 위 LoRA rank 4 | 99,970,881 | 미확정 | 195,129,590 |

`e112_full`, `e112_w56`, `e112_dmd`의 checkpoint는 제출 후보로 다루지 않았고, 이후 줄의 출발점으로도 쓰지 않았습니다.

---

## 7. Experiments & Ablations — 실험 전체 기록

전체 62행 표는 `docs/FASTGEN_EXPERIMENT_INVENTORY.csv`에 있습니다 (UTF-8 BOM, Excel에서 바로 열림). 여기서는 흐름을 볼 수 있게 묶어서 정리합니다.

### 7.0 FID 표기 규칙

| 표기 | 뜻 | 기준 이미지 | 용도 |
|---|---|---|---|
| **train 기준** | 3,020장(클래스당 20장)을 `results/reference_train`(train 17,079장)과 비교 | train split | 34.8M 줄과 E112 줄 대부분의 선택 |
| **H 기준** | 판별자에서 뺀 train 3,020장(`results/reference_holdout_h20`) | train split의 held-out | `e112_h20`의 선택 |
| **teacher 기준** | 판별자에 보이지 않은 T58 표본 15,100장(`results/reference_teacher58`) | teacher 표본 (train-only teacher) | `e112_treal*`의 선택 |
| **공식** | `evaluate.py`, val 3,020장, 3회 평균 | val split | **보고용만.** 선택에 쓰지 않음 |
| 보정 전 | gain 1.0, 학습 로그(`fid_train_ref`, 시드 하나) | | |
| 보정 후 | gain 1.0/1.1/1.2(34.8M 줄은 1.1~1.35 + smoothing) 중 최선, 시드 3개 평균 | | |

- 모두 1-NFE입니다. few-NFE 실험은 이 저장소에 없습니다.
- teacher의 값은 35 NFE(Heun 18) 또는 63 NFE(Heun 32)입니다.
- 공식 − train 기준 차이는 checkpoint마다 10~12.8로 흔들립니다. 그래서 train 기준 1.5 이하의 차이로는 공식 순서를 예측할 수 없었습니다 (`STATUS.md` 2026-10-09 20:38).

### 7.1 주요 실험 표

| Experiment ID | Base checkpoint | 변경한 요소 | 유지한 요소 | Training kimg | Params (고유) | FID (선택 기준, 보정 후 / 공식) | 결과 및 해석 | Log/Checkpoint |
|---|---|---|---|---|---|---|---|---|
| `edm_c96` | 처음부터 | 우리 EDM 사전학습 (DDPM++ 96ch) | - | 0-5000 | 34,798,755 | 35 NFE train 24.5 / 공식 37.8 (4000) | 우리 D2O의 출발점 | `runs/edm_c96/` |
| `d2o_x_*` 8개 | edm-001000 | lr, reuse_fake, D 노이즈, D backbone 학습 | D2O 기본 | 0-100~300 | 59,751,203 | 최선 lr2.5: 39.2 / 52.98 | lr 5e-5 / dlr 1e-4 채택. 나머지 해로움 | `runs/d2o_x_*` |
| `d2o_one_final` | edm-005000 | - (D2O 기본 + 채택 lr) | | 0-1150 | 59,751,203 | 21.7 (gain 1.35) / 33.23 | 처음으로 TA 기준을 넘음 | 삭제됨 |
| `d2o_one_f1` | edm-005000 | D2O-F | | 0-120 | 동일 | 보정 전 50.5 (메인 40.8 @100) | D2O-F 패배 | 삭제됨 |
| `d2o_one_dn30` | final@1000 | 판별자 노이즈 −3.0 | | 1000-2450 | 동일 | 17.373 / 28.29 | 우리 34.8M 최선 | `calib-002450.ckpt` |
| dn25/35/40, ada35, b128, aug20/40 | dn30 분기 | 노이즈, 증강, 배치 | | 150~1150 | 동일 | 보정 뒤 ±0.2 안 (aug20 17.94) | 효과 없음 | 삭제됨 |
| tmix / tdisc / tdisc8 | dn30@2200/2450 | teacher 표본 섞기 / E112 encoder 판별자 | 34.8M 생성기 | 250~500 | 59.8M / 58.1M / 77.3M | 17.56 / 17.42 / 17.67 | 보정 뒤 기준선(17.37)과 같음 | 삭제됨 |
| `dn30_dmd` | dn30@2450 | DMD2 항 (fake = 우리 EDM) | | 2450-2650 | 94,549,958 | 16.473 / 28.20 | train만 좋아지고 공식은 그대로 | `calib-002650.ckpt` |
| `e112_full` | E112 21.0M | E112 전체 + 별도 encoder 판별자 | | 0-250 | 137,618,692 (초과) | 15.68 / 재지 않음 | E112 생성기의 큰 이득 확인 | `best.ckpt` |
| `e112_live` | E112 | encoder도 학습 (`--d_shared 2`) | | 0-150 | 99,963,661 | 16.675 / 29.34 | 75 kimg부터 fit에 짐 | 삭제됨 |
| **`e112_fit`** | E112 | encoder 고정 + 공유 판별자 + 폭 0.5625 | GAN만 | 0-1950 | 99,963,661 | 14.029 / 25.63 | 한도 안 E112, 1500 뒤 평평 | `calib-001950.ckpt`, 상태 |
| imle30 / ect50 / kd / rp / s25 / dn30 | fit@1250~1500 또는 E112 | IMLE, ECT, KD, rp+R2, σ_g 2.5, D 노이즈 | | 50~150 | 99,963,661 | 해로움 또는 ±0.05 | 회귀형 항은 전부 해로움 | 삭제됨 |
| `e112_dmd` | fit@1750 | DMD2, 전체 복사 fake score | | 1750-2024 | 195,122,370 (초과) | 13.039 / 22.77 | DM 항 효과 확인, 제출 불가 | `calib-002000.ckpt` |
| `e112_lora` / `lora2` | fit@1950 | teacher 위 LoRA rank 4 | | 30 / 100 | 99.92M / 99.97M | - / 13.334, 24.53 | lora2는 LoRA로도 효과. 규정 미확정 | 삭제됨 |
| **`e112_lorag`** | fit@1950 | 생성기 위 LoRA rank 6 (decoder), head 1 제거, fake 3회 | | 1950-2875 | 99,992,471 | 11.890 / 22.40 (최저 22.22 @2675) | 한도 안에서 e112_dmd 수준 | 삭제됨 |
| `_f5` | lorag@2875 | fake 5회 | | 2875-3425 | 동일 | 11.603 / 22.46 (21.63 @3325) | train −0.29, 공식 평평 | 삭제됨 |
| `_g13` | f5@3425 | teacher guidance 1.3 | | 3425-3675 | 동일 | 11.552 / 22.08 | | `calib-003575.ckpt` |
| `_t58` | g13@3675 | teacher → T58 (guidance 유지) | | 3675-3975 | 동일 | 11.633 → 12.005 (악화) | 접음 | 삭제됨 |
| `_lr3b` | g13@3675 | lr 0.3배 | | 3675-9825 (실행 중) | 동일 | 11.303 @6825 / 22.79 | train 최저, 공식은 22.5~23.4로 오름 | `calib-006825.ckpt` |
| `e112_h20` | E112 | held-out H 제외 + 균등 표집 + H 기준 선택, 전 구성 처음부터 | | 0-4000 | 99,992,471 | H 16.954 @3150 / 21.57 | | `calib-003150.ckpt` |
| `e112_treal` | lr3b@5125 | 진짜 = T58 표본, teacher T58, guidance 끔, lr 1e-3 | | 5125-5525 (10-22에 5725까지 연장) | 동일 | teacher 10.578 @5525 / 21.47. 연장 구간 최저 10.565 @5625 (공식 미측정: 그때 이미 treal_lr3가 줄 최저라 `runs/ref_score.sh`가 재지 않음) | 공식 23.0 → 21.5대 | `calib-005625.ckpt` |
| **`e112_treal_lr3`** | treal@5525 | lr 0.3배 | | 5525-5825 | 동일 | teacher 10.460 @5825 / 21.74; 21.36 @5675 | 현재 후보 | `calib-005675`, `calib-005825` |

### 7.2 사후 보정 실험 (학습 없음)

| 보정 | 결과 | 사용 여부 |
|---|---|---|
| 출력 gain | 34.8M: train 24.45 → 21.71, 공식 36.51 → 33.23. E112 계열 gain 1.1로 0.3~1.1 | 사용 |
| smoothing | 34.8M: 21.36 → 20.66 (공식에선 안 보임). EDM2: 해로움 | 34.8M만 |
| EMA 외삽 α | gain 위에서는 이득 없음. E112에서는 α=0이 최선 | 미사용 |
| 클래스 embedding 배율, 채도, +0.5 offset | 전부 효과 없음 | 미사용 |
| 샘플링 latent 배율 0.9/1.05, 표본 선별, truncation | 해롭거나 흔들림 안 | 미사용 |

### 7.3 실패한 실험과 원인 판단의 확실성

| 실험 | 관측 | 원인에 대한 기록 (확정 아님) |
|---|---|---|
| IMLE (`e112_imle30`) | 50 kimg에 +1.6, 다양성 감소 | 가까운 짝이 없는 상태로 당겨 흐려짐 (추정) |
| ECT (`e112_ect50`) | 100 kimg 보정 뒤 +1.1, 최적 gain 1.1 → 1.3 (흐려짐) | σ=80에서만 GAN으로 학습된 생성기에 consistency 사슬이 맞지 않음 (추정) |
| KD (`e112_kd`) | +6.0 | one-step이 teacher의 그림을 정확히 못 맞춰 평균 쪽으로 감 (추정). LPIPS를 쓸 수 없음 |
| σ_g 2.5 (`e112_s25`) | +3.3, gain 보정이 듣지 않음 | 입력 노이즈가 출력에 섞이는 c_skip 항 (관측) |
| `e112_lora` | 새로 만든 판별자가 진짜와 가짜를 못 가림 (logit ±0.01) | 1950 kimg 동안 학습된 판별자를 버림 (관측) |
| `_t58` | train 기준이 매 지점 악화 | 원인 `확인 불가` |
| `dn30_dmd` | train만 개선, 공식 그대로, 다양성 감소 | teacher와 다른 구조의 fake score가 만드는 편향 (추정) |
| `d2o_one_dn30_lr2e5` | 결과 없음 | 디스크 가득 참 사고 중 시작 (관측) |
| `_lr3b` 후반 | train 기준은 내려가는데 공식은 올라감 | train 기준 FID의 기준 이미지가 판별자가 학습하는 이미지와 같아서 생기는 적합으로 추정. 클래스 균형은 원인이 아님 (확인: 모든 checkpoint +0.05) |

### 7.4 최고 성능 checkpoint와 정확한 연결

| 항목 | `calib-005825.ckpt` (규칙 후보) | `calib-005675.ckpt` (공식 최저) |
|---|---|---|
| 경로 | `runs/e112_treal_lr3/calib-005825.ckpt` | `runs/e112_treal_lr3/calib-005675.ckpt` |
| md5 | `ec1ca78a06754c685e76db510f74d919` | `2361d9d7c73ba1e56caf899ae11addf1` |
| 수정 시각 (UTC) | 2026-10-22 16:21:52 | 2026-10-22 12:39:53 |
| `cur_nimg` | 5,825,024 | 5,675,008 |
| 보정 | gain 1.10, smoothing 0, offset 0, α 0 | 같음 |
| 생성 명령 | `runs/tref_score.sh` → `runs/ref_score.sh:21-23` (`calibrate_generator.py --state runs/e112_treal_lr3/d2o-state-latest.pt --gains=1.0,1.1,1.2 --smooths=0 --seeds 0,1,2 --fid_reference results/reference_teacher58 --out ...`) | 같음 |
| 저장된 `args` (주요) | `g_edm2 ema-0020971520-0.100.pt`, `freeze_encoder 1`, `d_shared 1`, `d_width 0.5625`, `d_drop_heads [1]`, `lr 3e-4`, `dlr 3e-5`, `batch 64`, `batch_gpu 16`, `dmd_ratio 1.0`, `dmd_teacher teacher_unguided-0058720256-0.130.pt`, `dmd_fake_lora 6`, `dmd_fake_lora_base generator`, `dmd_fake_lora_dec 1`, `dmd_fake_lr 2e-4`, `dmd_fake_steps 5`, `dmd_from_kimg 2000`, `dmd_guidance 1.0`, `extra_real data/cache/teacher58_real`, `extra_real_p 1.0`, `fid_reference results/reference_teacher58`, `d_sigma_mean -2.0`, `r1_gamma 0.02`, `ema_halflife_kimg 500`, `xflip 1`, `ada_target 0.6`, `sigma_g 80` | 같음 (`kimg`만 다름) |
| code version | uncommitted working tree. 2026-10-22 실행 당시의 `train_d2o.py`는 2026-10-15 10:51 수정본 (파일 mtime)으로 추정. 지금 파일과 같음 | 같음 |

---

## 8. Architecture Evolution — 구조가 어떻게 변해 왔는가

| 날짜 (UTC) | 단계 | 이전에서 바꾼 것 | 이유 | 결과 | 결정과 다음 단계 영향 |
|---|---|---|---|---|---|
| 10-07 | 우리 EDM + D2O 재현 | 공개 EDM 대신 train split으로 EDM(34.8M)을 직접 학습. 판별자의 ImageNet 특징 → 우리 EDM encoder | 사전학습 모델 금지 | 1000-kimg EDM에서 300 kimg에 39.2 | lr 5e-5/1e-4, freeze 0, reuse 0 채택 |
| 10-08 | 5000-kimg EDM에서 메인 run | 더 학습된 EDM | teacher가 좋을수록 one-step이 좋음 (57.6 → 24.5 teacher에서 같은 75 kimg 63.2 → 41.0) | 공식 36.51 → gain으로 33.23 | TA 기준 통과. "D2O를 밀고 간다" (사용자) |
| 10-08 | 출력 gain/smoothing | 사후 보정 | 흰 배경이 끝까지 가지 않는 관측 | 공식 −3.3 | 이후 모든 후보에 적용 |
| 10-08 | 판별자 노이즈 −3.0 | σ_D 중앙값 0.135 → 0.050 | 노이즈가 선명함의 차이를 가린다는 가설 | 보정 전 −1.3, 보정 뒤 −0.35 | dn30이 34.8M 주력. 이후 −3.5/−4.0, 증강, 배치는 보정 뒤 효과 없음 |
| 10-09 | 팀원 E112 도착 → 판별자 encoder로 | E112 encoder를 판별자 특징으로 (tdisc, tdisc8), 표본 섞기 (tmix) | 판별자의 "눈"이 병목이라는 가설 | 보정 뒤 전부 17.4~17.7로 같음 | 병목은 생성기 쪽이라고 판단 → E112를 생성기로 |
| 10-09 | E112를 생성기로 (한도 무시 시험) | `e112_full` | 사용자: "한도 신경 쓰지 말고 일단 적용" | 50 kimg에 이미 34.8M 2450 kimg 수준 | 한도 안 구성 탐색 |
| 10-09 | 한도 안 E112 | encoder 고정 + 판별자가 그 encoder 공유 + head 폭 0.5625 (`e112_fit`), encoder 학습 변형(`e112_live`) | 생성기 + 판별자 ≤ 1억 | 150 kimg부터 fit이 live를 앞섬, 1950에서 14.03 / 25.6 | fit이 이후 모든 줄의 출발점. **동시에 바꾼 세 가지(고정, 공유, 폭) 중 무엇이 이득인지는 분리되지 않음** (w56으로 일부만 분리) |
| 10-10 | 보조 항들 | IMLE, ECT, KD, rp, σ_g 2.5 | coverage와 teacher와의 격차 | 전부 해롭거나 효과 없음 | 회귀형 항 포기 → distribution matching으로 |
| 10-10 | DMD2 항 (전체 fake score) | `e112_dmd` | teacher score로 분포를 미는 항은 흐려짐 문제를 겪지 않을 것 | 공식 22.77 (teacher 22.67 수준) | 효과 확인. 한도 초과 (TA 답) |
| 10-11 | 한도 안 fake score | 우리 EDM fake(`dn30_dmd`), teacher 위 LoRA(`e112_lora`, `lora2`), **생성기 위 LoRA**(`e112_lorag`) | TA: fake score는 생성기와 합산, frozen 포함 | lorag가 e112_dmd와 같은 공식 수준에 도달 | lorag 채택. 판별자 head 1 제거 → LoRA 자리 |
| 10-12~13 | DMD2 설정 | fake score 5회, teacher guidance 1.3 | DMD2 공식 설정, teacher의 FID를 낮추는 guidance | train 11.89 → 11.55, 공식 22.1~22.4로 평평 | TA 확인(10-13): 공유 텐서 한 번, phase별 → 잠정 표시 해제 |
| 10-13 | 새 teacher (T58) | DM teacher만 교체 (`_t58`) | train 기준 10.4~10.6으로 더 좋은 teacher | train 기준 악화 | 접음. lr 0.3배(`_lr3b`)로 |
| 10-13~15 | lr 0.3배 | `_lr3b` | | train 11.30 (최저)인데 공식 22.5~23.4로 오름 | **train 기준이 val을 따라가지 않음**을 확인 |
| 10-15 | 선택 기준 변경 두 갈래 | held-out H로 고르는 `e112_h20`(처음부터), teacher 표본을 진짜로 보고 teacher 표본으로 고르는 `e112_treal` | train과 val의 분포 차이 (14.9 대 21.0), teacher가 val에서 train 이미지보다 좋음 (20.3 대 21.0) | h20 21.57, treal 21.47 → treal_lr3 21.36 | 현재 후보. **이 방향을 고른 계기는 보고용 val 숫자였음** (`STATUS.md` 2026-10-15 11:50에 명시) |

주의:
- 여러 요소를 동시에 바꾼 단계가 있습니다: `e112_fit`(고정 + 공유 + 폭), `e112_lorag`(LoRA 위치 + rank + decoder만 + fake 3회 + head 제거), `e112_treal`(진짜 데이터 + teacher + guidance + lr + 선택 기준), `e112_h20`(처음부터 + held-out + 균등 표집 + DM 시점).
- 이 단계들에서는 **어떤 요소가 성능 변화의 원인인지 분리할 수 없습니다.**
- `e112_treal`은 그 직전(`_lr3b`)과 lr도 다릅니다 (3e-4 → 1e-3).

---

## 9. Current Status & Reproduction

### 9.1 현재 상태 (2026-10-25 12:32 UTC)

| 항목 | 내용 |
|---|---|
| Best confirmed (공식 최저) | `runs/e112_treal_lr3/calib-005675.ckpt`, 공식 21.36 (보호 파일: `runs/lowest_official.sh`가 자동 삭제에서 제외) |
| Best rule-compliant 후보 | 줄 안의 규칙(teacher 기준 최저)으로는 `runs/e112_treal_lr3/calib-005825.ckpt` (10.460 / 공식 21.74). 줄 사이의 규칙은 미정 (9.2) |
| 진행 중인 실험 | `e112_lorag_f5_g13_lr3b` 9825 → 9925 kimg (12:32 UTC에 9916 kimg, PID 655771. PID는 조각마다 바뀜). 개선이 멈춘 뒤 GPU를 쉬지 않게 하는 채우기 학습입니다. train 기준 9725에서 11.56, 9825에서 11.540으로 최저(11.303 @6825)보다 나쁨. `runs/director.py`가 50~100 kimg 조각을 계속 덧붙임 (마감 2026-11-06 12:00 UTC, 호출 158/600, `runs/director_state.json`). cron `runs/watchdog.sh`가 10분마다 체인을 감시 |
| 결론을 낼 수 없는 부분 | 21.36 대 21.74 같은 0.4 이하 차이 (측정 흔들림 안). treal 줄의 암기 정도 (미측정). 처음부터 한 번에 학습했을 때의 재현성 |
| 최종 sampling configuration | `ModelOneNFE` (EDM2 E112 구조), σ_g = 80, 1 NFE, gain 1.10 포함, `clamp(−1, 1)` |
| Peak VRAM (학습) | 13.98 GB (`e112_treal*`, `log.jsonl`의 `peak_vram_gb`, `torch.cuda.max_memory_allocated`). 한도 20 GB. 다른 줄: `e112_lorag*` 13.39 GB, `e112_dmd` 15.22 GB, `e112_h20` 13.36 GB, 34.8M 줄 11.8 GB |
| Sampling NFE | 1 |
| 디스크 | 여유 4.6 GB (96 GB 중) |

### 9.2 남은 결정 (사용자)

1. **제출 checkpoint를 어떻게 고를지.** 세 줄이 서로 다른 train 유래 기준으로 골라졌습니다 (train / H / teacher). 어느 기준으로 줄 사이를 비교할지 val 없이 정해야 합니다. 공식 점수가 가장 낮은 5675를 고르면 val로 고르는 것이 됩니다 (README: hyper-parameter selection도 train만).
2. **`checkpoints/one_nfe.ckpt` 만들기.** calib 파일이 이미 `{"state_dict": ModelOneNFE.state_dict(), ...}` 형식이라 복사하면 됩니다.
3. 제출물에는 `model.py`가 import하는 `d2o/config.py`, `d2o/edm2.py`, `d2o/networks.py`가 반드시 들어가야 합니다 (`model.py:7-9`). `checkpoints/edm/`이 딸려 들어가지 않게 해야 합니다.

### 9.3 실행 환경

| 항목 | 값 (2026-10-25 확인) |
|---|---|
| Python | `/home/jiin/.venvs/d2o/bin/python` (PATH의 `python` 없음) |
| 패키지 | torch 2.6.0+cu124, torchvision 0.21.0+cu124, clean-fid 0.1.35, scipy 1.17.1 (**1.18 미만 유지**: clean-fid가 `sqrtm(disp=False)` 사용), numpy 2.5.2, pillow 12.3.0, tqdm 4.70.1, kagglehub 1.0.2 |
| `requirements.txt` | kagglehub, Pillow, torch, torchvision, clean-fid, tqdm (numpy, scipy는 이 패키지들이 설치) |
| GPU | GRID A100D-20C (20,480 MiB vGPU 한 장, 학습 두 개 동시 불가) |
| 데이터 | `data/pokemon-generation-one-22k/PokemonData` → `/home/jiin/datasets/...` symlink. 학습은 `data/cache/train_64x64.pt` 캐시(uint8, 17,079장)를 씀 (`d2o/data.py:16-33`) |
| CPU 계산 | 학습과 같이 돌릴 때 `OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3` |

### 9.4 재현 명령 (저장소 스크립트와 실제 인자에서 옮김)

아래는 실제로 실행된 인자를 단계별로 정리한 것입니다 (인자는 `runs/chain.log`의 "starting D2O" 줄과 대조함). 실제 학습은 이를 50~100 kimg 조각으로 나눠 `--resume`으로 이어 붙였습니다 (`runs/queue_final.txt`, `runs/chain.log`).
**처음부터 아래 순서로 다시 돌려 같은 숫자가 나오는지는 확인하지 않았습니다** (RNG가 조각마다 seed 0으로 다시 시작하므로 bit 단위 재현은 되지 않음).

> **경고: 기존 run 폴더(`runs/e112_*`)를 `--outdir`이나 `--resume`에 쓰지 마세요.**
> - 기존 폴더의 `d2o-state-latest.pt`는 그 단계가 끝난 시점의 상태가 아닙니다. 예: `runs/e112_lorag_f5_g13_lr3b`는 지금 9,900 kimg 근처, `runs/e112_treal`은 5725 (P7이 출발한 5525가 아님).
> - 기존 폴더에 `--resume`으로 학습하면 그 폴더의 상태 파일과 `best.ckpt`를 덮어씁니다. `runs/e112_lorag_f5_g13_lr3b`는 지금도 체인이 쓰고 있습니다.
> - 그래서 아래 명령은 전부 새 폴더 `runs/repro/` 아래에 쓰고, 각 단계는 바로 앞 단계의 `runs/repro/...` 상태에서 이어 가게 고쳐 두었습니다. 학습이 도는 동안에는 GPU가 한 장이라 동시에 돌릴 수 없습니다.

```bash
cd /home/jiin/AIC313-D2O
PY=/home/jiin/.venvs/d2o/bin/python
R=runs/repro
BASE="--edm runs/edm_c96/edm-005000.pt --freeze 0 --xflip 1 --ada_target 0.6 --lr 5e-5 --dlr 1e-4 --snap_kimg 25 --keep_snapshots 0"   # runs/chain_final.sh:32
E112=checkpoints/edm/ema-0020971520-0.100.pt
T58=checkpoints/edm/teacher_unguided-0058720256-0.130.pt
G="--g_edm2 $E112 --freeze_encoder 1 --d_shared 1 --d_width 0.5625 --lr 1e-3 --batch_gpu 16"
LORA="--dmd_ratio 1.0 --dmd_fake_lora 6 --dmd_fake_lora_base generator --dmd_fake_lora_dec 1 --dmd_fake_lr 2e-4 --dmd_from_kimg 2000"

# P0: E112에서 GAN만 (원래 첫 조각은 --snap_kimg 10 --kimg 50)
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_fit $G --kimg 1950

# P1-P2: head 1 제거 + 생성기 위 LoRA, 1950-2000은 fake score만, DM 항은 2000부터 (fake 3회)
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_lorag $G --d_drop_heads 1 --resume $R/e112_fit/d2o-state-latest.pt \
    --dmd_teacher $E112 $LORA --dmd_fake_steps 3 --kimg 2875

# P3: fake score 5회
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_lorag_f5 $G --d_drop_heads 1 --resume $R/e112_lorag/d2o-state-latest.pt \
    --dmd_teacher $E112 $LORA --dmd_fake_steps 5 --kimg 3425

# P4: teacher guidance 1.3 (0.25 < sigma <= 4)
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_lorag_f5_g13 $G --d_drop_heads 1 --resume $R/e112_lorag_f5/d2o-state-latest.pt \
    --dmd_teacher $E112 $LORA --dmd_fake_steps 5 --dmd_guidance 1.3 --dmd_guidance_lo 0.25 --dmd_guidance_hi 4 --kimg 3675

# P5: 학습률 0.3배 (원래 줄은 이 상태로 9,900 kimg 넘게 계속 돌았고, P6은 그 중 5125 상태에서 갈라짐)
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_lorag_f5_g13_lr3b $G --d_drop_heads 1 --resume $R/e112_lorag_f5_g13/d2o-state-latest.pt \
    --dmd_teacher $E112 $LORA --dmd_fake_steps 5 --dmd_guidance 1.3 --dmd_guidance_lo 0.25 --dmd_guidance_hi 4 --lr 3e-4 --dlr 3e-5 --kimg 5125

# teacher 표본 (판별자용 51,340장 + 선택 기준용 15,100장)과 기준 통계.
# 이미 있습니다: data/cache/teacher58_real (17 shard), data/cache/teacher58_ref (5 shard), results/reference_teacher58(_stats.npz).
# 다시 만들 필요가 없으며, runs/treal_setup.py는 director 상태도 바꾸므로 재현 때 실행하지 마세요. 처음부터 만들 때의 명령은 아래와 같습니다.
#   bash runs/gen_teacher58.sh        # Heun 32단계, mean_image 시작, seed 5000(ref)/1000(real). 끝난 shard는 건너뜀
#   PYTHONPATH=. $PY -W ignore runs/treal_setup.py

# P6: teacher 중심
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_treal $G --d_drop_heads 1 --resume $R/e112_lorag_f5_g13_lr3b/d2o-state-latest.pt \
    --dmd_teacher $T58 $LORA --dmd_fake_steps 5 --dmd_guidance 1.0 \
    --extra_real data/cache/teacher58_real --extra_real_p 1.0 --dlr 1e-4 --fid_reference results/reference_teacher58 --kimg 5525

# P7: 학습률 0.3배
$PY -W ignore train_d2o.py $BASE --outdir $R/e112_treal_lr3 $G --d_drop_heads 1 --resume $R/e112_treal/d2o-state-latest.pt \
    --dmd_teacher $T58 $LORA --dmd_fake_steps 5 --dmd_guidance 1.0 \
    --extra_real data/cache/teacher58_real --extra_real_p 1.0 --lr 3e-4 --dlr 3e-5 --fid_reference results/reference_teacher58 --kimg 5825

# 출력 보정 (teacher 기준으로 gain 선택). 원래 줄에서는 runs/tref_score.sh <run> <kimg>가 이것을 불렀습니다.
# 그 스크립트는 results/tref_calibration.log에 줄을 더하고 같은 run의 다른 calib 파일을 지우므로 재현에는 직접 부르는 쪽을 쓰세요.
$PY -W ignore calibrate_generator.py --state $R/e112_treal_lr3/d2o-state-latest.pt --gains=1.0,1.1,1.2 --smooths=0 --seeds 0,1,2 \
    --fid_reference results/reference_teacher58 --workdir $R/calib_fid_samples --out $R/e112_treal_lr3/calib-005825.ckpt

# 공식 평가 (보고용). 기존 후보를 잴 때는 --model_checkpoint runs/e112_treal_lr3/calib-005825.ckpt (읽기만 함)
$PY -W ignore evaluate.py --model_checkpoint $R/e112_treal_lr3/calib-005825.ckpt --evaluate_mode one_nfe \
    --reference_dir ./results/reference --output_dir $R/evaluation

# 제출 조건 점검 (읽기만 함. 판별자는 head 4개로 셈. 아래 주의)
$PY -W ignore check_d2o.py --mode one_nfe --checkpoint runs/e112_treal_lr3/calib-005825.ckpt --d_shared 1 --d_width 0.5625
```

주의:
- 모든 실험 옵션(`--d_shared`, `--d_drop_heads`, `--dmd_*`, `--extra_real`, `--holdout`, `--balanced` 등)은 상태 파일에 저장되지 않습니다. `--resume`할 때마다 다시 적어야 합니다 (`train_d2o.py:765-861`의 help 문구).
- `--d_drop_heads 1`을 빠뜨리면 판별자 모양이 달라 새 판별자로 시작합니다 (`:259-262`).
- `check_d2o.py`는 `--d_drop_heads`와 LoRA를 모르므로 "generator + discriminator"를 head 4개 기준(99,963,661)으로 출력합니다. 실제 학습 framework 합계는 6절의 99,992,471입니다.
- `--resume` 없이 이미 run이 있는 폴더에서 시작하면 거부합니다 (`train_d2o.py:107-110`).

### 9.5 남아 있는 문제와 규정 관련 확인 사항

| # | 내용 | 상태 |
|---|---|---|
| 1 | 우리 코드가 commit되지 않음 → 결과와 code version의 대응을 hash로 남길 수 없음 | 미해결 |
| 2 | EMA 사본 산정 (README 문구 대 팀 기준) | TA 명시 답 없음 |
| 3 | teacher 학습(팀원)의 EMA 사본 2개 | 팀원 쪽 확인 필요. 팀 기준에서는 문제 없음 |
| 4 | 학습 중 FID 모니터에 clean-fid의 Inception을 씀 (선택용, loss에는 안 씀) | TA에 묻지 않음 (`STATUS.md` 2026-10-08 기록) |
| 5 | 판별자의 진짜 이미지를 teacher 표본으로 바꾼 것이 "train split만 사용" 규정에 맞는지 | teacher가 train split만으로 학습됐으므로 맞다고 보지만 TA에게 명시적으로 묻지 않음 |
| 6 | 선택 기준을 바꾼 계기가 보고용 val 숫자였음 | 선택 자체는 train 유래 기준. write-up에 그대로 적을 것 |
| 7 | 줄 사이 선택 규칙, 제출 파일 | 미정 (9.2) |
| 8 | treal 줄의 train 이미지 근접도 | 미측정 (`diagnostics/nearest_train.py`로 잴 수 있음) |
| 9 | 인용 의무 | README: 외부 코드와 논문을 전부 인용하지 않으면 0점. 2절 표 전체가 대상 (특히 EDM2 코드 CC BY-NC-SA 4.0, D2O 코드, DMD2) |
| 10 | GAN loss의 head 평균 대 공식 합 | 영향 미측정 |

---

## 10. Explanation for Teammates — 쉽게 이해하는 요약

각 용어는 "역할 → 실제 작동 방식 → 우리 코드에서의 구현" 순서로 설명합니다.

**1. 지금 우리 모델을 한 문장으로?**
"팀원이 만든 diffusion 모델(E112)의 뒷부분만 고쳐서, 노이즈 한 장을 넣으면 한 번에 포켓몬 그림이 나오게 만든 생성기"입니다. 고칠 때는 GAN 판별자와 teacher 비교 신호를 같이 씁니다.

**2. D2O로 시작했는데 지금은 무엇이 달라졌나?**
- 역할: D2O는 "사전학습된 diffusion 모델 + GAN loss만으로 one-step 생성기"를 만드는 방법입니다.
- 작동: σ=80 노이즈를 한 번 넣어 나온 그림을 판별자가 진짜와 비교하고, 그 gradient로 생성기를 고칩니다.
- 우리 코드: 그 틀(`train_d2o.py`의 생성기 step / 판별자 step / R1 / EMA)은 그대로입니다. 달라진 것은 다섯 가지입니다.
  - (a) 사전학습 모델이 공개 EDM이 아니라 팀원 E112입니다.
  - (b) 판별자의 눈이 ImageNet 네트워크가 아니라 생성기 자신의 고정 encoder입니다.
  - (c) DMD2의 distribution matching 항을 더했습니다.
  - (d) 판별자가 보는 진짜가 teacher 표본입니다.
  - (e) 출력 gain을 사후에 넣습니다.

**3. Fake Score는 왜 필요한가?**
- 역할: "지금 생성기가 만드는 그림들의 분포"를 아는 denoiser입니다.
- 작동: 생성기 그림에 노이즈를 섞은 뒤 teacher(진짜 분포의 denoiser)와 fake score가 각각 어떻게 복원하는지 봅니다. 그 차이가 "생성기 분포를 진짜 분포 쪽으로 옮기는 방향"입니다. fake score가 없으면 이 차이를 계산할 수 없습니다.
- 우리 코드: `dmd_loss` (`train_d2o.py:325-344`)의 `p_real − p_fake`. fake score는 생성기 그림으로만 denoising을 배웁니다 (`fake_score_loss`, `:357-362`).

**4. Fake Score를 LoRA로 구현했다는 것은?**
- 역할: fake score를 큰 네트워크 하나로 새로 두지 않고 작은 보정만 학습하는 것입니다.
- 작동: 생성기 네트워크를 그대로 빌리고 decoder의 conv 77개 옆에 rank 6짜리 작은 conv 두 개(줄였다가 다시 늘림)를 붙입니다. fake score를 계산할 때만 이 가지를 켭니다. 처음에는 0이라 아무 영향이 없습니다.
- 우리 코드: `d2o/lora.py`의 `LoRAFakeScore`. hook(`:54-61`)과 켜고 끄는 `active` 표시(`:75-79`). 학습되는 것은 122만 개뿐입니다.

**5. Generator와 Discriminator가 parameter를 공유한다는 것은?**
- 역할: 판별자가 따로 큰 특징 추출기를 들지 않고, 생성기 안의 encoder를 같이 씁니다.
- 작동: encoder는 학습 내내 고정이라 판별자의 "눈"이 흔들리지 않습니다. 판별자는 노이즈를 섞은 그림을 그 encoder에 통과시켜 특징을 얻고, 그 위의 작은 head만 학습합니다. 복사본이 아니라 같은 메모리의 같은 가중치를 씁니다.
- 우리 코드: `EDM2FeatureNetwork(net, shared=True)` (`d2o/discriminator.py:183-199`), `--freeze_encoder 1 --d_shared 1`.

**6. Teacher와 Fake Score의 역할은 어떻게 다른가?**
- teacher: "진짜 분포"의 기준입니다. 고정되어 있고, 생성기와 가중치를 공유하지 않는 별도 모델입니다 (T58, 95.2M).
- fake score: "지금 생성기 분포"를 따라가는 모델입니다. 계속 학습하며, 생성기 가중치 위에 LoRA를 얹은 것입니다.
- 둘의 차이가 DM 항의 방향입니다. 현재 줄에서는 teacher가 판별자의 진짜 표본도 만듭니다.

**7. GAN loss와 Distribution Matching loss는 각각 무엇을 개선하려는가?**
- GAN loss: 그림 하나하나가 진짜(현재는 teacher 표본)처럼 보이게 합니다. 판별자가 encoder 특징으로 진짜와 가짜를 가르고, 생성기는 판별자를 속이는 쪽으로 갑니다 (`train_d2o.py:618-619`).
- DM loss: 분포 전체를 teacher 쪽으로 옮깁니다. 그림 하나를 특정 목표로 끌어당기는 것이 아니라, teacher와 fake score의 복원 차이만큼 밀어 줍니다. 기록상 GAN만일 때 1500 kimg 뒤 train 기준 14.07 근처에서 멈췄던 것이 DM 항을 켜자 다시 내려갔습니다 (전체 복사 fake score `e112_dmd` 13.04, 한도 안 LoRA 줄 `e112_lorag` 11.89 → `_lr3b` 11.30).
- 우리 코드에서는 두 gradient의 크기를 같게 맞춰서 더합니다 (`:630-657`).

**8. 현재 구조가 100M 안에 들어오는 이유는?**
- 생성기 95,158,709에 더해 새로 드는 것이 판별자 head와 projection 3,610,032, LoRA 1,223,730뿐입니다.
- 판별자의 특징 추출기와 fake score의 본체가 생성기 텐서 자체라 다시 세지 않습니다 (TA: 같은 텐서는 한 번).
- teacher는 별도 모델이라 합산하지 않습니다 (README).
- 합계 99,992,471입니다. EMA 사본을 따로 세면 넘으므로, 팀 기준(EMA는 같은 모델)이 전제입니다.

**9. Sampling에서는 실제로 어떤 연산만 하나?**
- 노이즈 z를 뽑습니다 (연산 무료).
- 80을 곱합니다 (무료).
- EMA 생성기를 **한 번** 실행합니다 (1 NFE).
- [−1, 1]로 자릅니다 (무료).
- 판별자, fake score, teacher, LoRA는 쓰지 않습니다. gain은 이미 가중치 안에 있습니다.

**10. 가장 확실한 장점과 아직 검증되지 않은 가설**
- 확실한 것 (측정으로 확인):
  - E112를 생성기로 쓴 것이 가장 큰 이득입니다 (공식 28.3 → 25.6).
  - DM 항이 GAN만의 정체를 풀었습니다 (25.6 → 22.1~22.4).
  - 생성기 위 LoRA로 한도 안에서 전체 fake score와 같은 수준에 왔습니다.
  - teacher 중심 줄에서 공식이 21.4~21.8로 내려왔습니다.
  - 1 NFE, 한도, VRAM을 코드로 확인했습니다.
- 검증되지 않은 것:
  - teacher 기준 선택이 val과 계속 같이 움직일지 (한 줄, 열몇 지점의 관찰).
  - head 평균 loss가 공식의 합보다 나은지.
  - `e112_fit`의 세 가지 동시 변경 중 무엇이 이득인지.
  - treal 줄이 teacher의 near-copy 경향(0.40%)을 물려받았는지.
  - 처음부터 한 번에 학습해도 같은 결과가 나올지.
  - 더 좋은 teacher를 쓰면 얼마나 내려갈지 (사용자가 새 teacher를 가져올 예정).

---

## 부록 A. Parameter count 재계산 코드와 출력

전체 스크립트는 `/tmp/claude-1001/-home-jiin-AIC313-D2O/0da489a0-f8a6-4fbb-a8af-534b4f176a7d/scratchpad/count_params.py`이고, 출력은 같은 폴더의 `count_params.out`입니다. CPU에서 2026-10-25에 실행했습니다 (세션 scratchpad라 오래 남지 않을 수 있습니다).
핵심 부분은 아래와 같습니다 (`train_d2o.py:137-241`과 같은 순서로 구성).

```python
from d2o.edm2 import EDM2Precond
from d2o.discriminator import EDM2FeatureNetwork, ProjectedDiscriminator
from d2o.lora import LoRAFakeScore
from d2o.training_utils import apply_freeze
from src.utils import count_parameters

net = EDM2Precond.from_checkpoint("checkpoints/edm/ema-0020971520-0.100.pt")
net.encoder_frozen = True
apply_freeze(net, 0, freeze_encoder=True)                                        # train_d2o.py:194
d = ProjectedDiscriminator(None, c_dim=151,
        feature_network=EDM2FeatureNetwork(net, cout=36, shared=True), width=0.5625)  # :180, :183-186
del d.heads.mini_discs["1"]                                                     # --d_drop_heads 1
fake = LoRAFakeScore(net, rank=6, shared=True, decoder_only=True)               # :229

def unique(modules):
    by_id, by_ptr = {}, {}
    for m in modules:
        for p in m.parameters():
            by_id[id(p)] = p; by_ptr[p.data_ptr()] = p
    return sum(p.numel() for p in by_id.values()), sum(p.numel() for p in by_ptr.values())

print(d.feature_network.edm is net, fake._base[0] is net)   # True True
print([count_parameters(m, trainable_only=False) for m in (net, d, fake)])
print(unique([net, d, fake]))
```

출력 (발췌):
```text
discriminator.feature_network.edm is generator: True
'edm' registered as a submodule of the discriminator: False
fake_score._base[0] is generator: True
generator tensors reachable through fake_score.parameters(): 0
generator tensors reachable through discriminator.parameters(): 0
LoRA hooks: 77 convolutions, 77 down / 77 up tensors; first ['dec.8x8_in0.conv_res0', 'dec.8x8_in0.conv_res1'], last ['dec.64x64_block3.conv_skip', 'out_conv']
generator split: embeddings+encoder 29,425,103 (frozen by --freeze_encoder 1), decoder+out 65,733,606 (trainable 65,733,606)
discriminator split: projections ccm+csm 254,124 (frozen, random), heads 3,355,908 (trainable), heads kept ['0', '2', '3']
=== e112_lorag line: generator + discriminator + LoRA fake score
  generator      registered   95,158,709   trainable   65,733,606   frozen   29,425,103
  discriminator  registered    3,610,032   trainable    3,355,908   frozen      254,124
  fake score     registered    1,223,730   trainable    1,223,730   frozen            0
  sum of registered counts             99,992,471
  unique tensors (id / data_ptr)       99,992,471 / 99,992,471  (408 tensors)
  trainable (unique)                   70,313,244   frozen (unique) 29,679,227
teacher ...teacher_unguided-0058720256-0.130.pt: 95,158,709 parameters (frozen); shares tensors with the generator: 0
EMA generator (ModelOneNFE, family edm2): 95,158,709 parameters, shares tensors with generator: 0
runs/e112_treal_lr3/calib-005825.ckpt: family edm2, parameter_summary {'total': 95158709, 'trainable': 95158709, 'frozen': 0}
  sample((2,3,64,64)): forward calls {'backbone': 1, 'out_conv': 1, 'enc_blocks': 16, 'dec_blocks': 21}
```

## 부록 B. 파일 지도

| 경로 | 내용 |
|---|---|
| `model.py` | `Model`(수정 금지 부분), `ModelOneNFE`(현재 제출 대상), `ModelFewNFE`(미학습) |
| `d2o/config.py` | 제출 구조 설정 (`GENERATOR_EDM2_KWARGS`, `SIGMA_G`) |
| `d2o/edm2.py` | EDM2 블록 포팅, `EDM2Encoder`, `EDM2Precond`(생성기 / teacher) |
| `d2o/networks.py` | EDM/D2O SongUNet 포팅 (34.8M 줄) |
| `d2o/discriminator.py` | projected 판별자, `EDM2FeatureNetwork` |
| `d2o/lora.py` | LoRA fake score |
| `d2o/augment.py` | ADA |
| `d2o/training_utils.py` | freeze 규칙, EMA, gradient 정리 |
| `d2o/data.py`, `d2o/sampling.py`, `d2o/fid.py` | 데이터 캐시와 batch, sampler, 학습 중 FID |
| `train_d2o.py` | one-step 학습 전체 |
| `train_edm.py`, `eval_edm.py` | 우리 EDM 사전학습과 teacher FID |
| `calibrate_generator.py` | 사후 gain 보정과 내보내기 |
| `check_d2o.py` | 제출 점검 (parameter, NFE, 로드) |
| `runs/chain_final.sh`, `runs/queue_final.txt`, `runs/director.py`, `runs/watchdog.sh` | 무인 학습 체인, 큐, 자동 결정, 감시 |
| `runs/watch_e112.sh`, `runs/ref_score.sh`, `runs/tref_score.sh`, `runs/h20_score.sh` | 조각마다 보정, 선택 점수, 보고용 공식 측정 |
| `runs/<run>/train.log`, `log.jsonl`, `samples-*.png` | run별 로그와 샘플 grid (write-up 자료) |
| `results/official_fid.log` | 공식 FID 전체 기록 (보고용) |
| `results/e112_calibration.log`, `long_comparison.log`, `h20_calibration.log`, `tref_calibration.log` (+`_refs.log`) | 보정 뒤 선택 점수 |
| `results/teacher_*.log`, `results/nearest_train.log`, `results/duplicates.json`, `results/holdout_h20.json` | teacher 측정, 암기 점검, 중복 분석, held-out 분할 |
| `diagnostics/` | 진단 스크립트와 코드 변경 전 백업 |
| `checkpoints/edm/` | teacher / 초기값 checkpoint (제출물에 넣지 말 것) |
| `STATUS.md` | 작업 일지 (사람이 쓴 기록. 이 문서는 코드와 로그로 대조함) |

## 부록 C. 용어

| 용어 | 뜻 |
|---|---|
| NFE | 학습된 네트워크를 실행한 횟수. one-NFE는 정확히 1회 |
| kimg | 학습에 쓴 이미지 수 / 1000 (batch 64면 1 kimg ≈ 15.6 step) |
| E112 | 팀원이 학습한 EDM2 preset (model_channels 112, 95.2M) |
| T58 | E112를 5,872만 장 학습한 뒤 post-hoc EMA σ_rel 0.13으로 재구성한 teacher |
| calib checkpoint | EMA 생성기에 gain을 접어 넣어 선택 점수가 가장 낮았던 것 (`calib-<kimg>.ckpt`) |
| 줄 (line) | 같은 계열의 run 묶음 (예: `e112_lorag` → `_f5` → `_g13` → `_lr3b`) |
| teacher 기준 / H 기준 / train 기준 | 7.0절 |
| real score / fake score | 진짜 분포 / 생성기 분포의 denoiser (DMD 용어) |
