# 공식 코드로 논문 재현하기 (github.com/JRHU0048/FundusGAN @ 71e9354)

논문 저자 코드를 **최소한만 고쳐서** 그대로 돌리는 파이프라인. GAN → 융합(식 5) → FedAvg U-Net → 평가.
`../fed_train.py`(독립 재구현, FDA 대체)는 비교·보조용.

## 1. 공식 코드에서 확인한 것 (재현 보고서에 그대로 쓸 수 있음)

| # | 내용 | 확인 방법 |
|---|---|---|
| 1 | **GAN이 그대로는 실행 안 됨**: 기본 `--model cut` → `models/cut_model.py` 없음 (ModuleNotFoundError) | 실행 |
| 2 | `--model ppfg` → 클래스 이름 불일치로 종료. **종료 코드 0**이라 자동 실행에선 성공처럼 보임 | 실행 |
| 3 | 옵션은 `--CUT_model`로 선언, 코드는 `opt.CUT_mode`를 읽음 → AttributeError | 실행 |
| 4 | **융합(식 5) 코드가 레포에 없음** (α, Gaussian 섭동 관련 코드 0건) | 전체 검색 |
| 5 | 분할 모델은 U-Mamba가 아니라 **6단 U-Net, 4096채널, 497,669,699 파라미터(fp32 약 2.0 GB)**. 입력 최소 128px | 실행 (파라미터 수 계산) |
| 6 | **서버가 목표(=테스트) 데이터로 매 라운드 평가하고, 가장 좋은 라운드만 저장** → 테스트셋으로 모델 선택 | 코드 읽기 |
| 7 | Dice = 림(1)·컵(2)을 따로 one-hot으로 본 평균 (디스크 전체 아님). F1·IoU·정밀도·재현율은 **배경 포함 3클래스 macro** → 표 II에서 F1 ≠ Dice인 이유 | 코드 읽기 |
| 8 | MGDA: 가중치 w ≥ 0 제약 없음(논문은 w ≥ 0). MGDA 기울기를 직접 넣은 뒤 가중합 손실로 한 번 더 backward | 코드 읽기 (실행 영향은 측정 안 함) |
| 9 | 기본값이 논문과 다름: GAN 10+10에폭(논문 50+50), FL 10라운드×10에폭(논문 20×5), lr 1e-5, batch 1 + 누적 4, 이미지 0.5배(512→256) | 코드 읽기 |
| 10 | `--federated`, `--kfold`, `--amp`가 `store_true` + 기본 True → 명령줄로 끌 수 없음 | 코드 읽기 |
| 11 | requirements에 scipy, opencv-python, openpyxl 누락. 전처리(ROI·CLAHE·median) 코드 없음 | 실행 |
| 12 | GAN은 5에폭마다만 저장 → 총 에폭이 5의 배수가 아니면 마지막 모델 유실 | 실행 |

**통신량 (과제 §3 보완)**: 라운드당 클라이언트 1곳이 약 2.0 GB를 올리고 2.0 GB를 받는다. 20라운드 × 3곳 → 왕복 합계 약 240 GB.

## 2. 우리가 고친 것 (`patch_official.py`, 재실행해도 안전)
1~3번 실행 오류 수정. seed를 환경변수로 받게 함(42/123/2026). 실행마다 체크포인트 이름 분리.
**마지막 라운드 모델도 저장** → "테스트로 고른 best"와 "공정한 last"를 둘 다 보고.
클라이언트 가중치(각 2 GB)를 집계 전에 CPU로 옮김 → 16 GB GPU에서 메모리 부족 방지. 결과는 동일(패치 전후 라운드별 Dice 일치 확인).
방법 자체(손실, MGDA, FedAvg, 지표)는 **건드리지 않음**.

## 2-1. 메모리 (0.5B 파라미터 모델)
- GPU: 학습 중 모델 + 기울기 + Adam ≈ 8 GB, 전역 모델 2 GB, 활성값 추가 → **16 GB에서 빠듯함 (추정, 실측 필요)**. 첫 라운드에서 `nvidia-smi`로 확인
- RAM: 집계 때 클라이언트 3개 가중치 + 평균 ≈ 8 GB → **32 GB 권장**. 이 세션(15 GB)에선 원본 크기 모델이 메모리 부족으로 종료됨(-9)
- 그래도 안 되면 `--scale 0.5`(기본, 256px) 유지, 다른 프로그램 종료. 모델 축소는 재현이 아니게 되므로 마지막 수단이며 보고서에 명시

## 3. 실행 (RTX 5080)
```bash
# 0) 코드
git clone https://github.com/JRHU0048/FundusGAN fundusgan
python official/patch_official.py fundusgan
# RTX 5080(Blackwell)은 PyTorch 2.4.1(cu121)이 안 됨 → 최신 PyTorch(CUDA 12.8 이상) 설치 후:
pip install torchvision torchmetrics pandas openpyxl scikit-learn tqdm dominate opencv-python scipy scikit-image

# 1) 데이터 → 표준 형식 (논문 전처리)   ※ 데이터 출처는 공식 README의 Kaggle 링크
python prepare_data.py --kind ... (README 상위 폴더 참고)

# 2) 명령만 먼저 확인 (사전점검)
python official/run_official.py --repo fundusgan --data data --target G1020 --dry-run

# 3) 전체 실행 (단계별로 끊어도 됨: --stage prep | gan | fuse | fl | eval)
python official/run_official.py --repo fundusgan --data data --target G1020 --stage all

# 4) 근거 자료
python make_evidence.py --runs runs_official/last --data data --target G1020 --out evidence_last
python make_evidence.py --runs runs_official/best --data data --target G1020 --out evidence_best
```

### 조건 4개 (`--conds`)
| 이름 | 내용 | 논문 표 II |
|---|---|---|
| `pooled` | 소스 3곳 합쳐 한 클라이언트 (= 중앙 학습) | Source only |
| `raw` | FedAvg, 원본 이미지 | 없음 (**논문에 빠진 비교군**) |
| `gen` | FedAvg, GAN 변환 이미지 | Ours ✗ |
| `fused` | FedAvg, 원본 30% + 변환 70% (α 이미지당 1개, `alphas.csv`에 기록) | Ours ✓ |

### 시간이 부족하면
- `--seeds 42` (seed 1개), `--gan-epochs 10,10` (레포 기본값), `--rounds 10 --local-epochs 5`
- 모델이 0.5B 파라미터라 무겁다. 첫 라운드 시간을 보고 전체 시간을 추정할 것 (`runs_official/logs/fl_*.log`)
- 줄인 설정은 전부 보고서 차이표에 적는다

## 4. 우리 설계 선택 (보고서에 명시)
- 목표 데이터 반은 GAN 스타일용(라벨 미사용), 반은 테스트 (`split.json`, 고정 seed 0). 논문은 분할 방식 불명
- GAN 학습 이미지: 소스 전부(trainA) + 목표 적응용 절반(trainB)
- 평가: 공식 방식 Dice/F1과, 표준 OD(디스크 전체)·OC Dice + vCDR을 **둘 다** 출력

## 5. 이 세션에서 확인한 범위 (성능 수치는 의미 없음)
합성 이미지(128px)로 전체 단계(분할 → GAN 학습·변환 → 융합 → 4조건 FL → best/last 평가 → 근거 자료)가 끝까지 도는 것을 확인했다.
- **평가 스크립트 검증**: `eval_official.py`의 "official Dice"(0.0321)와 macro-F1(0.0240)이 공식 코드가 같은 라운드에 직접 기록한 값(0.032105, 0.024020)과 일치
- CPU 테스트를 위해 **테스트 사본에서만** 바꾼 것 3가지 (실제 실행엔 적용 안 함):
  ① MobileNet 사전학습 가중치 끔 (네트워크 차단)
  ② CPU에서만 생기는 저장 버그 수정 (`module.` 접두사, 원본 CUT 코드 문제. GPU에선 발생 안 함)
  ③ U-Net 채널 1/16 축소 (원본 크기는 15 GB RAM에서 메모리 부족)
