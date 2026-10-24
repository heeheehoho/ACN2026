# Hu et al. (IEEE IoT-J 2026) 재현 + 확장 실험 코드

> **논문 재현은 `official/`부터.** 저자 공식 코드(github.com/JRHU0048/FundusGAN)를 최소 패치해 GAN → 융합 → FedAvg를 그대로 돌린다. 공식 코드에서 확인한 문제 12개도 `official/README.md`에 정리.
> 이 폴더의 `fed_train.py`는 독립 재구현(U-Net 소형, FDA로 GAN 대체)으로, 비교·빠른 파일럿용이다.

GPU PC(RTX 5080)에서 돌리는 용도. 클라우드 세션에서는 합성 데이터로 코드가 끝까지 도는지만 확인했다.

## 파일
| 파일 | 역할 |
|---|---|
| `common.py` | 논문 전처리(최고 밝기 ROI → 1000px 크롭 → 512 → CLAHE → median 3×3), 지표(Dice/IoU/vCDR/의뢰 판정 뒤집힘) |
| `prepare_data.py` | 원본 데이터셋 → 표준 형식(`images/`, `masks/` 0 배경·1 림·2 컵) 변환, 실패·ROI 빗나감 기록 |
| `fed_train.py` | leave-one-domain-out: `pooled`(중앙집중) / `local`(사이트 단독) / `fedavg`, `--style fda` 또는 생성 이미지 폴더, `--fusion`(식 5) |
| `privacy_attack.py` | 융합 이미지 역산(A) + 혈관 패턴 재식별(B) |
| `make_evidence.py` | 실행 결과 → 논문 대비 결과표, 학습 곡선, vCDR 산점도, 윤곽선 그림, SHA-256 목록 (제출 근거 자료) |
| `env_snapshot.sh` | GPU·버전·데이터 수 기록 |
| `sim_vcdr.py` | 합성 타원으로 "Dice가 같아도 vCDR 오차는 다를 수 있다" 확인 |

## 설치
```bash
pip install torch numpy scipy scikit-image
```
U-Mamba를 쓰려면 `mamba-ssm`, `causal-conv1d` CUDA 빌드가 필요하다. RTX 5080(Blackwell)은 미리 빌드된 wheel이 없을 수 있다 → 우선 기본 U-Net으로 진행하고, 논문 수치와 직접 비교는 U-Mamba를 붙인 뒤에 한다.

## 0단계 — 데이터 준비 (마스크 형식 반드시 확인)
```bash
python prepare_data.py --kind refuge   --name REFUGE --images raw/REFUGE/images --masks raw/REFUGE/masks --out data
python prepare_data.py --kind labelmap --map 0,1,2 --name G1020 --images raw/G1020/Images --masks raw/G1020/Masks --out data
python prepare_data.py --kind labelmap --map 0,1,2 --name ORIGA --images raw/ORIGA/Images --masks raw/ORIGA/Masks --out data
python prepare_data.py --kind papila   --name PAPILA --images raw/PAPILA/FundusImages --masks raw/PAPILA/ExpertsSegmentations/Contours --expert exp1 --out data
```
- 각 `data/<DS>/prepare_log.json`에서 `failed`, `roi_misses_disc` 개수를 먼저 본다. 논문의 최고 밝기 ROI가 디스크를 놓치는 비율 자체가 결과다.
- `--roi smooth --crop 0.45`(블러 후 최댓값, 해상도 비례 크롭)는 논문 전처리의 개선판. 비교용.
- G1020은 컵 주석이 없는 이미지가 있을 수 있다(확인 필요). 지표 계산에서 GT가 빈 경우는 NaN으로 빠지고 `gt_cup_empty`로 따로 센다.

## 1단계 — 사전점검
```bash
python fed_train.py --data data --target G1020 --mode fedavg --dry-run
```
이미지당 시간과 전체 예상 시간이 나온다. 예상 시간이 길면 `--size 256`으로 파일럿.

## 2단계 — 재현 (G1020 목표부터)
```bash
for s in 42 123 2026; do
  python fed_train.py --data data --target G1020 --mode pooled                    --seed $s --out runs/G1020/pooled_s$s
  python fed_train.py --data data --target G1020 --mode local                     --seed $s --out runs/G1020/local_s$s
  python fed_train.py --data data --target G1020 --mode fedavg                    --seed $s --out runs/G1020/fedavg_s$s
  python fed_train.py --data data --target G1020 --mode fedavg --style fda        --seed $s --out runs/G1020/fda_s$s
  python fed_train.py --data data --target G1020 --mode fedavg --style fda --fusion --seed $s --out runs/G1020/fda_fus_s$s --save-preds
done
```
- 논문 GAN(FundusGAN) 생성 이미지가 있으면 `--style gen/G1020`(폴더 구조 `gen/G1020/<DS>/<id>.png`)로 같은 실험.
- 기본 `--target-split disjoint`: 목표 데이터 절반은 스타일 참조(라벨 안 씀), 나머지 절반만 테스트. 논문처럼 전부 쓰려면 `--target-split paper` (전이적 평가, 비교용).
- 결과: `summary.json`(평균·SD·유효 n), `preds.csv`(이미지별 Dice·vCDR·의뢰 뒤집힘), `ckpt.pt`(라운드마다, 자동 재개).

## 3단계 — 프라이버시 공격
```bash
python privacy_attack.py --data data --site REFUGE --style fda --style-site G1020 --n 200 --out runs/privacy/REFUGE
```
`psnr_inverted`/`ssim_inverted`(역산 복원 품질, 생성본을 가진 공격자), `reid_top1_fused`(융합 이미지로 원본 찾기 top-1), `reid_chance`(우연 수준)를 본다.

## 4단계 — 하이퍼파라미터
α0(0.1~0.9), σ(0.01~0.30)는 **소스 검증셋**으로 고른다. 목표 테스트셋으로 고르면 논문과 같은 문제(테스트 튜닝)가 생긴다.

## 주의
- 합성 데이터 스모크 테스트 결과는 코드 검증용이다. 수치를 결과로 쓰지 않는다.
- 데이터셋 라이선스·재배포 조건은 각 공식 배포처에서 확인한다.

## 5단계 — 제출 근거 자료
```bash
bash env_snapshot.sh data /path/to/FundusGAN evidence/env.txt
python make_evidence.py --runs runs/G1020 --data data --target G1020 --out evidence
python fed_train.py --data data --target G1020 --mode fedavg --style fda --fusion --seed 42 --out runs/G1020/fda_fus_s42 --eval-only   # 영상용 라이브 재평가
```
`--eval-only`는 저장된 `ckpt.pt`(fedavg 모드만 저장)로 같은 테스트 분할을 다시 평가한다. 학습 때와 같은 `--seed`, `--size`, `--width`, `--target-split`을 줘야 같은 분할·모델이 된다.
