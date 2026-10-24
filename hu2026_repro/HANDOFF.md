# 인수인계 — 중간고사 재현 (Hu et al., IEEE IoT-J 2026) — 2026-10-23

> 로컬 PC 세션(Claude Code)이 이어받을 때 먼저 읽을 것. 사용자: 한국어, 쉬운 설명, 짧게, 표 선호. 수치는 실제 실행 결과만.

## 과제
선택 논문 재현 → 영상 + 근거 자료 제출. 범위: **G1020 목표, 4조건(pooled / raw FedAvg / GAN / GAN+융합)**, seed 42 우선(시간 되면 123, 2026), + 과제1 §7 아이디어(vCDR 오차·의뢰 판정 뒤집힘). 저널 아이디어는 넣지 않음.

## 폴더 (~/Projects/고급네트워크)
| 경로 | 내용 |
|---|---|
| `FundusGAN/` | 저자 공식 코드 (commit 71e9354), `official/patch_official.py`로 패치 6개 적용 완료 |
| `hu2026_repro/` | 우리 코드. 재현은 `official/run_official.py` |
| `raw/` | Kaggle 데이터 (G1020·ORIGA·REFUGE·PAPILA). `raw/models/*.pkl`은 출처 불명 → 열지 말 것 |
| `data/` | `prepare_data.py` 출력 (현재 `--roi paper`로 변환됨 → 다시 변환 필요) |
| `venv/` | Python venv, torch 2.11+cu128, RTX 5080 인식 확인 |

## 확인된 데이터 형식
- G1020 `Images/*.jpg` + `Masks/*.png`(0 배경, 1 림, 2 컵), 1,020장
- ORIGA `Images/` + `Masks/`(0/1/2), 650장
- REFUGE `{train,val,test}/Images` + `/Masks`(0/1/2), 400×3. `gts/`(0 컵,128 림,255 배경)와 일치 확인
- PAPILA `FundusImages/*.jpg` + `ExpertsSegmentations/Contours/<id>_{disc,cup}_exp1.txt`(x y), 488장. `... 2.txt` 중복본 무시
- 마스크 1 = 림, 2 = 컵 확인 (컵 주변이 전부 1)

## 진행 상황
1. 데이터 변환(`--roi paper`) 완료: 실패 0. 그러나 디스크 놓침(<90% 포함) G1020 155, ORIGA 628/650, PAPILA 139, REFUGE 10
2. `roi_check.py` (표본 40장): paper 방식 ORIGA 100% 놓침(최대 밝기 동점 95% → argmax가 왼쪽 위 선택), G1020 15%, PAPILA 30%; smooth 방식 ORIGA 5%, G1020 12.5%, PAPILA 10%, REFUGE 0%
   - G1020 실패 예: 큰 밝은 병변/반사 영역(image_2898)
3. **다음**: 최신 `roi_check.py`(blob 방식 포함)로 비교 → 방식 확정 → 4개 데이터 다시 변환(`--roi <방식>`)
   - blob도 G1020에서 실패가 남으면: GT 디스크 중심 크롭(oracle)을 대안으로 사용자와 결정, 보고서에 명시
4. 그다음: `run_official.py --repo FundusGAN --data data --target G1020 --dry-run` → `--stage all --seeds 42`
5. `make_evidence.py`(best/last 각각), `env_snapshot.sh`

## 공식 코드에서 확인한 것 (보고서용, `official/README.md` 표 참고)
GAN 실행 불가 버그 3개(패치함), 융합 코드 없음, 분할 모델 U-Net 497,669,699 파라미터(2 GB, 라운드당 클라이언트당 왕복 4 GB), 테스트셋으로 best 모델 선택, F1/IoU 배경 포함 macro, 기본값이 논문과 다름, 전처리 코드 없음 + 논문 전처리 설명대로면 ORIGA 실패.

## 주의
- 0.5B 모델: 16 GB GPU 빠듯(추정), RAM 32 GB 권장. 첫 라운드에서 nvidia-smi 확인
- 합성 데이터 테스트 수치는 결과로 쓰지 말 것
- 줄인 설정(seed 수, GAN 에폭 등)은 전부 보고서 차이표에 기록
