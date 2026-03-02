# Cerebral Microbleeds (CMB) Detection

이 프로젝트는 뇌 MRI 이미지에서 미세출혈(Cerebral Microbleeds, CMB)을 탐지하는 딥러닝 모델 모듈입니다. 데이터 전처리, 모델 훈련(K-Fold 및 Fixed Split 지원), 추론 및 평가 파이프라인으로 구성되어 있습니다.

## 🚀 파이프라인 실행 가이드

기본적으로 여러 파이프라인(학습, 평가, 고정 분할, K-Fold 등)은 제공된 `scripts/run.sh` 스크립트를 통해 원클릭으로 쉽게 실행하는 것을 권장합니다.

```bash
# 기본 사용법
./scripts/run.sh [MODE] [MODEL] [RUN_NAME]

# 예시: SSD_FE 모델을 test라는 이름으로 학습 및 평가(Fixed+K-fold 모두)
./scripts/train_eval/both.sh SSD_FE test
```

만약 직접 파이썬 스크립트를 제어하고 싶다면 아래 가이드를 참고하세요.

### 1. 모델 훈련 (`train.py`)
모델 학습과 관련된 모든 과정을 제어합니다. 인자(옵션)를 통해서 전처리부터 세밀한 Fold 훈련까지 조정할 수 있습니다.

```bash
python train.py --model SSD_FE --run_name default [options]
```

#### 📌 주요 인자 (Arguments)
| 인자 (Args) | 타입 | 설명 |
|---|---|---|
| `--model` | String | **(필수)** 학습에 사용할 모델 아키텍처를 지정합니다. (예: `SSD_FE`) |
| `--run_name` | String | **(필수)** 현재 학습을 식별할 식별자/이름입니다. 가중치 저장 및 체크포인트 로드 시 기준이 됩니다. (기본값: `default`) |
| `--prepare_data` | Flag | 학습 시작 전, NIfTI 이미지 전처리(Skull Stripping, N4 bias correction, PNG 변환) 및 LMDB 데이터베이스를 새로 구성할 때 사용합니다. (최초 1회 필수 권장) |
| `--fixed_split` | Flag | (선택적) 모델을 훈련할 때 설정(config)의 K-Fold 옵션을 덮어쓰고, 강제로 고정된 분할(Fixed Split) 데이터셋으로만 학습을 수행합니다. |
| `--folds` | Integers | (선택적) 특정 Fold 번호만 콕 집어서 학습할 때 사용합니다. (예: `--folds 0 1 2`). 지정하지 않으면 `config.py`에 정의된 개수만큼 전체 K-Fold를 순차적으로 돌립니다. |

- **가중치 이어하기(Resume) 자동화**: 학습이 중단되더라도 `train.py` 실행 시 동일한 `--model`과 `--run_name`을 입력하면 가장 마지막 체크포인트를 자동으로 찾아 이어서 학습합니다.

#### 💡 사용 예시
```bash
# 1. 처음 데이터를 전처리하고 SSD_FE 모델로 학습 시작하기
python train.py --model SSD_FE --run_name my_first_run --prepare_data

# 2. 전처리 없이 0번, 2번 Fold만 특정 이름으로 학습하기 
python train.py --model SSD_FE --run_name run_kfold_only --folds 0 2

# 3. K-Fold 대신 고정 분할 모드로만 훈련하기
python train.py --model SSD_FE --run_name run_fixed --fixed_split
```

---

### 2. 모델 평가 (`evaluate.py`)
학습이 끝난 후 모델 아키텍처와 학습 이름(런 네임)을 지정해주면, `evaluate.py`가 설정된 분할 설정(K-Fold 혹은 Fixed)에 따라 자동으로 가중치와 대상 LMDB를 매핑하여 mAP, FROC, 오차 행렬(Confusion Matrix) 계산 및 결과 시각화를수행합니다.

```bash
python evaluate.py --model SSD_FE --run_name default [options]
```

#### 📌 주요 인자 (Arguments)
| 인자 (Args) | 타입 | 설명 |
|---|---|---|
| `--model` | String | **(필수)** 평가할 타겟 모델의 아키텍처입니다. (예: `SSD_FE`) |
| `--run_name` | String | **(필수)** 훈련 시 사용했던 식별자 이름입니다. 이를 바탕으로 가중치 폴더를 역추적합니다. (기본값: `default`)|
| `--fixed_split` | Flag | (선택적) K-Fold 대신 Fixed Split으로 학습된 모델을 평가할 때 붙입니다. |
| `--folds` | Integers | (선택적) K-Fold 중 특정 Fold 결과만 평가하고 싶을 때 사용합니다. (예: `--folds 0 1 2`). 지정하지 않으면 10개 폴드를 모두 연속 스캔하여 종합 평균을 냅니다. |
| `--lmdb` | String | (선택적) 평가에 사용할 Test LMDB 경로를 수동으로 강제 오버라이드합니다. |
| `--patient` | String | (선택적) 전체 Test 셋을 보지 않고, 특정 환자(예: `VK049`) 데이터만 필터링하여 단독으로 평가 결과를 뽑아볼 때 사용합니다. |

#### 💡 사용 예시
```bash
# 1. 학습했던 모델로 K-Fold 테스트 세트 10개 전체에 대한 정밀 평가 및 종합 평균 요약 계산
python evaluate.py --model SSD_FE --run_name my_first_run

# 2. Fixed Split으로 훈련한 모델 평가 수행
python evaluate.py --model SSD_FE --run_name run_fixed --fixed_split

# 3. K-Fold의 특정 폴드(0번)만 특정 환자('VK049') 집중 평가
python evaluate.py --model SSD_FE --run_name my_first_run --folds 0 --patient VK049
```