# Cerebral Microbleeds (CMB) Detection

이 프로젝트는 뇌 MRI 이미지에서 미세출혈(Cerebral Microbleeds, CMB)을 탐지하는 딥러닝 모델 모듈입니다. 데이터 전처리, 모델 훈련(K-Fold 및 Fixed Split 지원), 추론 및 평가 파이프라인으로 구성되어 있습니다.

## 🚀 파이프라인 실행 가이드

모든 실행 스크립트는 프로젝트 루트 디렉토리에서 실행합니다. 핵심 실행 파일은 `train.py`와 `evaluate.py` 두 가지로 분리되어 있습니다.

### 1. 모델 훈련 (`train.py`)
모델 학습과 관련된 모든 과정을 제어합니다. 인자(옵션)를 통해서 전처리부터 세밀한 Fold 훈련까지 조정할 수 있습니다.

```bash
python train.py [options]
```

#### 📌 주요 인자 (Arguments)
| 인자 (Args) | 타입 | 설명 |
|---|---|---|
| `--prepare_data` | Flag | 학습 시작 전, NIfTI 이미지 전처리(Skull Stripping, N4 bias correction, PNG 변환) 및 LMDB 데이터베이스를 새로 구성할 때 사용합니다. (최초 1회 필수 권장) |
| `--weights` | String | (선택적) 이어서 학습하거나 파인튜닝할 가중치(`.pth`) 파일 경로를 입력합니다. 미입력 시 처음부터 학습을 시작합니다. |
| `--fixed_split` | Flag | (선택적) 모델을 훈련할 때 설정(config)의 K-Fold 옵션을 덮어쓰고, 강제로 고정된 분할(Fixed Split) 데이터셋으로만 학습을 수행합니다. |
| `--folds` | Integers | (선택적) 특정 Fold 번호만 콕 집어서 학습할 때 사용합니다. (예: `--folds 0 1 2`). 지정하지 않으면 `config.py`에 정의된 개수만큼 전체 K-Fold를 순차적으로 돌립니다. |

#### 💡 사용 예시
```bash
# 1. 처음 데이터를 전처리하고 기본 학습 시작하기
python train.py --prepare_data

# 2. 전처리 없이 0번, 2번 Fold만 특정 가중치를 베이스로 학습하기 
python train.py --folds 0 2 --weights results/latest.pth

# 3. K-Fold 대신 고정 분할 모드로만 훈련하기
python train.py --fixed_split
```

---

### 2. 모델 평가 (`evaluate.py`)
학습이 끝난 가중치를 불러와서 Test 셋을 대상으로 mAP, FROC, 오차 행렬(Confusion Matrix) 계산 및 결과 시각화를수행합니다.

```bash
python evaluate.py --weights [가중치 경로] [options]
```

#### 📌 주요 인자 (Arguments)
| 인자 (Args) | 타입 | 설명 |
|---|---|---|
| `--weights` | String | **(필수)** 평가할 타겟 모델의 가중치(`.pth`) 파일 경로입니다. |
| `--lmdb` | String | (선택적) 평가에 사용할 Test LMDB 경로입니다. (기본값: `data/lmdb/fixed_split/test.lmdb`) |
| `--patient` | String | (선택적) 전체 Test 셋을 보지 않고, 특정 환자(예: `VK049`) 데이터만 필터링하여 단독으로 평가 결과를 뽑아볼 때 사용합니다. |

#### 💡 사용 예시
```bash
# 1. 지정된 가중치로 Test 셋 전체에 대한 정밀 평가 수행
python evaluate.py --weights results/train_xxx/latest_ssd.pth

# 2. 특정 환자('VK049')의 슬라이스들에 대해서만 탐지력 집중 평가
python evaluate.py --weights results/train_xxx/latest_ssd.pth --patient VK049
```