# 뇌 미세출혈 자동 검출 시스템

> **SSD 기반 소형 병변 탐지 모델 — 앵커 재설계와 Feature Enhancement를 통한 성능 개선**
> Python · PyTorch · SSD (Single Shot MultiBox Detector) · VGG-16 · LMDB · Kornia · HD-BET · Docker
> 개발 기간: 2025년 10월 ~ 2025년 05월 / R&D 과제 프로젝트 (개인 수행)

---

## 1. 도입 배경

### 문제 인식

뇌 미세출혈(Cerebral Microbleeds, CMB)은 SWI(Susceptibility-Weighted Imaging) MRI에서 관찰되는 직경 2~10mm의 미세한 출혈 병변입니다. 임상에서는 전문의가 수백 장의 MRI 슬라이스를 육안으로 확인하며 병변을 찾아야 하는데, 병변이 매우 작고 주변 조직과의 대비가 낮아 **숙련된 전문의도 놓치기 쉬운 영역**입니다.

참고 논문에서는 SSD(Single Shot MultiBox Detector) 기반 모델로 CMB를 탐지하는 접근법을 제시했지만, **기본 SSD 아키텍처를 그대로 사용**하고 있었습니다. SSD는 원래 일반 객체(사람, 차량 등)를 탐지하기 위해 설계된 모델로, 깊은 레이어의 고해상도 특징맵에서 큰 객체를 탐지하는 데 최적화되어 있습니다. **이미지 대비 1~2%에 불과한 CMB를 탐지하기에는 구조적으로 부적합**한 설정이었습니다.

이 프로젝트는 SSD의 앵커 배치와 특징맵 추출 전략을 CMB의 크기 특성에 맞게 **근본적으로 재설계**하고, 데이터 전처리부터 학습, 평가까지의 End-to-End 파이프라인을 구축하는 것을 목표로 했습니다.

### 프로젝트 목표

- SSD 모델의 **앵커를 얕은 레이어(conv3_3, conv4_3, conv5_3)에 재배치**하여 소형 병변 탐지 성능 확보
- Feature Enhancement Layer로 **병변 영역의 특징을 학습 시 선택적으로 강화**
- HD-BET 기반 두개골 제거 → N4 보정 → LMDB 변환의 **자동화된 전처리 파이프라인** 구축
- K-Fold 교차검증과 Fixed Split을 모두 지원하는 **재현 가능한 학습/평가 시스템** 구현

---

## 2. 기술적 문제 해결 과정

### 문제 1 — 기본 SSD 아키텍처로는 소형 병변을 탐지할 수 없는 문제

**상황**
참고 논문의 구조를 따라 기본 SSD 아키텍처(VGG-16 백본 + conv6~conv10 Extra Layer)로 학습을 진행했습니다. 기본 SSD는 깊은 레이어에서 추출한 특징맵(38×38, 19×19, 10×10, ...)에 앵커를 배치하는 구조인데, 깊은 레이어로 갈수록 풀링(Pooling)이 반복되면서 **공간 해상도가 급격히 줄어듭니다**. CMB는 512×512 입력 기준으로 수 픽셀에 불과한 크기이므로, 깊은 레이어에서는 병변의 공간 정보가 이미 소실된 상태입니다.

결과적으로 기본 SSD에서는 Recall, Precision, AP 모두 실용적인 수준에 미치지 못했습니다.

**원인 분석**
SSD의 앵커 설계를 CMB의 크기와 대조해 분석했습니다.

- 기본 SSD의 최소 앵커 스케일은 이미지 크기의 약 20%부터 시작 — CMB는 **1~2%** 수준
- 깊은 레이어(conv6 이후)의 특징맵은 해상도가 낮아 소형 객체의 위치 정보가 손실됨
- 기본 SSD의 aspect ratio(1:2, 2:1 등)는 일반 객체용이며, **거의 정사각형인 CMB에는 부적합**

**해결**
SSD 아키텍처를 근본적으로 수정하여, **얕은 레이어(conv3_3, conv4_3, conv5_3)에서 특징맵을 추출하고 앵커를 배치**하는 구조로 재설계했습니다.

```python
class SSD_FE_V1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 512×512 입력 기준, 얕은 레이어에서 고해상도 특징맵 추출
        self.feature_maps = [(128, 128), (64, 64), (32, 32)]

        # CMB 크기(1~2%)에 맞춘 미세 앵커 스케일
        self.scales = [
            [0.015, 0.025],   # conv3_3: 이미지 대비 1.5~2.5%
            [0.035, 0.050],   # conv4_3: 3.5~5.0%
            [0.070, 0.100],   # conv5_3: 7.0~10.0%
        ]

        # 정사각형에 가까운 aspect ratio (CMB 형태 반영)
        self.ratios = [
            [0.875, 1.0, 1.125],
            [0.875, 1.0, 1.125],
            [0.875, 1.0, 1.125],
        ]
```

| 비교 항목 | 기본 SSD | 수정된 SSD (SSD_FE_V1) |
|-----------|---------|----------------------|
| 특징맵 추출 위치 | conv6~conv10 (깊은 레이어) | conv3_3, conv4_3, conv5_3 (얕은 레이어) |
| 최대 특징맵 해상도 | 38×38 | **128×128** |
| 최소 앵커 스케일 | ~0.20 (20%) | **0.015 (1.5%)** |
| Aspect Ratio | 1:1, 1:2, 2:1 등 | **0.875, 1.0, 1.125** (정사각형 중심) |

이 수정만으로 **Recall, Precision, AP가 약 50%p 이상 상승**했습니다. SSD의 성능이 낮았던 원인이 모델 자체의 한계가 아니라, **탐지 대상의 크기 특성에 앵커가 맞지 않았기 때문**이라는 것을 확인한 핵심 실험이었습니다.

---

### 문제 2 — Feature Enhancement Layer의 데이터 증강 불일치

**상황**
Feature Enhancement(FE) Layer는 학습 시 GT(Ground Truth) 바운딩 박스 정보를 활용하여 **병변 영역의 특징값을 선택적으로 강화**하는 모듈입니다. 병변 부위의 어두운 픽셀(CMB는 SWI에서 저신호)일수록 강화 계수가 높아지는 방식으로, 모델이 미세한 병변 패턴을 더 잘 학습하도록 유도합니다.

그런데 학습 과정에서 **데이터 증강(특히 이동, 회전 등 기하학적 변환)이 적용된 이미지에 대해 FE가 올바르게 동작하지 않는 문제**를 발견했습니다.

**원인 분석**
데이터 증강은 이미지와 바운딩 박스 좌표에 동일한 기하학적 변환을 적용해야 합니다. 하지만 FE Layer에 전달되는 GT 마스크에는 증강이 반영되지 않아, **이동된 이미지에서 원래 위치의 영역을 강화하는 불일치**가 발생하고 있었습니다. 즉, 병변이 아닌 영역이 강화되거나 병변 영역이 누락되는 상황이었습니다.

**해결**
FE Layer의 입력으로 사용되는 GT 마스크를 **증강 후의 바운딩 박스에서 실시간으로 생성**하도록 수정하여, 기하학적 변환이 적용된 이미지와 FE 마스크 간의 일관성을 보장했습니다.

```python
# FE Layer: 병변의 밝기 기반으로 특징 강화
class FELayer(nn.Module):
    def forward(self, feature_map, gt_image, gt_mask):
        with torch.no_grad():
            # 병변 영역의 평균 밝기 계산
            lesion_pixels = img_gray[b][gt_mask[b] > 0]
            region_mean = lesion_pixels.mean().item()

            # 강화 계수: 어두울수록(병변일수록) 값이 커짐
            # M = 1 - (pixel_brightness / (β × mean_brightness))
            B_mean = self.beta * region_mean
            M = (1.0 - (img_resized[b] / B_mean)).clamp(min=0, max=1)

            # GT 마스크 영역에만 강화 적용
            enhanced_mask[b] = M * mask_resized[b]

        # 최종: X_new = X × (1 + M) — 마스크 바깥은 원본 유지
        return feature_map * (1.0 + enhanced_mask)
```

FE Layer는 `torch.no_grad()` 블록 내에서 마스크를 계산하므로 **역전파에는 영향을 주지 않으면서**, 순전파 시 병변 영역의 특징 응답만 증폭시키는 구조입니다. 추론 시에는 GT가 없으므로 FE가 비활성화되어 **학습 시에만 동작하는 보조 모듈**입니다.

---

### 문제 3 — 뇌 배경 영역으로 인한 학습 불안정

**상황**
MRI 이미지에서 **실제 뇌 조직이 차지하는 영역은 전체 이미지의 일부**에 불과하고, 나머지는 두개골, 공기, 배경(검은색) 등 병변이 존재할 수 없는 영역입니다. SSD 모델은 전체 이미지에 걸쳐 수만 개의 앵커를 배치하므로, 배경 영역의 앵커들이 대량의 **무의미한 음성 샘플**로 작용하여 학습의 안정성과 효율을 떨어뜨리고 있었습니다.

**해결 — 다단계 배경 억제 전략**

**1단계: HD-BET 기반 두개골 제거 (전처리)**

HD-BET(HD Brain Extraction Tool)을 전처리 파이프라인에 통합하여, 뇌 영역만 남기고 두개골 및 외부 조직을 자동으로 제거했습니다. 이를 통해 모델이 학습해야 할 영역 자체를 줄였습니다.

**2단계: Brain Mask 기반 앵커 필터링 (학습 시)**

전처리에서 생성된 ROI(뇌 영역) 마스크를 손실 함수에 전달하여, **뇌 영역 바깥에 위치한 앵커를 학습에서 완전히 제외**했습니다.

```python
def _apply_brain_mask(self, anchors, brain_masks, valid_anchor_mask, ...):
    with torch.no_grad():
        # 각 앵커의 중심점이 뇌 영역 내부인지 확인
        anchor_centers = anchors[:, :2]
        grid = anchor_centers.view(1, num_anchors, 1, 2) * 2 - 1

        # grid_sample로 뇌 마스크에서 앵커 중심점의 값을 샘플링
        sampled_mask = F.grid_sample(brain_masks, grid, align_corners=False)

        # 뇌 영역 밖의 앵커는 학습에서 제외
        valid_anchor_mask[:] = sampled_mask > 0.01
```

이 필터링은 `grid_sample`을 사용하여 배치 단위로 효율적으로 수행되며, 배경 앵커가 **손실 계산 자체에서 제외**되므로 gradient에 영향을 주지 않습니다.

**3단계: Hard Negative Mining (학습 시)**

뇌 영역 내부에서도 배경 앵커가 병변 앵커보다 압도적으로 많은 클래스 불균형이 존재합니다. SSD의 Hard Negative Mining을 적용하여, **손실 값이 가장 높은 음성 샘플만 선별적으로 학습**에 포함시켰습니다.

```python
# 음성 샘플 중 손실이 가장 큰 것만 선택 (양성:음성 = 1:3)
num_neg = torch.clamp(self.neg_pos_ratio * num_pos, min=100)
_, loss_idx = loss_neg.sort(1, descending=True)
_, idx_rank = loss_idx.sort(1)
neg_mask = idx_rank < num_neg.expand_as(idx_rank)
```

> 전처리(HD-BET) → 손실 함수(Brain Mask 필터링) → Mining(Hard Negative) 의 3단계로 배경 영향을 점진적으로 억제하여, 모델이 **뇌 조직 내부의 미세한 병변 패턴에 집중**할 수 있도록 설계했습니다.

---

### 문제 4 — 의료 영상 데이터의 전처리 자동화와 파이프라인 설계

**상황**
원본 데이터는 NIfTI(.nii.gz) 형식의 3D 볼륨으로, 학습에 사용하려면 두개골 제거 → 바이어스 보정 → 2D 슬라이스 변환 → 바운딩 박스 추출 → LMDB 패킹의 여러 단계를 거쳐야 합니다. 환자별로 수십~수백 슬라이스가 생성되므로, 수동 처리는 비현실적이었습니다. 또한 각 단계가 이전 단계의 출력에 의존하는 순차적 파이프라인이므로, 중간에 하나가 실패하면 전체를 다시 실행해야 하는 문제도 있었습니다.

**해결**
각 전처리 단계를 독립적인 함수로 모듈화하고, `DataPipeLine` 클래스가 이를 순차적으로 오케스트레이션하는 구조를 설계했습니다. 사용자는 `--prepare_data` 플래그 하나로 전체 파이프라인을 실행할 수 있습니다.

```python
class DataPipeLine:
    def run(self):
        self.process_skull_stripping()      # 1. HD-BET 뇌 추출
        self.process_n4_correction()        # 2. N4 바이어스 보정
        self.process_nii_to_png()           # 3. NIfTI → 2D PNG 슬라이스
        self.extract_bboxes()               # 4. 마스크 → 바운딩 박스 JSON
        self.generate_splits_patients()     # 5. 환자 단위 K-Fold + Fixed 분할
        self.generate_lmdb_dataset()        # 6. LMDB 데이터베이스 생성
```

```bash
[전처리 파이프라인 흐름]
NIfTI 3D Volume (.nii.gz)
  ├─ 1. Skull Stripping (HD-BET)     → 뇌 영역만 추출, 두개골/배경 제거
  ├─ 2. N4 Bias Field Correction     → MRI 밝기 편향 보정 (일관된 강도 분포)
  ├─ 3. NIfTI → PNG Slice 변환        → 2D 슬라이스 + ROI 마스크 생성
  ├─ 4. Bounding Box 추출             → 마스크 컨투어에서 병변 좌표 JSON 생성
  ├─ 5. 환자 단위 데이터 분할           → K-Fold (10-Fold) / Fixed Split
  └─ 6. LMDB 데이터베이스 생성          → 고속 I/O용 메모리 매핑 저장

[실행 진입점]
python train.py --model SSD_FE_V1 --run_name exp1 --prepare_data   # 전처리 + 학습
./scripts/train_eval/both.sh SSD_FE_V1 exp1                        # 원클릭 학습 + 평가
```

데이터 분할 시 **환자 단위(patient-level)로 분리**하여, 동일 환자의 슬라이스가 학습/검증/테스트 세트에 동시에 포함되는 데이터 누출(data leakage)을 방지했습니다. LMDB 형식을 채택하여 학습 중 수만 장의 PNG를 개별 디스크 I/O로 읽는 대신 **메모리 매핑 기반 고속 접근**을 구현했습니다.

각 파이프라인 단계를 모듈화한 덕분에, 특정 단계만 개별 실행하거나 설정을 변경하여 재실행하는 것이 용이합니다. 학습 스크립트도 `Fixed Split 학습 → 평가 → K-Fold 학습 → 평가`를 하나의 셸 스크립트로 연결하여, 실험 재현성과 편의성을 모두 확보했습니다.

---

### 문제 5 — Docker 기반 멀티 환경 실행 체계 구축

**상황**
학습을 집과 연구실 워크스테이션 두 곳에서 병행하고 있었습니다. 두 환경은 OS, CUDA 버전, Python 버전, 시스템 라이브러리 등이 서로 달랐고, 한쪽에서 정상 동작하는 코드가 다른 쪽에서는 의존성 충돌이나 라이브러리 누락으로 실패하는 일이 반복되었습니다. 특히 HD-BET(뇌 추출 모델)은 별도의 설치 과정이 필요하고, PyTorch CUDA 빌드는 환경에 따라 호환 버전이 달라 **환경 세팅에 소모되는 시간이 실제 개발 시간을 잠식**하는 상황이었습니다.

**해결**
전체 실행 환경을 Docker 이미지로 패키징하여, 어떤 환경에서든 동일한 조건으로 학습과 평가를 수행할 수 있도록 했습니다.

```dockerfile
FROM ubuntu:24.04

# Python 3.10 + venv 기반 격리 환경
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch CUDA 11.8 빌드 고정 설치
RUN pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# HD-BET 서드파티 의존성을 이미지 내에 포함
COPY HD-BET /workspace/HD-BET
RUN pip install -e /workspace/HD-BET

# 소스 코드, 스크립트, 설정 파일 복사
COPY src tools scripts config.py train.py evaluate.py /workspace/
```

핵심 설계 포인트:

- **CUDA 버전 고정**: `cu118` 빌드를 명시적으로 지정하여 호스트의 CUDA 드라이버와 무관하게 일관된 동작 보장
- **HD-BET 내장**: 서드파티 모델을 이미지에 포함시켜 별도 설치 과정 제거
- **데이터 분리**: `data/` 디렉토리는 이미지에 포함하지 않고 볼륨 마운트로 연결 — 이미지 크기 절약 및 데이터 보안
- **실행 스크립트 포함**: `scripts/` 내의 셸 스크립트로 컨테이너 내부에서 원클릭 학습/평가 가능

이를 통해 집이나 연구실 어디서든 `docker run` 한 줄로 동일한 학습 환경을 즉시 구성할 수 있게 되었고, **환경 세팅 문제로 인한 시간 낭비가 완전히 해소**되었습니다.

---

### 문제 6 — 학습 안정성 확보

**상황**
소형 객체 탐지 모델은 앵커와 GT 간의 오프셋이 매우 민감하고, 양성 샘플 수가 극히 적어 **gradient가 불안정**해지기 쉽습니다. 초기 학습 과정에서 loss가 `NaN`으로 발산하거나, gradient explosion이 간헐적으로 발생하는 문제가 있었습니다.

**해결**

| 기법 | 적용 방식 |
|------|-----------|
| Mixed Precision (AMP) | `GradScaler`로 FP16/FP32 혼합 연산, 수치 안정성과 메모리 효율 확보 |
| Gradient Clipping | `max_norm=5.0`으로 gradient 크기 제한 |
| NaN/Inf Guard | gradient에 NaN/Inf 발생 시 0으로 대체하여 학습 중단 방지 |
| Offset Variance | 바운딩 박스 인코딩 시 xy는 0.1, wh는 0.2의 variance를 적용해 오프셋 범위 정규화 |
| Learning Rate Schedule | 반복 기반 3단계 스케줄링 (1e-3 → 1e-4 → 1e-5) |

---

## 3. 아키텍처 설계

```bash
┌─────────────────────────────────────────────────────────────────┐
│                     Input: SWI MRI Slice                        │
│                512 × 512 × 3 (Grayscale → 3ch 복제)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  VGG-16 Backbone (Pretrained)                   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  conv3_3    │  │  conv4_3    │  │  conv5_3    │              │
│  │  128 × 128  │  │  64 × 64    │  │  32 × 32    │              │
│  │  256ch      │  │  512ch      │  │  512ch      │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                     │
│         │         ┌──────▼──────┐         │                     │
│         │         │  FE Layer   │         │                     │
│         │         │ (학습 시만)   │         │                     │
│         │         └──────┬──────┘         │                     │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
   │  SSD Head 1 │  │  SSD Head 2 │  │  SSD Head 3 │
   │ anchors: 4  │  │ anchors: 4  │  │ anchors: 4  │
   │ scale: 1.5~ │  │ scale: 3.5~ │  │ scale: 7.0~ │
   │       2.5%  │  │       5.0%  │  │       10.0% │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │         Predictions         │
            │   loc: (N, num_anchors, 4)  │
            │   conf: (N, num_anchors, 2) │
            │   anchors: (num_anchors, 4) │
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │       Post-Processing       │
            │  Confidence Filter → NMS →  │
            │  Brain Mask Filter          │
            └─────────────────────────────┘
```

**레이어 분리 원칙**

- `src/models/` — SSD 아키텍처, FE Layer, 앵커 생성, 예측 헤드 정의
- `src/core/training/` — 학습 루프, 컴파일러(모델/옵티마이저 조립), 체크포인트 관리
- `src/core/evaluation/` — 후처리(NMS), 메트릭 계산(AP, FROC), 시각화
- `src/core/data_procssing/` — 전처리 파이프라인 (Skull Stripping, N4, 변환, 분할)
- `src/datasets/` — LMDB 데이터셋 로더, Kornia 기반 데이터 증강
- `src/losses/` — MultiBox Loss (Hard Negative Mining 포함)
- `src/pipelines/` — 전처리/학습/평가 파이프라인 오케스트레이션

---

## 4. 개발 과정

| 단계 | 내용 |
|------|------|
| **데이터 전처리** | HD-BET 통합, N4 보정, NIfTI→PNG 변환, LMDB 패킹 파이프라인 구축 |
| **기본 SSD 구현** | 논문 기반 SSD + VGG-16 구조 구현 및 학습 → 낮은 성능 확인 |
| **앵커 재설계** | 얕은 레이어(conv3_3~conv5_3)로 특징맵 추출 위치 변경, 미세 스케일 앵커 설계 → **성능 ~50%p 향상** |
| **FE Layer** | 병변 영역 특징 강화 모듈 구현, 데이터 증강과의 정합성 버그 수정 |
| **학습 안정화** | Brain Mask 앵커 필터링, Hard Negative Mining, Gradient Clipping 적용 |
| **평가 체계** | mAP, FROC, Confusion Matrix 메트릭 구현, K-Fold 교차검증 |
| **Docker화** | 집/연구실 워크스테이션 이중 환경에서 동일하게 동작하는 컨테이너 이미지 구축 |
| **실행 자동화** | 셸 스크립트 기반 원클릭 학습/평가 실행 (Fixed + K-Fold 연속 수행) |

---

## 5. 성과 및 배운 점

### 기술적 성과

- 기본 SSD 대비 **앵커 재설계만으로 Recall, Precision, AP ~50%p 향상** — 탐지 대상의 크기 특성에 맞춘 앵커 설계가 가장 결정적인 성능 요인임을 검증
- **3단계 배경 억제**(HD-BET → Brain Mask 필터링 → Hard Negative Mining)로 뇌 영상 특유의 넓은 배경 영역 문제를 체계적으로 해결
- Feature Enhancement Layer의 **증강 불일치 버그**를 발견하고 수정하여, GT 정보 기반 학습 보조 모듈의 정합성 확보
- NIfTI 전처리부터 K-Fold 평가까지 **재현 가능한 End-to-End 파이프라인** 구축 — 각 단계를 모듈화하여 개별/전체 실행 모두 지원
- Docker 컨테이너화로 집/연구실 **이중 환경에서 동일한 학습 조건**을 보장하는 실행 체계 구축

### 배운 점

- 모델 아키텍처 자체보다 **탐지 대상의 물리적 특성에 맞는 설계**가 성능을 좌우한다는 것을 체감 — SSD라는 동일한 프레임워크에서 앵커 위치와 스케일만 바꿨을 뿐인데 성능이 극적으로 개선됨
- 논문의 접근법을 그대로 재현하는 것과, 실제로 **문제를 분석하고 아키텍처를 수정하는 것** 사이에는 큰 차이가 있으며, 후자가 실질적인 성능 개선으로 이어짐
- 의료 영상 데이터는 일반 이미지와 달리 **배경 비율, 객체 크기, 클래스 불균형** 등 도메인 특수성이 강하므로, 범용 모델을 그대로 쓰기보다 도메인에 맞춘 수정이 필수적
- 데이터 증강과 모델의 학습 보조 모듈(FE Layer) 간의 **정합성 검증**이 중요 — 증강이 적용된 데이터에 원본 기준의 마스크를 사용하면 학습 신호가 오염됨
- 전처리 파이프라인을 체계적으로 설계하는 과정에서, 각 단계의 입출력을 명확히 정의하고 모듈화하는 것이 **실험 반복 속도와 재현성 모두에 직결**된다는 점을 체감
- Docker를 통한 환경 격리는 단순한 편의가 아니라, 다중 환경에서 **실험 결과의 일관성을 보장**하는 필수 인프라 — 환경 차이로 인한 디버깅 시간을 개발에 투자할 수 있게 됨

---

## 6. 개선 가능한 부분

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| 탐지 모델 | SSD 단일 모델 | YOLO, RetinaNet 등 최신 탐지 모델과의 비교 실험 |
| 입력 구조 | 2D 단일 슬라이스 (3채널 복제) | 인접 슬라이스를 활용한 진정한 2.5D 또는 3D 입력 |
| 데이터 규모 | 제한된 환자 수 | 다기관(multi-center) 데이터 확보 및 외부 검증 |
| 앵커 설계 | 수동 스케일/비율 설정 | Anchor-Free 방식 또는 학습 기반 앵커 최적화 |
| 데이터 저장 | LMDB 파일 기반 | 대규모 데이터셋 대응을 위한 분산 저장소 |
| FE Layer | 고정 β 파라미터 | β를 학습 가능한 파라미터로 전환하여 자동 조절 |
