# SSD_FE_V6 (Cerebral Microbleeds Model)

이 문서는 `SSD_FE_V6` 모델의 아키텍처 및 내부 구조에 대한 상세한 설명을 기술합니다.
`ssd_fe_v6`는 **conv2_2 → conv3_1 → conv3_3** 으로 구성된 구조입니다. V5(conv2_2 → conv3_3 → conv4_3)에서 conv4_3을 제거하고, 그 자리에 conv3_1을 추가하여 conv4가 오탐율에 미치는 영향을 격리하는 실험입니다. 모든 스테이지가 conv3 블록 이내에 머물면서도 conv2_2의 얕은 특징을 함께 활용합니다.

## 1. 개요 (Overview)
* **Backbone:** VGG-16 (`torchvision.models.vgg16`)
* **해상도 지원 (Input Size):** 기본적으로 512×512 크기의 영상을 상정하여 Feature Map 사이즈를 잡음
* **특징 (Features):**
  * Stage 1: conv2_2 (256×256, 128ch) — 가장 얕은 저수준 텍스처 특징
  * Stage 2: conv3_1 (128×128, 256ch) — pool2 직후의 초기 block3 특징
  * Stage 3: conv3_3 (128×128, 256ch) — block3 최종 특징
  * conv4 이상 사용하지 않아 V5 대비 FP 감소 여부 확인 목적
  * `Feature Enhancement (FE)` 레이어를 `Stage 3(conv3_3)` 위에 적용

---

## 2. 예측 헤드 및 Feature Map 스테이지 (Heads & Stages)

이 모델은 VGG-16 백본에서 총 **3개의 계층(Feature Maps)** 을 뽑아와 Head(`loc`, `conf`)를 거치게 됩니다.

| Stage | VGG Block | VGG16 index | Feature Map Size | In Channels | 출력 Head / Anchors (비율) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | `conv2_2` | `[:9]` | $256 \times 256$ | 128 | `SSDHead(128)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |
| **Stage 2** | `conv3_1` | `[9:12]` | $128 \times 128$ | 256 | `SSDHead(256)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |
| **Stage 3** | `conv3_3` | `[12:16]` | $128 \times 128$ | 256 | `SSDHead(256)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |

> 💡 *참고: 위 Feature Map 사이즈는 모델에 `512`x`512` 크기의 이미지가 들어갔을 때의 기준입니다.*

---

## 3. 핵심 구성 모듈

### 3.1 `anchors.py` (AnchorGenerator)

객체를 담기 위해 사용되는 Bounding Box의 틀(Anchor Box)들을 사전에 정의합니다.
CMBs의 크기가 작고 비율이 비교적 정사각형에 가까운 점을 감안하여 각 스테이지마다 4개씩 뿌려줍니다.

* **Scale:** $[0.015 \sim 0.025]$, $[0.035 \sim 0.05]$, $[0.07 \sim 0.1]$ (원본 이미지 대비 박스의 작은 크기 할당)
* **Ratio:** Aspect Ratio를 주로 $0.875, 1.0, 1.125$ 로 고정. (미세출혈은 대부분 원형~정사각형에 가까우므로 다양한 비율은 삭제됨)

---

### 3.2 `feature_enhancement.py` (FELayer)

CMBs는 주로 T2* MRI 혹은 SWI 이미지 상에서 어두운 강도를 지닌 점 형태로 나타납니다. `FELayer`는 정답 뇌 영역(GT Mask 및 Image)의 밝기를 활용해, 모델 내부의 채널 특징값들을 선택적으로 **증폭(Enhance)** 시켜주는 모듈입니다.

1. **역정규화 및 밝기 변환:** $img\_gray = (gt\_image + 1.0) \times 0.5$ 로 0~1 값으로 돌림
2. **배경 지식 (Beta):** 병변 영역의 평균 밝기를 먼저 측정하고, 거기에 $\beta$ 값을 곱해 `B_mean` 도출
3. **강화 마스크 연산 M:** $M = \text{clamp}(1.0 - (img\_resized / B\_mean), 0, 1)$
4. **특징 강화(Fusion):** 최종 Feature Map에 $1 + M$ 값을 곱해주어 $X_{new} = X_{old} \times (1 + M)$ 형태로 가중치 증폭 $\rightarrow$ 다음 레이어나 Head에 강하게 주입

> ⚡ **참고:** 이 모듈은 **학습(Train) 간에만 `gt_image`, `gt_mask`가 전달될 때 동작**합니다. 평가나 추론 시 인자가 `None`으로 들어오면 단순히 입력 $X_{old}$ 를 바이패스(Pass-through) 합니다.

---

### 3.3 `head.py` (SSDHead)

예측의 최종 관문.
각 스테이지별 Feature Map 픽셀마다 (Channel 수를 입력받아) 다음과 같은 두 개의 3x3 Convolution을 나란히 진행합니다.
1. `loc`: Bounding Box 오프셋(cx, cy, w, h) 회귀용 (출력크기: `Anchors × 4`)
2. `conf`: 클래스(병변 vs 배경) 예측 점수 산출용 (출력크기: `Anchors × 2`)

---

## 4. 버전별 비교

| 항목 | V1 | V2 | V3 | V4 | V5 | V6 |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 구성** | conv3_3 → conv4_3 → conv5_3 | conv3_1 → conv3_2 → conv3_3 | conv3_2 → conv3_3 → conv4_3 | conv4_1 → conv4_2 → conv4_3 | conv2_2 → conv3_3 → conv4_3 | conv2_2 → conv3_1 → conv3_3 |
| **Feature Map 크기** | 128/64/32 | 128/128/128 | 128/128/64 | 64/64/64 | 256/128/64 | **256/128/128** |
| **In Channels** | 256/512/512 | 256/256/256 | 256/256/512 | 512/512/512 | 128/256/512 | **128/256/256** |
| **FE 적용 위치** | Stage 2 (conv4_3) | Stage 2 (conv3_2) | Stage 2 (conv3_3) | Stage 2 (conv4_2) | Stage 2 (conv3_3) | Stage 3 (conv3_3) |
| **Pooling 횟수** | 3회 | 2회 | 3회 | 3회 | 2회 | **2회** |

V6는 V5에서 conv4_3을 제거하고 conv3_1을 추가한 구조입니다. conv4가 FP 증가에 기여하는지 격리 검증하면서, conv2_2의 얕은 특징과 conv3 블록의 점진적 특징을 함께 활용합니다.
