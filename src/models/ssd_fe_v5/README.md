# SSD_FE_V5 (Cerebral Microbleeds Model)

이 문서는 `SSD_FE_V5` 모델의 아키텍처 및 내부 구조에 대한 상세한 설명을 기술합니다.
`ssd_fe_v5`는 **conv2_2 → conv3_3 → conv4_3** 으로 구성된 진정한 multi-scale 구조입니다. V2~V4가 같은 블록 내 레이어를 반복하던 것과 달리, V5는 각 스테이지가 서로 다른 공간 해상도(256×256, 128×128, 64×64)를 가져 다양한 크기의 병변을 포착할 수 있습니다.

## 1. 개요 (Overview)
* **Backbone:** VGG-16 (`torchvision.models.vgg16`)
* **해상도 지원 (Input Size):** 기본적으로 512×512 크기의 영상을 상정하여 Feature Map 사이즈를 잡음
* **특징 (Features):**
  * 3개의 스테이지가 각각 **다른 공간 해상도**를 가짐 (256×256 → 128×128 → 64×64)
  * conv2_2(128ch) → conv3_3(256ch) → conv4_3(512ch) 순으로 채널 수도 단계적으로 증가
  * `Feature Enhancement (FE)` 레이어를 `Stage 2(conv3_3)` 위에 적용

---

## 2. 예측 헤드 및 Feature Map 스테이지 (Heads & Stages)

이 모델은 VGG-16 백본에서 총 **3개의 계층(Feature Maps)** 을 뽑아와 Head(`loc`, `conf`)를 거치게 됩니다.

| Stage | VGG Block | VGG16 index | Feature Map Size | In Channels | 출력 Head / Anchors (비율) |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | `conv2_2` | `[:9]` | $256 \times 256$ | 128 | `SSDHead(128)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |
| **Stage 2** | `conv3_3` | `[9:16]` | $128 \times 128$ | 256 | `SSDHead(256)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |
| **Stage 3** | `conv4_3` | `[16:23]` | $64 \times 64$ | 512 | `SSDHead(512)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |

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

| 항목 | V1 | V2 | V3 | V4 | V5 |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Stage 구성** | conv3_3 → conv4_3 → conv5_3 | conv3_1 → conv3_2 → conv3_3 | conv3_2 → conv3_3 → conv4_3 | conv4_1 → conv4_2 → conv4_3 | conv2_2 → conv3_3 → conv4_3 |
| **Feature Map 크기** | 128/64/32 | 128/128/128 | 128/128/64 | 64/64/64 | **256/128/64** |
| **In Channels** | 256/512/512 | 256/256/256 | 256/256/512 | 512/512/512 | **128/256/512** |
| **FE 적용 위치** | Stage 2 (conv4_3) | Stage 2 (conv3_2) | Stage 2 (conv3_3) | Stage 2 (conv4_2) | Stage 2 (conv3_3) |
| **Pooling 횟수** | 3회 | 2회 | 3회 | 3회 | **2회** |

V5는 얕은 레이어(conv2_2)부터 중간(conv3_3), 깊은(conv4_3) 레이어까지 각기 다른 해상도로 골고루 포괄하는 구조입니다. V4의 Recall 저하 문제를 conv5_3 없이 해결하는 것을 목표로 합니다.
