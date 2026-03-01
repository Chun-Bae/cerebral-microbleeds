# SSD_FE_V1 (Cerebral Microbleeds Model)

이 문서는 `SSD_FE_V1` 모델(Cerebral Microbleeds 탐지용 Object Detection 신경망)의 아키텍처 및 내부 구조에 대한 상세한 설명을 기술합니다.
`ssd_fe_v1` 폴더는 기존 `ssd_fe_base` 구조를 기반으로 하면서, 뇌의 작은 출혈병변(CMBs)을 더 높은 해상도의 Feature Map에서 포착할 수 있도록 **레이어(Stage)와 Channel 수가 단축, 변경된 새로운 1번째 아키텍처 튜닝 버전**입니다.

## 1. 개요 (Overview)
* **Backbone:** VGG-16 (`torchvision.models.vgg16`)
* **해상도 지원 (Input Size):** 기본적으로 512×512 크기의 영상을 상정하여 Feature Map 사이즈를 잡음
* **특징 (Features):** 
  * 원본 데이터의 미세한 정보를 잡아내기 위해 얕은 Conv 층(Stage 1: `conv3_3`)을 앵커 베이스로 채택
  * Base 버전의 깊은 Extra Block들을 덜어내어 모델 연산량을 줄인 **Lite 형태**
  * `Feature Enhancement (FE)` 레이어를 `Stage 2(conv4_3)` 위에 씌워서 병변 의심 부위의 강도를 증폭시키는 기법 활용 

---

## 2. 예측 헤드 및 Feature Map 스테이지 (Heads & Stages)

이 모델은 멀티스케일의 영상을 보면서 작은 객체와 중간 크기 객체를 분류하고 위치를 잡기 위해, VGG-16 백본에서 총 **3개의 계층(Feature Maps)** 을 뽑아와 Head(`loc`, `conf`)를 거치게 됩니다.

| Stage | VGG Block | Feature Map Size | In Channels | 출력 Head / Anchors (비율) |
| :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | `conv3_3` | $128 \times 128$ | 256 | `SSDHead(256)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |
| **Stage 2** | `conv4_3` | $64 \times 64$ | 512 | `SSDHead(512)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |
| **Stage 3** | `conv5_3` | $32 \times 32$ | 512 | `SSDHead(512)`, 4 anchors ($r \in \{0.875, 1, 1.125\}$) |

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

## 4. `SSD_FE_base` 와의 가장 큰 차이

기존 Base 모델이 영상 깊은 곳 Conv6~Conv10 까지의 Extra Block을 사용했었다면, `V1` 모델은 원본이 뭉개지기 전인 **얕은 레이어(얕은 레벨의 공간 특징)** 위주로 예측을 수행합니다. 미세출혈 객체는 그 크기가 매우 작으므로, 풀링을 많이 타게되는 깊은 레이어보다 `conv3_3` , `conv4_3` 에서 객체를 포착하는 것이 공간 정보 손실 방지에 더 유리하다는 구조적 의도를 담고 있습니다. 
