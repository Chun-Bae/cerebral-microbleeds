## 현재까지 바뀐 목록

1. 256 → 512 이미지 입력

2. 데이터 증강 paper 그대로 구현

3. 영상을 8-bit → 16-bit로 변환

4. stratify 시 환자 분포를 살핀뒤 병변 개수에 따라 라벨을 부여하고 계층화
    - 단, VK049환자의 병변 개수는 약 300개 정도로 16개 이하인 다른 환자에 비해 많은 편
    - 그래서 VK049 환자는 우선적으로 train에만 사용
    - count == 0 → "none"
    - count <= 2 → "ver low"
    - count <= 5 → "low"
    - count <= 10 → "medium"
    - count <= 20 → "high"
    - count >  20 → "extreme"

5. 이상치의 존재로 K=5 fold로 나누어 학습
    - hold-out 20% 분리 후 8:2 = train:val 비율 구성

6. 매 학습 마다 bbox를 구하는 형식 → bbox json를 따로 구해놓고 평가
    - cv2는 cpu 자원을 사용하기에, 매 번 사용하면 학습 과정에서 gpu와의 병목이 심해짐

7. model
    - 단순 feature map to location 방식 → anchor to location model로 변경
    - SSD-512 모델로 작은 병변을 찾는데에는 한계 존재
    - 따라서 conv4_3 까지 가져오는 방식 대신,
    - conv3_3, conv4_3, conv5_3에서 각 계층별 적절한 scale, ratio 사용
    - scale은 bboxes들의 분포를 구하고 확인한 뒤, 논문에 나온 적정 공식을 사용하여 부여
    - FE 논문 수식은 아직 적용 x (제대로 된 성능 확인 후 비교 예정)

8. 손실 함수
    - anchor 방식으로 변경됨에 따라 각 병변별 anchor에 대응되는 IoU를 구함
    - 병변이 작아, IoU=0.35이면 pos로 하고 손실 합산
    - 배경이 절대 적으로 많기 때문에 Hard Negative Mining를 적용하여 손실이 큰 배경만 계산

9. 그 외 자잘한 최적화
    - png를 그대로 불러오는 것은 I/O 병목이 있을 수 있어, lmdb로 바이너리화 시킨 후, db로 통신 (에폭별 약 12분 → 10분으로 단축)
    - Window 환경은 오버헤드가 크기 때문에 Linux 환경에서 학습
    - CuDNN 적용 (VRAM 절약 및 연산 속도 증가)
    - AMP 적용 (VRAM 절약 및 연산 속도 증가)

10. 성능에 영향은 없지만 편리한 것
    - logger를 통해 매 로그마다 타임스탬프를 찍기
    - weight 저장
    - 3D 병변 시각화 스크립트
    - 3D 병변 개수 세기
    - 환자별 3D → 2D로 했을 때 2D 병변 세기
    - 병변을 윤곽선만, bbox만, 둘다 표시를 각 swi, roi별 시각화하는 이미지 제작
    - 이 때 생성된 bbox의 정보가 json에 저장
    - 추론 시 평가된 데이터도 시각화 하는 작업 추가
    - 모듈화된 메소드는 __name__ = "__main__"를 적용하여 개별 테스트 가능

11. hd-bet로 뇌 영역 마스크 처리

12. bbox 증강 처리 (좌표 변환은 정확하지 않을 수 있어, 변환된 roi로 bbox 추출)

13. Focal loss: 배경을 오탐할 때 손실을 극대화 하여 오탐 능력 완화

14. CIoU를 통해 병변을 단순 IoU 일치 없이도 정답 loc를 찾아가도록 유도