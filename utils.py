import sys
import datetime
import torch


class Logger(object):
    def __init__(self, filename):
        # 기존 stdout 사용
        self.terminal = sys.stdout
        # 로그 파일 (추가 모드)
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        # 빈 줄이 아닌 경우에만 타임스탬프 추가
        if message.strip():
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            # 현재 터미널(stdout)에 쓰기
            self.terminal.write(f"{timestamp}{message}")
            # 로그 파일에 쓰기
            self.log.write(f"{timestamp}{message}")
        else:
            # 개행이 들어가는 거 처럼, 빈 줄은 그대로 출력
            self.terminal.write(f"{message}")
            self.log.write(f"{message}")

    def flush(self):
        """
        실시간 출력 보장: 출력 버퍼 강제 비움 함수
        """
        self.terminal.flush()
        self.log.flush()


def intersect(box_a, box_b):
    """
    두 박스 세트의 교집합 영역 계산
    
    Args:
        box_a (1, 4): 정답(GT) 박스 [[2, 2, 6, 6]]
        box_b (2, 4): 앵커 박스들 [[4, 4, 8, 8], [10, 10, 12, 12]]
    """

    # 1. 오른쪽 위 벽 찾기 (xmax, ymax 중 작은 값)
    # box_a[:, 2:].unsqueeze(1) -> [[[6, 6]]] (1, 1, 2)
    # box_b[:, 2:].unsqueeze(0) -> [[[8, 8], [12, 12]]] (1, 2, 2)
    # 
    # 브로드캐스팅 비교:
    # 앵커 B[0]와 비교: min(6, 8), min(6, 8) -> [6, 6]
    # 앵커 B[1]와 비교: min(6, 12), min(6, 12) -> [6, 6]
    # 결과 max_xy: [[[6, 6], [6, 6]]] (1, 2, 2)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1), box_b[:, 2:].unsqueeze(0))

    # 2. 왼쪽 아래 벽 찾기 (xmin, ymin 중 큰 값)
    # box_a[:, :2].unsqueeze(1) -> [[[2, 2]]] (1, 1, 2)
    # box_b[:, :2].unsqueeze(0) -> [[[4, 4], [10, 10]]] (1, 2, 2)
    # 
    # 브로드캐스팅 비교:
    # 앵커 B[0]와 비교: max(2, 4), max(2, 4) -> [4, 4]
    # 앵커 B[1]와 비교: max(2, 10), max(2, 10) -> [10, 10]
    # 결과 min_xy: [[[4, 4], [10, 10]]] (1, 2, 2)
    min_xy = torch.max(box_a[:, :2].unsqueeze(1), box_b[:, :2].unsqueeze(0))

    # 3. 교집합 가로/세로 길이 계산 (끝점 - 시작점)
    # B[0] 조합: [6, 6] - [4, 4] = [2, 2] (양수 -> 겹침)
    # B[1] 조합: [6, 6] - [10, 10] = [-4, -4] (음수 -> 안 겹침)
    # 
    # clamp(min=0) 적용:
    # inter: [[[2, 2], [0, 0]]] (1, 2, 2)
    inter = torch.clamp((max_xy - min_xy), min=0)

    # 4. 최종 면적 리턴 (가로 * 세로)
    # B[0] 면적: 2 * 2 = 4
    # B[1] 면적: 0 * 0 = 0
    # return: tensor([[4.0, 0.0]]) (1, 2)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """ 
    IoU (Intersection over Union) 계산
    
    Args:
        box_a (1, 4): 정답 [cx, cy, w, h] -> [[4, 4, 4, 4]] (중심 4,4 / 가로세로 4)
        box_b (2, 4): 앵커 [[6, 6, 4, 4], [11, 11, 2, 2]]
    """

    # 1. 중심/크기(cx, cy, w, h) 포맷을 좌표(xmin, ymin, xmax, ymax)로 변환
    # A : [4-2, 4-2, 4+2, 4+2] -> [[2, 2, 6, 6]] (1, 4)
    # B0: [6-2, 6-2, 6+2, 6+2] -> [[4, 4, 8, 8]]
    # B1: [11-1, 11-1, 11+1, 11+1] -> [[10, 10, 12, 12]]
    # 결과 A: (1, 4), B: (2, 4)
    def to_coords(boxes):
        return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2,
                         boxes[:, :2] + boxes[:, 2:] / 2], dim=1)
    
    A = to_coords(box_a)
    B = to_coords(box_b)
    
    # 2. 교집합 면적 계산
    # inter 결과: tensor([[4.0, 0.0]]) (1, 2)
    # (A0와 B0는 4만큼 겹치고, A0와 B1는 안 겹침)
    inter = intersect(A, B)

    # 3. 각 박스의 개별 전체 면적 계산 (가로 * 세로)
    # area_a: [[(6-2)*(6-2)]] -> [[16.0]] (1, 1)
    # area_b: [[(8-4)*(8-4)], [(12-10)*(12-10)]] -> [[16.0, 4.0]] (2,)
    area_a = ((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])).unsqueeze(1) # (1, 1)
    area_b = ((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])).unsqueeze(0) # (1, 2)

    # 4. 합집합(Union) 면적 계산 (A면적 + B면적 - 교집합면적)
    # 중복되는 교집합 부분을 한 번 빼줘야 순수한 전체 테두리 면적이 나옴
    # 
    # B0와 조합: 16.0 + 16.0 - 4.0 = 28.0
    # B1와 조합: 16.0 + 4.0 - 0.0 = 20.0
    # 결과 union: tensor([[28.0, 20.0]]) (1, 2)
    union = area_a + area_b - inter

    # 5. 최종 IoU 계산 (교집합 / 합집합)
    # 0.0 ~ 1.0 사이의 점수가 나옴 (1.0이면 완전 일치)
    # 
    # B0 점수: 4.0 / 28.0 = 0.1428 (살짝 겹침)
    # B1 점수: 0.0 / 20.0 = 0.0 (전혀 안 겹침)
    # return: tensor([[0.1428, 0.0000]]) (1, 2)
    return inter / union

def encode(matched, anchors, variances=[0.1, 0.2]):
    """ 
    Ground Truth 좌표를 앵커 기준의 Offset(locs)으로 인코딩
    "정답까지 가려면 이만큼 변해야 함"
    
    Args:
        matched (N, 4): 앵커와 매칭된 실제 정답 박스 [cx, cy, w, h]
        anchors (N, 4): 기준이 되는 앵커 박스 [cx, cy, w, h]
    """

    # 1. 중심점 이동량 계산 (cx, cy)
    # (정답 중심 - 앵커 중심) / 앵커 너비
    # 앵커 너비로 나누는 이유: 앵커 크기에 상관없이 '상대적인 이동량'을 구하기 위함
    # variances[0](0.1)로 나누는 이유: 값을 키워서(10배) 모델이 더 민감하게 학습하게 함
    # 결과 g_cxcy: "앵커 중심에서 이만큼 옆으로/위로 가라"는 지시서
    g_cxcy = (matched[:, :2] - anchors[:, :2]) / (variances[0] * anchors[:, 2:])

    # 2. 너비/높이 변화량 계산 (w, h)
    # 정답 크기를 앵커 크기로 나눈 뒤 로그(log)를 취함
    # log를 쓰는 이유: 크기가 커지는 건 무한대지만 작아지는 건 0에 수렴하므로, 
    # 양방향 균형을 맞추기 위함 (2배 커짐 = log 2 / 0.5배 작아짐 = log 0.5)
    # 결과 g_wh: "앵커 크기를 이만큼 키우거나 줄여라"는 지시서
    g_wh = torch.log(matched[:, 2:] / anchors[:, 2:]) / variances[1]

    # 3. 하나로 합치기
    # return: (N, 4) 형태의 '수정 지시서(Target Locs)'
    return torch.cat([g_cxcy, g_wh], 1)


def decode(loc, anchors, variances=[0.1, 0.2]):
    """ 
    인코딩된 locs를 다시 이미지 좌표 [cx, cy, w, h]로 복원
    "모델이 말한 차이만큼 움직이면 해당 위치가 됨"
    
    Args:
        loc (N, 4): 모델이 예측한 수정치 (예: [0.1, -0.05, 0.2, 0.1])
        anchors (N, 4): 기준 앵커 좌표
    """

    # 1. 중심점 복원 (cx, cy)
    # 앵커 중심 + (예측치 * 0.1 * 앵커 너비)
    # "기준점(앵커)에서 모델이 가라고 한 만큼 이동시킨다"
    res_cxcy = anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:]

    # 2. 너비/높이 복원 (w, h)
    # 앵커 크기 * exp(예측치 * 0.2)
    # log의 반대인 지수함수(exp)를 써서 크기를 다시 곱셈으로 바꿈
    # "기준 크기(앵커)를 모델이 키우라는 배수만큼 키운다"
    res_wh = anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])

    # 3. 하나로 합쳐서 최종 박스 리턴
    # return: (N, 4) 이미지 상의 실제 [cx, cy, w, h]
    return torch.cat((res_cxcy, res_wh), 1)