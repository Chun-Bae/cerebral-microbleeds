import torch
import torch.nn as nn
import time
import torch.nn.functional as F

# 1. 매우 깊은 ResNet 스타일의 Bottleneck 블록 정의
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# 2. ResNet-101 구조 정의 (MRI 1채널용)
class ResNet101_Microbleed(nn.Module):
    def __init__(self, num_blocks=[3, 4, 23, 3]):
        super(ResNet101_Microbleed, self).__init__()
        self.in_planes = 64
        # 초기 Conv 층
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # _make_layer 호출 (planes, num_blocks, stride 순서)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * 4, 1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides: # 변수명을 s로 변경하여 중복 방지
            layers.append(Bottleneck(self.in_planes, planes, s))
            self.in_planes = planes * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.linear(out))
        return out

# 3. 성능 측정 함수
def benchmark_model(model, input_data, iterations=30):
    with torch.no_grad():
        for _ in range(5): # Warm-up
            _ = model(input_data)
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
    torch.cuda.synchronize()
    return (time.time() - start_time) / iterations

# 4. 테스트 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet101_Microbleed().to(device).eval()

# 성능 차이를 극대화하기 위해 배치 사이즈와 해상도를 높임
dummy_data = torch.randn(64, 1, 224, 224).to(device) 

print(f"--- ResNet-101 Benchmark (Device: {device}) ---")

# Case A: cuDNN ON
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
time_on = benchmark_model(model, dummy_data)
print(f"✅ cuDNN ON : {time_on:.6f} s/iter")

# Case B: cuDNN OFF
torch.backends.cudnn.enabled = False
time_off = benchmark_model(model, dummy_data)
print(f"❌ cuDNN OFF: {time_off:.6f} s/iter")

print(f"\n결과: cuDNN 가속으로 약 {time_off / time_on:.2f}배 빨라졌습니다.")