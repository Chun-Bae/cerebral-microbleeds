# ==============================================================================
# Docker 빌드 및 실행 명령어 가이드
#
# 1. 이미지 빌드 (현재 디렉토리에서 실행)
# docker build -t samsung_cmbs .
#
# 2. 컨테이너 실행 (GPU 메모리 확보, 호스트 볼륨 마운트, ttyd 웹 터미널 포트 및 ssh 포트 개방)
# -v : 마운트, 도커에 복제할 필요 없이 호스트(Windows/WSL) 경로에 파일을 직접 연결해 쓸 수 있음.
#
# (Windows PowerShell / CMD 환경용 절대 경로 마운트 예시)
# docker run --gpus all -it --shm-size=8g \
#     -v C:\Users\Bisnel\Desktop\samsung\samsung data\samsung_data:/workspace/data/samsung_data \
#     -p 7681:7681 -p 2222:22 \
#     --name samsung_cmbs_container samsung_cmbs
#
# 참고: 컨테이너 실행 시 SSH 서버가 자동으로 켜지도록 설정되어 있습니다.
#       외부(Windows 등) 터미널 또는 VSCode에서 `ssh root@100.114.178.85 -p 2222` 커맨드로 바로 접속할 수 있습니다. (비밀번호: 0000)
#
#       컨테이너 실행 후 웹 브라우저를 통해 터미널을 띄우려면,
#       컨테이너 내부에서 [ ttyd -p 7681 tmux ] 등을 실행하고,
#       Windows(호스트)의 웹 브라우저에서 http://localhost:7681 로 접속하면 됩니다.
# ==============================================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    git \
    libgl1-mesa-glx \
    tmux \
    ttyd \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# SSH 설정 (비밀번호: 0000)
RUN mkdir /var/run/sshd && \
    echo 'root:0000' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

WORKDIR /workspace

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /workspace/third_party
RUN git clone https://github.com/MIC-DKFZ/HD-BET.git /workspace/third_party/HD-BET && \
    cd /workspace/third_party/HD-BET && \
    pip install -e .

RUN mkdir -p /workspace/data
RUN mkdir -p /workspace/results
RUN mkdir -p /workspace/src
RUN mkdir -p /workspace/tools
RUN mkdir -p /workspace/notebooks
RUN mkdir -p /workspace/scripts

# COPY data/samsung_data /workspace/data/
# data/samsung_data는 직접 copy
COPY src /workspace/src
COPY tools /workspace/tools
COPY records /workspace/records
COPY notebooks /workspace/notebooks
COPY scripts /workspace/scripts
COPY config.py train.py evaluate.py /workspace/
COPY .gitignore .README.md /workspace/

RUN chmod -R +x /workspace/tools
RUN chmod -R +x /workspace/scripts

CMD ["/bin/sh", "-c", "service ssh start && exec /bin/bash"]
