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
# 경로 중간에 띄어쓰기('samsung data')가 있으므로 반드시 따옴표("")로 감싸야 합니다!
# docker run --gpus all -it --shm-size=8g `
#     -v "C:\Users\Bisnel\.ssh\bisnel.pub:/root/.ssh/authorized_keys" `
#     -v "D:\samsung_project\mnt\data:/workspace/data" `
#     -v "D:\samsung_project\mnt\results:/workspace/results" `
#     -v "D:\samsung_project\mnt\weights:/workspace/weights" `
#     -p 7681:7681 -p 2222:22 `
#     --name samsung_cmbs_container samsung_cmbs

# 참고: 컨테이너 실행 시 SSH 서버가 자동으로 켜지도록 설정되어 있습니다.
#       외부(Windows 등) 터미널 또는 VSCode에서 `ssh root@100.114.178.85 -p 2222` 커맨드로 바로 접속할 수 있습니다. (비밀번호: 0000)
#
#       컨테이너 실행 후 웹 브라우저를 통해 터미널을 띄우려면,
#       컨테이너 내부에서 [ ttyd -p 7681 tmux ] 등을 실행하고,
#       Windows(호스트)의 웹 브라우저에서 http://localhost:7681 로 접속하면 됩니다.
# ==============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# apt-get 개별 실행 및 Ubuntu 22.04 기본 python3(3.10) 설치
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-venv
RUN apt-get install -y python3-dev
RUN apt-get install -y git
RUN apt-get install -y libgl1
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y tmux
RUN apt-get install -y ttyd
RUN apt-get install -y openssh-server
RUN rm -rf /var/lib/apt/lists/*

# SSH 설정 (비밀번호: 0000)
RUN mkdir -p /var/run/sshd
RUN echo 'root:0000' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

WORKDIR /workspace

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /workspace/

# pip 및 패키지 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# HD-BET 라이브러리 설치 과정 분리
RUN mkdir -p /workspace/third_party
RUN git clone https://github.com/MIC-DKFZ/HD-BET.git /workspace/third_party/HD-BET

WORKDIR /workspace/third_party/HD-BET
RUN pip install -e .
WORKDIR /workspace

# 디렉토리 생성 개별 실행
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
COPY .gitignore README.md Dockerfile /workspace/

RUN chmod -R +x /workspace/tools
RUN chmod -R +x /workspace/scripts

RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

CMD ["/bin/sh", "-c", "service ssh start && exec /bin/bash"]