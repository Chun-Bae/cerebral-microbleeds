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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

COPY HD-BET /workspace/HD-BET
RUN pip install -e /workspace/HD-BET

RUN mkdir -p /workspace/data
RUN mkdir -p /workspace/results
RUN mkdir -p /workspace/src
RUN mkdir -p /workspace/tools
RUN mkdir -p /workspace/third_party
RUN mkdir -p /workspace/notebooks
RUN mkdir -p /workspace/scripts

# COPY data/samsung_data /workspace/data/
# data/samsung_data는 직접 copy
COPY src /workspace/src
COPY tools /workspace/tools
COPY third_party /workspace/third_party
COPY records /workspace/records
COPY notebooks /workspace/notebooks
COPY scripts /workspace/scripts
COPY config.py train.py evaluate.py /workspace/
COPY .gitignore .README.md /workspace/

RUN chmod -R +x /workspace/tools
RUN chmod -R +x /workspace/scripts

CMD ["/bin/bash"]
