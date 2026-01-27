FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# ----- copy the code -----
COPY . /workspace/mamba-causal-analysis
WORKDIR /workspace/mamba-causal-analysis

# ----- install dependencies -----
RUN pip install --upgrade pip
RUN pip install uv

RUN uv sync --no-install-project

# ----- copy wrapper script -----
COPY docker/run_causal_trace.sh /usr/local/bin/run_causal_trace.sh
RUN chmod +x /usr/local/bin/run_causal_trace.sh

