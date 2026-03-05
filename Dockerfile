FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install uv system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | bash \
    && mv /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/mamba-causal-analysis

# Copy source code
COPY . .

# Install dependencies into local venv
RUN uv pip install --system .

# Make entire workspace world-readable/executable
RUN chmod -R a+rX /workspace
