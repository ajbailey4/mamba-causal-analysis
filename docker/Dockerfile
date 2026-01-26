FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ----- basic environment hygiene -----
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ----- system dependencies -----
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python available as `python`
RUN ln -s /usr/bin/python3 /usr/bin/python

# ----- install uv -----
RUN pip install --no-cache-dir uv

# ----- working directory -----
WORKDIR /workspace

# ----- dependency resolution first (for Docker caching) -----
COPY pyproject.toml uv.lock ./

# Install deps into system Python
RUN uv pip install --system -r uv.lock

# ----- copy your code -----
COPY mamba_causal_analysis /workspace/mamba_causal_analysis

# ----- copy wrapper script -----
COPY docker/run_job.sh /usr/local/bin/run_job.sh
RUN chmod +x /usr/local/bin/run_job.sh


# ----- default entrypoint -----
ENTRYPOINT ["bash"]
