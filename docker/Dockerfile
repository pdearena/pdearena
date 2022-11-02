# tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags?quick-deploy=false

ARG BASE_IMAGE=22.08-py3

FROM nvcr.io/nvidia/pytorch:${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    gpg \
    curl \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    openssh-client \
    openssh-server \
    libsm6 \
    libxext6 \
    ffmpeg \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml

# Create environment
ENV PATH=/opt/conda/bin:${PATH}
ENV conda update -n base conda
RUN conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "pdearena", "/bin/bash", "-c"]


ENV PATH /opt/conda/envs/pdearena/bin:$PATH
ENV CONDA_DEFAULT_ENV pdearena
