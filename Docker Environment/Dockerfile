# New SFDA Basic Docker
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg software-properties-common vim curl wget git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add - \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/cuda.list

RUN apt-get update && apt-get install -y --no-install-recommends cuda-10-2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH=/opt/conda/bin:/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

WORKDIR /workspace

RUN conda create -n dla_sfda python=3.6 -y && conda clean -a -y
COPY install_dependencies.sh /install_dependencies.sh
RUN /bin/bash -c "bash /install_dependencies.sh"

RUN echo "source activate dla_sfda" >> ~/.bashrc
CMD ["/bin/bash"]

# sudo docker build -t dla_sfda_basic /home/stewes/source-free-domain-adaptive-document-layout-analysis/dockerfiles/dla_sfda_basic
