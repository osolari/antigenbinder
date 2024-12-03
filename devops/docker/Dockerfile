
FROM continuumio/miniconda3 AS build
WORKDIR /root

### Taken from miniconda3 # https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile ######
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

### Installing wget and essentials
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion gcc && \
    apt-get clean


### Install miniconda and add sources
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh -O ~/miniconda.sh && \
#    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#    rm ~/miniconda.sh && \
#    /opt/conda/bin/conda clean -tipsy && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#    echo "conda activate base" >> ~/.bashrc && \
#    find /opt/conda/ -follow -type f -name '*.a' -delete && \
#    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
#    /opt/conda/bin/conda clean -afy
### END Taken from miniconda3 ##########################################################################################

# gives us the ability to find the host ip address
RUN apt update && apt install -y \
    iputils-ping \
    iproute2 \
    htop \
    vim \
    openssh-server \
    tmux

# Setup SSH
RUN mkdir -p /root/.ssh
RUN chmod 600 /root/.ssh

#
ADD etc/requirements/conda_pinned_requirements_linux.txt /root/etc/requirements/conda_pinned_requirements_linux.txt
RUN echo `ip route | awk 'NR==1 {print $3}'` conda-package-cache >> /etc/hosts && \
    conda install --freeze-installed -y --file etc/requirements/conda_pinned_requirements_linux.txt \
     -c nvidia -c pytorch -c anaconda -c conda-forge -c bioconda -c defaults \
        --override-channels && \
    conda clean -afy

# install the pip requirements
ADD etc/requirements/pip_requirements.in etc/requirements/pip_requirements.in
RUN python -m pip install -q -r etc/requirements/pip_requirements.in --upgrade --force-reinstall --no-dependencies


# copy git into the source directory, and install
RUN mkdir -p /root/postmodict/
COPY ../.. /root/ptm/
RUN cd /root/postmodict/ && \
    python -m pip install --editable .



