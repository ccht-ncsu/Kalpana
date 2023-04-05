# SPDX-FileCopyrightText: 2022 Renaissance Computing Institute. All rights reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-License-Identifier: LicenseRef-RENCI
# SPDX-License-Identifier: MIT

##############
# Docker file for running Kalpana.
#
# to create image: docker build -t kalpana:latest .
# to push image:
#       docker tag kalpana:latest containers.renci.org/eds/kalpana:latest
#       docker push containers.renci.org/eds/kalpana:latest
##############
# Use grass alpine image.
FROM mundialis/grass-py3-pdal:8.2.1-alpine

# Install libraries required to install miniconda.
RUN apk --update add bash curl wget ca-certificates libstdc++ glib \
&& wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://raw.githubusercontent.com/sgerrand/alpine-pkg-node-bower/master/sgerrand.rsa.pub \
&& curl -L "https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.23-r3/glibc-2.23-r3.apk" -o glibc.apk \
&& apk add glibc.apk \
&& curl -L "https://github.com/sgerrand/alpine-pkg-glibc/releases/download/2.23-r3/glibc-bin-2.23-r3.apk" -o glibc-bin.apk \
&& apk add glibc-bin.apk \
&& curl -L "https://github.com/andyshinn/alpine-pkg-glibc/releases/download/2.25-r0/glibc-i18n-2.25-r0.apk" -o glibc-i18n.apk \
&& apk add --allow-untrusted glibc-i18n.apk \
&& /usr/glibc-compat/bin/localedef -i en_US -f UTF-8 en_US.UTF-8 \
&& /usr/glibc-compat/sbin/ldconfig /lib /usr/glibc/usr/lib \
&& rm -rf glibc*apk /var/cache/apk/*

# Install miniconda.
ENV CONDA_DIR /opt/conda
RUN apk update && apk add ca-certificates wget && update-ca-certificates \
&& wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
&& bash ~/miniconda.sh -b -u -p $CONDA_DIR \
&& rm -rf ~/miniconda.sh \
&& $CONDA_DIR/bin/conda init bash \
&& $CONDA_DIR/bin/conda init zsh

# Put conda in path so conda activate can be used.
ENV PATH=$CONDA_DIR/bin:$PATH

# Set bash as default shell.
ENV SHELL /bin/bash

# Add user kalpana; -u xxxx (1324)  is specific to running on a RENCI VM. It
# enables writing to the /projects directory. To use change 1324 to your user ID.
# If not on a RENCI VM use RUN adduser -D kalpana kalpana.
#RUN adduser -D kalpana -u 1324 kalpana
RUN adduser -D kalpana kalpana

# Make working directory /home/kalpana.
WORKDIR /home/kalpana

# Update conda.
RUN conda update conda

# Create the virtual environment.
COPY build/env_kalpana_v1.yml .
RUN conda env create -f env_kalpana_v1.yml

# Copy Kalpana Python scripts.
COPY kalpana kalpana

# Change owner of /home/kalpoana to kalpana.
RUN chown -R kalpana:kalpana /home/kalpana

# Make user kalpana.
USER kalpana

# Initialize conda using the bash shell.
RUN conda init bash

