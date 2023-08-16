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
FROM frolvlad/alpine-miniconda3 as build

# author
MAINTAINER Jim McManus

# extra metadata
LABEL version="v0.0.9"
LABEL description="Kalpana image with Dockerfile."

# update conda
RUN conda update conda

# Create the virtual environment
COPY build/env_kalpana_v1.yml .
RUN conda env create -f env_kalpana_v1.yml

# install conda pack to compress this stage
RUN conda install -c conda-forge conda-pack

# conpress the virtual environment
RUN conda-pack -n env_kalpana_v1 -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# fix up the paths
RUN /venv/bin/conda-unpack

##############
# stage 2: create a python implementation using the stage 1 virtual environment
##############
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

# Set bash as default shell.
ENV SHELL /bin/bash

# Add user kalpana; -u xxxx (1324)  is specific to running on a RENCI VM. It
# enables writing to the /projects directory. To use change 1324 to your user ID.
# If not on a RENCI VM use RUN adduser -D kalpana kalpana.
#RUN adduser -D nru -u 1324 nru
RUN adduser -D nru -u 1000 nru

# Make working directory /home/nru.
WORKDIR /home/nru

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# make the virtual environment active
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH

# Copy Kalpana Python scripts.
COPY kalpana kalpana

# Set GDAL env variables
ENV GDAL_DATA=/venv/share/gdal
ENV GDAL_DRIVER_PATH=/venv/lib/gdalplugins
ENV PROJ_LIB=/venv/share/proj

# Change owner of /home/nru to nru.
RUN chown -R nru:nru /home/nru

# Make user kalpana.
USER nru

# Initialize conda using the bash shell.
#RUN conda init bash

