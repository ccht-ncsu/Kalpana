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
FROM continuumio/miniconda3:master-alpine as build

# author
MAINTAINER Jim McManus

# extra metadata
LABEL version="v0.1.0"
LABEL description="Kalpana image with Dockerfile."

# update conda
RUN conda update conda

# install conda pack to compress this stage
RUN conda install -c conda-forge conda-pack

# Create the virtual environment
COPY build/env_kalpana_v1.yml .
RUN conda env create -f env_kalpana_v1.yml

# conpress the virtual environment
RUN conda-pack -n env_kalpana_v1 -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# fix up the paths
RUN /venv/bin/conda-unpack

##############
# stage 2: create a python implementation using the stage 1 virtual environment
##############
FROM mundialis/grass-py3-pdal:latest-alpine
 
# Update apk and install bash and curl.
RUN apk --update add bash curl wget ca-certificates libstdc++
ARG APK_GLIBC_VERSION=2.29-r0
ARG APK_GLIBC_FILE="glibc-${APK_GLIBC_VERSION}.apk"
ARG APK_GLIBC_BIN_FILE="glibc-bin-${APK_GLIBC_VERSION}.apk"
ARG APK_GLIBC_I18N_FILE="glibc-i18n-${APK_GLIBC_VERSION}.apk"
ARG APK_GLIBC_BASE_URL="https://github.com/sgerrand/alpine-pkg-glibc/releases/download/${APK_GLIBC_VERSION}"
RUN wget -q -O /etc/apk/keys/sgerrand.rsa.pub https://alpine-pkgs.sgerrand.com/sgerrand.rsa.pub \
    && wget "${APK_GLIBC_BASE_URL}/${APK_GLIBC_FILE}"       \
    && apk --force-overwrite --no-cache add "${APK_GLIBC_FILE}"               \
    && wget "${APK_GLIBC_BASE_URL}/${APK_GLIBC_BIN_FILE}"   \
    && apk --no-cache add "${APK_GLIBC_BIN_FILE}"           \
    && wget "${APK_GLIBC_BASE_URL}/${APK_GLIBC_I18N_FILE}"   \
    && apk --no-cache add "${APK_GLIBC_I18N_FILE}"           \
    && /usr/glibc-compat/bin/localedef -i en_US -f UTF-8 en_US.UTF-8 \
    && /usr/glibc-compat/sbin/ldconfig /lib /usr/glibc/usr/lib \
    && rm glibc-* /var/cache/apk/*

# Set bash as default shell.
ENV SHELL /bin/bash

# Add user and group nru
RUN adduser -D nru -u 1000 nru
 
# Make working directory /home/nru.
WORKDIR /home/nru

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

# Make user kalpana. 
USER nru

# make the virtual environment active
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH

# Copy Kalpana Python scripts.
COPY --chown=nru kalpana kalpana

# set the python path
ENV PYTHONPATH=/home/nru

# Set GDAL env variables
ENV GDAL_DATA=/venv/share/gdal
ENV GDAL_DRIVER_PATH=/venv/lib/gdalplugins
ENV PROJ_LIB=/venv/share/proj
