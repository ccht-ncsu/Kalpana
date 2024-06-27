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
FROM continuumio/miniconda3 as build

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
FROM mundialis/grass-py3-pdal:7.8.8-debian

# Install libraries required to install miniconda.
RUN apt-get update

# install wget and bc
RUN apt-get install -y wget bc

# clear out the apt cache
RUN apt-get clean

# Set bash as default shell.
ENV SHELL /bin/bash

# Add user and group nru
RUN useradd --create-home -u 1000 nru
 
# Make working directory /home/nru.
WORKDIR /home/nru

# Copy /venv from the previous stage:
COPY --chown=nru --from=build /venv /venv

# Make user kalpana. 
USER nru

# make the virtual environment active
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH

# Copy Kalpana Python scripts.
COPY kalpana kalpana

# set the python path
ENV PYTHONPATH=/home/nru

# Set GDAL env variables
ENV GDAL_DATA=/venv/share/gdal
ENV GDAL_DRIVER_PATH=/venv/lib/gdalplugins
ENV PROJ_LIB=/venv/share/proj
