#!/bin/bash
# setup specific to Kalpana
version=$1;

docker run -ti --name kalpana_$version \
  --volume /xxxx/xxxxx/xxxx:/data \
  -d kalpana:$version /bin/sh
