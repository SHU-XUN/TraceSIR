#!/bin/bash
set -e

VERSION=1.0.0
NAME=TraceSIR-v${VERSION}

docker build -t tracesir:${VERSION} .
docker save tracesir:${VERSION} -o ${NAME}-docker.tar
gzip -f ${NAME}-docker.tar

echo "✅ Generated ${NAME}-docker.tar.gz"