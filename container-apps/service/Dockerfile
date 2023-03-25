FROM python:3.7-alpine as base

RUN apk add --update \
    gcc \
    g++ \
    make \
    musl-dev \
    python3-dev \
    libc6-compat \
    && rm -rf /var/cache/apk/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install all requirements
RUN pip install protobuf grpcio

COPY grpc_compiled /app/grpc_compiled/
COPY gossip-service.py /app/service.py

CMD ["python", "-u", "/app/service.py"]