FROM python:3.8
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
    fakeroot \
    ca-certificates \
    openjdk-11-jdk-headless \
    curl

RUN pip install --no-cache-dir multi-model-server && \
    pip install --no-cache-dir "mxnet~=1.7.0"

RUN useradd -m model-server && \
    mkdir -p /home/model-server/tmp && \
    mkdir -p /home/model-server/models

COPY mms.docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY conf/config.properties /home/model-server
COPY mar /home/model-server/models

RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    chown -R model-server /home/model-server

EXPOSE 8080/tcp 8081/tcp

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
