FROM python:3.10
ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

RUN sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libmkl-dev \
    libomp-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements
RUN pip install --no-cache-dir --upgrade pip wheel build twine && \
    pip install --no-cache-dir -r requirements/requirements-pytorch-cpu.txt


COPY . .

# Build, check and install the birder package
CMD ["sh", "-c", "python -m build && twine check dist/* && pip install dist/birder*.whl  && \
    # Make sure it's importable
    env --chdir=/ python -c \"import birder; print(birder.__version__)\" && \
    # Install test dependencis and run tests
    pip install --no-cache-dir -r requirements/requirements-dev.txt && inv ci"]
