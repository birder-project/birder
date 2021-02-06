# Birder

This project aim to classify bird species using deep neural networks.

This is a very early stage of the project, mostly data collection at this point.

List of supported bird species [docs/classes.md](docs/classes.md).

## Setup

Required native packages:

- protobuf-compiler
- gcc
- g++ (gcc-c++ on Fedora and CentOS)

Python 3.7 or above is required.

After cloning the repository, setup up venv and activate it (recommended).

```text
python3 -m venv .
source bin/activate
```

Update pip.

```text
pip3 install --upgrade pip
```

Install appropriate version of MXNet (e.g. mxnet-cu102), see <https://mxnet.apache.org/get_started/>
if you are unsure install non-optimized version by running:

```text
pip3 install mxnet~=1.7.0
```

Install OpenCV, for simple non-optimized install run the following:

```text
pip3 install opencv-python opencv-contrib-python
```

Install requirements

```text
pip3 install -r requirements/requirements-dev.txt
```

