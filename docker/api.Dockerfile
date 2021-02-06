FROM python:3.8
ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

COPY requirements/requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY frontend/server ./

RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

USER appuser
