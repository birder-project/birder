# Build stage
FROM python:3.8 as build-stage
ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

COPY requirements/requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

COPY frontend/server ./
# COPY frontend/server/config-docker.json ./config.json

RUN python manage.py collectstatic --no-input --clear

# Production stage
FROM nginx:1.18 as production-stage

COPY --from=build-stage /usr/src/app/assets /usr/share/nginx/html/api

RUN rm /etc/nginx/conf.d/default.conf /usr/share/nginx/html/index.html
COPY docker/conf/api-nginx.conf /etc/nginx/nginx.conf

EXPOSE 80/tcp
