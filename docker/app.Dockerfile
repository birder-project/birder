# Build stage
FROM node:12-buster as build-stage
WORKDIR /usr/src/app
COPY frontend/client/package.json \
    frontend/client/package-lock.json \
    # frontend/client/.env.production \
    ./

RUN npm install && npm dedupe
COPY frontend/client/ .
RUN npm run build -- --skip-plugins=eslint

# Production stage
FROM nginx:1.18 as production-stage
COPY --from=build-stage /usr/src/app/dist /usr/share/nginx/html/app

RUN rm /etc/nginx/conf.d/default.conf /usr/share/nginx/html/index.html
COPY docker/conf/app-nginx.conf /etc/nginx/nginx.conf

EXPOSE 80/tcp
