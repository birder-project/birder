# TorchServe

## Preparing

Create model archive file (mar)

```sh
torch-model-archiver --model-name convnext_v2_4 --version 1.0 --handler birder/service/classification.py --serialized-file models/convnext_v2_4_0.pts --export-path ts
```

```sh
torch-model-archiver --model-name convnext_v2_4 --version 1.0 --handler birder/service/classification.py --serialized-file models/convnext_v2_4_0.pt2 --export-path ts --config-file ts/example_config.yaml
```

## Running

Run TorchServe

```sh
LOG_LOCATION=ts/logs METRICS_LOCATION=ts/logs torchserve --start --ncs --foreground --ts-config ts/config.properties --model-store ts/ --models convnext_v2_4.mar
```

Verify service is running

```sh
curl http://localhost:8080/ping
```

Run inference

```sh
curl http://localhost:8080/predictions/convnext_v2_4 -F "data=@data/validation/African crake/000001.jpeg"
```

### Docker

```sh
docker run --rm -it -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -v $(pwd)/ts:/home/model-server/model-store:ro pytorch/torchserve:0.11.0-cpu torchserve --model-store /home/model-server/model-store --models convnext_v2_4.mar
```
