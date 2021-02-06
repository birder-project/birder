#!/bin/bash

# Taken from https://github.com/awslabs/multi-model-server/blob/master/docker/dockerd-entrypoint.sh

set -e

if [[ "$1" = "serve" ]]; then
	shift 1
	multi-model-server --start --mms-config config.properties
else
	eval "$@"
fi

# Prevent docker exit
tail -f /dev/null
