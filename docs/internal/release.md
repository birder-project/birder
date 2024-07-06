# Release

1. Make sure the full CI passes

   ```sh
   inv ci
   ```

1. Update CHANGELOG.

1. Bump version (`--major`, `--minor` or `--patch`)

    ```sh
    bumpver update --patch
    ```

1. Review the commit and tag and push.

1. Test the package

    ```sh
    docker build -f docker/test.Dockerfile . -t birder-package-test
    docker run birder-package-test:latest
    ```

1. Release to PyPI

    ```sh
    twine upload dist/*
    ```
