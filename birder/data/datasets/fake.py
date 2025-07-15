from typing import Any

from torchvision.datasets import FakeData


class FakeDataWithPaths(FakeData):
    def __getitem__(self, index: int) -> tuple[str, Any, Any]:
        (img, target) = super().__getitem__(index)
        path = f"fake/path/{index}.jpeg"
        return (path, img, target)
