import json
import logging
from typing import Any
from typing import TypedDict

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from birder.conf import settings

logger = logging.getLogger(__name__)

MetricsType = TypedDict(
    "MetricsType",
    {
        "map": float,
        "map_50": float,
        "map_75": float,
        "map_small": float,
        "map_medium": float,
        "map_large": float,
        "mar_1": float,
        "mar_10": float,
        "mar_100": float,
        "mar_small": float,
        "mar_medium": float,
        "mar_large": float,
        "ious": dict[tuple[int, int], torch.Tensor],
        "precision": torch.Tensor,
        "recall": torch.Tensor,
        "scores": torch.Tensor,
        "map_per_class": list[float],
        "mar_100_per_class": list[float],
        "classes": list[int],
    },
)


class Results:
    """
    Detection result analysis class
    """

    def __init__(
        self,
        sample_paths: list[str],
        targets: list[dict[str, Any]],
        detections: list[dict[str, torch.Tensor]],
        class_to_idx: dict[str, int],
    ):
        assert len(sample_paths) == len(targets)
        assert len(sample_paths) == len(detections)

        detections = [{k: v.cpu() for k, v in detection.items()} for detection in detections]
        targets = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        for target in targets:
            if "image_id" in target:
                del target["image_id"]

            # TorchMetrics can't handle "empty" images
            if "boxes" not in target:
                target["boxes"] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
                target["labels"] = torch.tensor([], dtype=torch.int64, device=torch.device("cpu"))

        metrics = MeanAveragePrecision(
            iou_type="bbox", box_format="xyxy", class_metrics=True, extended_summary=True, average="macro"
        )
        metrics(detections, targets)
        metrics_dict = metrics.compute()

        self._class_to_idx = class_to_idx
        self._detections = detections
        self._targets = targets
        self._sample_paths = sample_paths

        self.metrics_dict: MetricsType = {
            "map": metrics_dict["map"].item(),
            "map_50": metrics_dict["map_50"].item(),
            "map_75": metrics_dict["map_75"].item(),
            "map_small": metrics_dict["map_small"].item(),
            "map_medium": metrics_dict["map_medium"].item(),
            "map_large": metrics_dict["map_large"].item(),
            "mar_1": metrics_dict["mar_1"].item(),
            "mar_10": metrics_dict["mar_10"].item(),
            "mar_100": metrics_dict["mar_100"].item(),
            "mar_small": metrics_dict["mar_small"].item(),
            "mar_medium": metrics_dict["mar_medium"].item(),
            "mar_large": metrics_dict["mar_large"].item(),
            "ious": metrics_dict["ious"],
            "precision": metrics_dict["precision"],
            "recall": metrics_dict["recall"],
            "scores": metrics_dict["scores"],
            "map_per_class": metrics_dict["map_per_class"].tolist(),
            "mar_100_per_class": metrics_dict["mar_100_per_class"].tolist(),
            "classes": metrics_dict["classes"].tolist(),
        }

    def __len__(self) -> int:
        return len(self._sample_paths)

    def __repr__(self) -> str:
        head = self.__class__.__name__
        body = [
            f"Number of samples: {len(self)}",
        ]

        body.append(f"mAP: {self.map:.3f}")

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)

    @property
    def map(self) -> float:
        return self.metrics_dict["map"]

    def save(self, name: str) -> None:
        """
        Save results object to file

        Parameters
        ----------
        name
            output file name.
        """

        detections = [{k: v.numpy().tolist() for k, v in detection.items()} for detection in self._detections]
        targets = [{k: v.numpy().tolist() for k, v in target.items()} for target in self._targets]
        output = dict(zip(self._sample_paths, detections))
        output["targets"] = dict(zip(self._sample_paths, targets))
        output["class_to_idx"] = self._class_to_idx

        if settings.RESULTS_DIR.exists() is False:
            logger.info(f"Creating {settings.RESULTS_DIR} directory...")
            settings.RESULTS_DIR.mkdir(parents=True)

        results_path = settings.RESULTS_DIR.joinpath(name)
        logger.info(f"Saving results at {results_path}")

        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

    @staticmethod
    def load(path: str) -> "Results":
        """
        Load results object from file

        Parameters
        ----------
        path
            path to load from.
        """

        # Read label names
        with open(path, "r", encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)

        targets = data.pop("targets")
        class_to_idx = data.pop("class_to_idx")

        sample_paths = list(data.keys())
        detections = [{k: torch.tensor(v) for k, v in detection.items()} for detection in data.values()]
        targets = [{k: torch.tensor(v) for k, v in target.items()} for target in targets.values()]

        return Results(sample_paths, targets, detections, class_to_idx=class_to_idx)
