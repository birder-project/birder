import json
import logging
from collections import Counter
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypedDict

import polars as pl
import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text
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

        self._iou_thresholds = metrics.iou_thresholds
        self._class_to_idx = class_to_idx
        self._label_names = ["Background"] + list(class_to_idx.keys())
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
        body.append(f"mAP: {self.map:.4f}")

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)

    @property
    def map(self) -> float:
        return self.metrics_dict["map"]

    def detailed_report(self) -> pl.DataFrame:
        """
        Returns a detailed detection report with per-class metrics
        """

        object_count: Counter[int] = Counter()
        for t in self._targets:
            object_count.update(t["labels"].tolist())

        row_list = []
        for class_num, mean_ap in zip(self.metrics_dict["classes"], self.metrics_dict["map_per_class"]):
            if mean_ap < 0:
                continue

            row_list.append(
                {
                    "Class": class_num,
                    "Class name": self._label_names[class_num],
                    "mAP": mean_ap,
                    "Objects": object_count[class_num],
                }
            )

        return pl.DataFrame(row_list)

    def log_short_report(self) -> None:
        """
        Log using the Python logging module a short metrics summary
        """

        report_df = self.detailed_report()
        total_objects = report_df["Objects"].sum()
        lowest_map = report_df[report_df["mAP"].arg_min()]  # type: ignore[index]
        highest_map = report_df[report_df["mAP"].arg_max()]  # type: ignore[index]

        logger.info(f"mAP {self.map:.4f} on {len(self)} images with {total_objects} objects")
        logger.info(f"Lowest mAP {lowest_map['mAP'][0]:.4f} for '{lowest_map['Class name'][0]}'")
        logger.info(f"Highest mAP {highest_map['mAP'][0]:.4f} for '{highest_map['Class name'][0]}'")

    def pretty_print(
        self,
        sort_by: Literal["class", "map"] = "class",
        order: Literal["ascending", "descending"] = "ascending",
        n: Optional[int] = None,
    ) -> None:
        console = Console()

        table = Table(show_header=True, header_style="bold dark_magenta")
        table.add_column("Class")
        table.add_column("Class name", style="dim")
        table.add_column("mAP", justify="right")
        table.add_column("Objects", justify="right")

        report_df = self.detailed_report()
        total_objects = report_df["Objects"].sum()
        if sort_by == "map":
            sort_column = "mAP"
        else:
            sort_column = sort_by.capitalize()

        report_df = report_df.sort(sort_column, descending=order == "descending")
        if n is not None:
            report_df = report_df[:n]

        for row in report_df.iter_rows(named=True):
            map_msg = f"{row['mAP']:.4f}"
            if row["mAP"] < 0.4:
                map_msg = "[red1]" + map_msg + "[/red1]"
            elif row["mAP"] < 0.5:
                map_msg = "[dark_orange]" + map_msg + "[/dark_orange]"

            table.add_row(
                f"{row['Class']}",
                row["Class name"],
                map_msg,
                f"{row['Objects']}",
            )

        console.print(table)

        map_text = Text()
        map_text.append(f"mAP {self.map:.4f} on {len(self)} images with {total_objects} objects")

        console.print(map_text)

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
