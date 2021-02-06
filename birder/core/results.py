from __future__ import annotations  # PEP 563 (required until Python 3.9)

import logging
import os
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from birder.conf.settings import TOP_K


# Remove when https://github.com/scikit-learn/scikit-learn/pull/10488/ is merged
def top_k_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 5) -> int:
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    (num_obs, num_labels) = y_pred.shape
    idx = num_labels - top_k - 1
    counter = 0
    argsorted = np.argsort(y_pred, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1 :]:
            counter += 1

    return counter


class Results:
    num_desc_cols = 3
    results_dir = "results/"

    def __init__(
        self,
        image_list: List[str],
        labels: List[int],
        label_names: List[str],
        output: List[np.ndarray],
        results_df: Optional[pd.DataFrame] = None,
    ):
        """
        TODO: label names should be unique with name at index 'label'
        """

        self._label_names = label_names

        if results_df is None:
            assert len(label_names) == len(output[0]), "Must provide all label names"

            synset = [label_names[label] if label != -1 else "" for label in labels]
            self._results_df = np.column_stack([labels, output])
            self._results_df = pd.DataFrame(
                self._results_df, columns=["label"] + np.arange(len(output[0])).tolist()
            )
            self._results_df = self._results_df.astype({"label": int})
            self._results_df.insert(0, "image", image_list)
            self._results_df.insert(2, "label_name", synset)

        else:
            assert "image" in results_df
            assert (
                len(label_names) == len(results_df.columns) - Results.num_desc_cols
            ), "Must provide all label names"

            self._results_df = results_df

        # Calculate metrics
        accuracy: int = accuracy_score(self.labels, self.predictions, normalize=False)
        self._num_mistakes = len(self) - accuracy
        self._accuracy = accuracy / len(self)

        top_k: int = top_k_accuracy_score(self.labels, self.output, top_k=TOP_K)
        self._out_of_top_k = len(self) - top_k
        self._top_k = top_k / len(self)

        self._confusion_matrix = confusion_matrix(self.labels, self.predictions)

    @property
    def df(self) -> pd.DataFrame:
        return self._results_df

    @property
    def labels(self) -> np.ndarray:
        return self._results_df["label"].values

    @property
    def label_names(self) -> List[str]:
        return self._label_names

    @property
    def unique_labels(self) -> np.ndarray:
        return unique_labels(self.labels, self.predictions)

    @property
    def missing_labels(self) -> bool:
        if -1 in self.labels:
            return True

        return False

    @property
    def output(self) -> np.ndarray:
        return self._results_df.iloc[:, Results.num_desc_cols :].values

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.output, axis=1)

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def top_k(self) -> float:
        return self._top_k

    @property
    def num_mistakes(self) -> int:
        return self._num_mistakes

    @property
    def out_of_top_k(self) -> int:
        return self._out_of_top_k

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._confusion_matrix

    @property
    def detailed_report(self) -> pd.DataFrame:
        raw_report_dict: Dict[str, Dict[str, float]] = classification_report(
            self.labels, self.predictions, output_dict=True, zero_division=0
        )
        del raw_report_dict["accuracy"]
        del raw_report_dict["macro avg"]
        del raw_report_dict["weighted avg"]

        report_df = pd.DataFrame(
            columns=[
                "Class",
                "Class name",
                "Precision",
                "Recall",
                "F1-Score",
                "Samples",
                "False negative",
                "False positive",
            ]
        )
        report_df = report_df.astype(
            {
                "Class": np.int,
                "Class name": str,
                "Precision": np.float,
                "Recall": np.float,
                "F1-Score": np.float,
                "Samples": np.int,
                "False negative": np.int,
                "False positive": np.int,
            }
        )
        for class_idx, metrics in raw_report_dict.items():
            class_num = int(class_idx)

            # Skip metrics on classes we did not predict
            if metrics["support"] == 0:
                continue

            # Get label name
            label_name = self._label_names[class_num]

            # Calculate additional metrics
            itemindex = np.where(self.unique_labels == class_num)[0][0]
            false_negative = (
                np.sum(self.confusion_matrix[itemindex]) - self.confusion_matrix[itemindex][itemindex]
            )
            false_positive = (
                np.sum(self.confusion_matrix[:, itemindex]) - self.confusion_matrix[itemindex][itemindex]
            )

            # Save metrics
            row: Dict[str, Union[int, float, str]] = {}
            row["Class"] = class_num
            row["Class name"] = label_name
            row["Precision"] = metrics["precision"]
            row["Recall"] = metrics["recall"]
            row["F1-Score"] = metrics["f1-score"]
            row["Samples"] = metrics["support"]
            row["False negative"] = false_negative
            row["False positive"] = false_positive
            report_df = report_df.append(row, ignore_index=True)

        return report_df

    def log_short_report(self) -> None:
        report_df = self.detailed_report
        lowest_precision = report_df.iloc[report_df["Precision"].argmin()]
        lowest_recall = report_df.iloc[report_df["Recall"].argmin()]

        logging.info(f"Accuracy {self.accuracy:.3f} on {len(self)} samples ({self.num_mistakes} mistakes)")
        logging.info(
            f"Top-{TOP_K} accuracy {self.top_k:.3f} on {len(self)} samples "
            f"({self.out_of_top_k} samples out of top-{TOP_K})"
        )
        logging.info(
            f"Lowest precision {lowest_precision['Precision']:.3f} for '{lowest_precision['Class name']}' "
            f"({lowest_precision['False negative']} false negatives, "
            f"{lowest_precision['False positive']} false positives)"
        )
        logging.info(
            f"Lowest recall {lowest_recall['Recall']:.3f} for '{lowest_recall['Class name']}' "
            f"({lowest_recall['False negative']} false negatives, "
            f"{lowest_recall['False positive']} false positives)"
        )

    def pretty_print(self) -> None:
        console = Console()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Class")
        table.add_column("Class name", style="dim")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("False negative", justify="right")
        table.add_column("False positive", justify="right")

        report_df = self.detailed_report
        fn_cutoff = report_df["False negative"].quantile(0.95)
        fp_cutoff = report_df["False positive"].quantile(0.95)

        for _, row in report_df.iterrows():
            recall_msg = f"{row['Recall']:.3f}"
            if row["Recall"] < 0.75:
                recall_msg = "[red1]" + recall_msg + "[/red1]"

            elif row["Recall"] < 0.9:
                recall_msg = "[dark_orange]" + recall_msg + "[/dark_orange]"

            f1_msg = f"{row['F1-Score']:.3f}"
            if row["F1-Score"] == 1.0:
                f1_msg = "[green]" + f1_msg + "[/green]"

            fn_msg = f"{row['False negative']}"
            if row["False negative"] > fn_cutoff:
                fn_msg = "[underline]" + fn_msg + "[/underline]"

            fp_msg = f"{row['False positive']}"
            if row["False positive"] > fp_cutoff:
                fp_msg = "[underline]" + fp_msg + "[/underline]"

            table.add_row(
                f"{row['Class']}",
                row["Class name"],
                f"{row['Precision']:.3f}",
                recall_msg,
                f1_msg,
                f"{row['Samples']}",
                fn_msg,
                fp_msg,
            )

        console.print("'False negative' is a simple mistake in the context of multi-class classification")
        console.print(
            "Per-class 'recall' is the equivalent of 'per-class accuracy' "
            "in the context of multi-class classification"
        )
        console.print(table)

        accuracy_text = Text()
        accuracy_text.append(f"Accuracy {self.accuracy:.3f} on {len(self)} samples (")
        accuracy_text.append(f"{self.num_mistakes}", style="bold")
        accuracy_text.append(" mistakes)")

        top_k_text = Text()
        top_k_text.append(f"Top-{TOP_K} accuracy {self.top_k:.3f} on {len(self)} samples (")
        top_k_text.append(f"{self.out_of_top_k}", style="bold")
        top_k_text.append(f" samples out of top-{TOP_K})")

        console.print(accuracy_text)
        console.print(top_k_text)

    def save(self, path: str) -> None:
        results_path = os.path.join(Results.results_dir, path)
        logging.info(f"Saving results at {results_path}")
        os.makedirs(Results.results_dir, exist_ok=True)

        # Write label names list
        with open(results_path, "w") as handle:
            handle.write(",".join(self.label_names))
            handle.write(os.linesep)

        # Write the dataframe
        self._results_df.to_csv(results_path, index=False, mode="a")

    @staticmethod
    def load(path: str) -> Results:
        # Read label names
        with open(path, "r") as handle:
            label_names = handle.readline().rstrip(os.linesep).split(",")

        # Read the dataframe
        results_df = pd.read_csv(path, skiprows=1)
        return Results([], [], label_names, [], results_df=results_df)

    def __len__(self) -> int:
        return len(self._results_df)
