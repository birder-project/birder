import logging
import os
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from birder.conf import settings


def top_k_accuracy_score(y_true: npt.NDArray[Any], y_pred: npt.NDArray[np.float64], top_k: int) -> list[int]:
    """
    Returns all the sample indices which are in the top-k predictions
    """

    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    (num_samples, _num_labels) = y_pred.shape
    indices: list[int] = []
    # arg_sorted = np.argsort(y_pred, axis=1)[:, -top_k:]
    arg_sorted = np.argpartition(y_pred, -top_k, axis=1)[:, -top_k:]
    for i in range(num_samples):
        if y_true[i] in arg_sorted[i]:
            indices.append(i)

    return indices


class Results:
    """
    Classification result analysis class
    """

    num_desc_cols = 4

    def __init__(
        self,
        sample_list: list[str],
        labels: list[int],
        label_names: list[str],
        output: list[npt.NDArray[np.float32]],
        predictions: Optional[npt.NDArray[np.int_]] = None,
    ):
        """
        Initialize a result object

        Parameters
        ----------
        sample_list
            Sample names.
        labels
            The ground truth labels per sample.
        label_names
            Label names by order.
        output
            Probability of each class for each sample.
        predictions
            Prediction of each sample.
        """

        assert len(label_names) == len(output[0]), "Model output and label name list do not match"
        assert len(sample_list) == len(output), "Each output must have a sample name"

        names = [label_names[label] if label != -1 else "" for label in labels]
        self._label_names = label_names

        output_df = pd.DataFrame(output, columns=np.arange(len(output[0])).tolist(), dtype=np.float64)
        if predictions is None:
            predictions = output_df.idxmax(axis=1)

        self._results_df = pd.DataFrame(
            np.column_stack([sample_list, labels, names, predictions]),
            columns=["sample", "label", "label_name", "prediction"],
        )
        self._results_df = pd.concat([self._results_df, output_df], axis="columns")
        self._results_df = self._results_df.astype({"label": int, "prediction": int})
        self._results_df = self._results_df.sort_values(by="sample", ascending=True).reset_index(drop=True)

        if np.all(self.labels == -1) is np.True_:
            self.missing_all_labels = True
        else:
            self.missing_all_labels = False

        # Calculate metrics
        if self.missing_all_labels is False:
            self.valid_idx = self.labels != -1
            self._valid_length: int = np.sum(self.valid_idx)
            accuracy: int = int(
                accuracy_score(self.labels[self.valid_idx], self.predictions[self.valid_idx], normalize=False)
            )
            self._num_mistakes = self._valid_length - accuracy
            self._accuracy = accuracy / self._valid_length

            self._top_k_indices = top_k_accuracy_score(
                self.labels[self.valid_idx], self.output[self.valid_idx], top_k=settings.TOP_K
            )
            self._num_out_of_top_k = self._valid_length - len(self._top_k_indices)
            self._top_k = len(self._top_k_indices) / self._valid_length

            self._confusion_matrix = confusion_matrix(self.labels, self.predictions)

    def __len__(self) -> int:
        return len(self._results_df)

    def __repr__(self) -> str:
        head = self.__class__.__name__
        body = [
            f"Number of samples: {len(self)}",
            f"Number of valid samples: {self._valid_length}",
        ]

        if self.missing_all_labels is False:
            body.append(f"Accuracy: {self.accuracy:.3f}")

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)

    @property
    def labels(self) -> npt.NDArray[np.int_]:
        return self._results_df["label"].values  # type: ignore

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def unique_labels(self) -> npt.NDArray[np.int_]:
        return unique_labels(self.labels, self.predictions)  # type: ignore

    @property
    def missing_labels(self) -> bool:
        if -1 in self.labels:
            return True

        return False

    @property
    def output(self) -> npt.NDArray[np.float64]:
        return self.output_df.values  # type: ignore

    @property
    def output_df(self) -> pd.DataFrame:
        return self._results_df.iloc[:, Results.num_desc_cols :]

    @property
    def predictions(self) -> npt.NDArray[np.float64]:
        return self._results_df["prediction"].values  # type: ignore

    @property
    def prediction_names(self) -> pd.Series:
        prediction_names = pd.Series(self._label_names)
        return prediction_names[np.argmax(self.output, axis=1)]

    @property
    def mistakes(self) -> pd.DataFrame:
        return self._results_df[self._results_df["label"].values != np.argmax(self.output, axis=1)]

    @property
    def out_of_top_k(self) -> pd.DataFrame:
        return self._results_df[~self._results_df.index.isin(self._top_k_indices)]

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def top_k(self) -> float:
        return self._top_k

    @property
    def macro_f1_score(self) -> float:
        report_df = self.detailed_report()
        return report_df["F1-Score"].mean()  # type: ignore

    @property
    def confusion_matrix(self) -> npt.NDArray[np.int_]:
        return self._confusion_matrix  # type: ignore

    def get_as_df(self) -> pd.DataFrame:
        return self._results_df.copy()

    def detailed_report(self) -> pd.DataFrame:
        """
        Returns a detailed classification report with per-class metrics
        """

        raw_report_dict: dict[str, dict[str, float]] = classification_report(
            self.labels[self.valid_idx], self.predictions[self.valid_idx], output_dict=True, zero_division=0
        )
        del raw_report_dict["accuracy"]
        del raw_report_dict["macro avg"]
        del raw_report_dict["weighted avg"]

        row_list = []
        for class_idx, metrics in raw_report_dict.items():
            class_num = int(class_idx)

            # Skip metrics on classes we did not predict
            if metrics["support"] == 0:
                continue

            # Get label name
            label_name = self._label_names[class_num]

            # Calculate additional metrics
            item_index = np.where(self.unique_labels == class_num)[0][0]
            false_negative = (
                np.sum(self._confusion_matrix[item_index, :]) - self._confusion_matrix[item_index][item_index]
            )
            false_positive = (
                np.sum(self._confusion_matrix[:, item_index]) - self._confusion_matrix[item_index][item_index]
            )

            # Save metrics
            row: dict[str, int | float | str] = {}
            row["Class"] = class_num
            row["Class name"] = label_name
            row["Precision"] = metrics["precision"]
            row["Recall"] = metrics["recall"]
            row["F1-Score"] = metrics["f1-score"]
            row["Samples"] = metrics["support"]
            row["False negative"] = false_negative
            row["False positive"] = false_positive
            row_list.append(row)

        report_df = pd.DataFrame(row_list)
        report_df = report_df.astype(
            {
                "Class": int,
                "Class name": str,
                "Precision": float,
                "Recall": float,
                "F1-Score": float,
                "Samples": int,
                "False negative": int,
                "False positive": int,
            }
        )

        return report_df

    def log_short_report(self) -> None:
        """
        Log using the Python logging module a short metrics summary
        """

        report_df = self.detailed_report()
        lowest_precision = report_df.iloc[report_df["Precision"].argmin()]
        lowest_recall = report_df.iloc[report_df["Recall"].argmin()]
        highest_precision = report_df.iloc[report_df["Precision"].argmax()]
        highest_recall = report_df.iloc[report_df["Recall"].argmax()]

        logging.info(f"Accuracy {self._accuracy:.3f} on {self._valid_length} samples ({self._num_mistakes} mistakes)")
        logging.info(
            f"Top-{settings.TOP_K} accuracy {self._top_k:.3f} on {self._valid_length} samples "
            f"({self._num_out_of_top_k} samples out of top-{settings.TOP_K})"
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

        logging.info(
            f"Highest precision {highest_precision['Precision']:.3f} for '{highest_precision['Class name']}' "
            f"({highest_precision['False negative']} false negatives, "
            f"{highest_precision['False positive']} false positives)"
        )
        logging.info(
            f"Highest recall {highest_recall['Recall']:.3f} for '{highest_recall['Class name']}' "
            f"({highest_recall['False negative']} false negatives, "
            f"{highest_recall['False positive']} false positives)"
        )
        if self.missing_labels is True:
            logging.warning(
                f"{len(self) - self._valid_length} of the samples did not have labels, metrics calculated only on "
                f"{self._valid_length} out of total {len(self)} samples"
            )

    def pretty_print(self) -> None:
        console = Console()

        table = Table(show_header=True, header_style="bold dark_magenta")
        table.add_column("Class")
        table.add_column("Class name", style="dim")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-Score", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("False negative", justify="right")
        table.add_column("False positive", justify="right")

        report_df = self.detailed_report()
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
        accuracy_text.append(f"Accuracy {self._accuracy:.3f} on {self._valid_length} samples (")
        accuracy_text.append(f"{self._num_mistakes}", style="bold")
        accuracy_text.append(" mistakes)")

        top_k_text = Text()
        top_k_text.append(f"Top-{settings.TOP_K} accuracy {self._top_k:.3f} on {self._valid_length} samples (")
        top_k_text.append(f"{self._num_out_of_top_k}", style="bold")
        top_k_text.append(f" samples out of top-{settings.TOP_K})")

        console.print(accuracy_text)
        console.print(top_k_text)
        if self.missing_labels is True:
            console.print(
                "[bold][bright_red]NOTICE[/bright_red][/bold]: "
                f"{len(self) - self._valid_length} of the samples did not have labels, metrics calculated only on "
                f"{self._valid_length} out of total {len(self)} samples"
            )

    def save(self, path: str) -> None:
        """
        Save results object to file

        Parameters
        ----------
        path
            file output path.
        """

        if settings.RESULTS_DIR.exists() is False:
            logging.info(f"Creating {settings.RESULTS_DIR} directory...")
            settings.RESULTS_DIR.mkdir(parents=True)

        results_path = settings.RESULTS_DIR.joinpath(path)
        logging.info(f"Saving results at {results_path}")

        # Write label names list
        with open(results_path, "w", encoding="utf-8") as handle:
            handle.write("," * Results.num_desc_cols)
            handle.write(",".join(self._label_names))
            handle.write(os.linesep)

        # Write the data frame
        self._results_df.to_csv(results_path, index=False, mode="a")

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
            label_names = handle.readline().rstrip(os.linesep).split(",")
            label_names = label_names[Results.num_desc_cols :]

        # Read the data frame
        results_df = pd.read_csv(path, skiprows=1)
        return Results(
            results_df["sample"].values,
            results_df["label"].values,
            label_names,
            results_df.iloc[:, Results.num_desc_cols :].values,
            results_df["prediction"],
        )
