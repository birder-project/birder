import logging
import os
from functools import cached_property
from typing import Any
from typing import Literal
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.text import Text
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from birder.conf import settings

logger = logging.getLogger(__name__)


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


# pylint: disable=too-many-public-methods
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
        output: npt.NDArray[np.float32],
        predictions: Optional[npt.NDArray[np.int_]] = None,
        *,
        lazy: bool = True,
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
        lazy
            If True, metrics and computed properties will be lazily evaluated (computed only when accessed).
            If False, all metrics will be computed during initialization.
        """

        assert len(label_names) == len(output[0]), "Model output and label name list do not match"
        assert len(sample_list) == len(output), "Each output must have a sample name"

        if predictions is None:
            predictions = output.argmax(axis=1)

        names = [label_names[label] if label != -1 else "" for label in labels]
        self._label_names = label_names

        output_df = pl.DataFrame(
            {
                **{f"{i}": output[:, i] for i in range(output.shape[-1])},
            }
        )

        self._results_df = pl.DataFrame(
            {"sample": sample_list, "label": labels, "label_name": names, "prediction": predictions}
        )
        self._results_df = pl.concat([self._results_df, output_df], how="horizontal")
        self._results_df = self._results_df.sort("sample", descending=False)

        if np.all(self.labels == -1) is np.True_:
            self.missing_all_labels = True
        else:
            self.missing_all_labels = False

        # Calculate metrics
        if self.missing_all_labels is False:
            self.valid_idx = self.labels != -1
            self._valid_length: int = np.sum(self.valid_idx).item()
            accuracy: int = int(
                accuracy_score(self.labels[self.valid_idx], self.predictions[self.valid_idx], normalize=False)
            )
            self._num_mistakes = self._valid_length - accuracy
            self._accuracy = accuracy / self._valid_length

            if lazy is False:
                _ = self.top_k
                _ = self.confusion_matrix

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

    @cached_property
    def _top_k_indices(self) -> list[int]:
        return top_k_accuracy_score(self.labels[self.valid_idx], self.output[self.valid_idx], top_k=settings.TOP_K)

    @property
    def labels(self) -> npt.NDArray[np.int_]:
        return self._results_df["label"].to_numpy()

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @cached_property
    def unique_labels(self) -> npt.NDArray[np.int_]:
        return (  # type: ignore
            np.unique(np.concatenate([self.labels[self.valid_idx], self.predictions[self.valid_idx]], axis=0))
        )

    @property
    def missing_labels(self) -> bool:
        if -1 in self.labels:
            return True

        return False

    @property
    def output(self) -> npt.NDArray[np.float64]:
        return self.output_df.to_numpy()

    @property
    def output_df(self) -> pl.DataFrame:
        return self._results_df.select(pl.all().exclude(self._results_df.columns[: Results.num_desc_cols]))

    @property
    def predictions(self) -> npt.NDArray[np.int_]:
        return self._results_df["prediction"].to_numpy()

    @property
    def prediction_names(self) -> pl.Series:
        prediction_names = pl.Series("prediction_names", self._label_names)
        return prediction_names[np.argmax(self.output, axis=1)]  # type: ignore

    @property
    def mistakes(self) -> pl.DataFrame:
        return self._results_df.filter(pl.col("label") != pl.col("prediction"))

    @property
    def out_of_top_k(self) -> pl.DataFrame:
        return self._results_df.with_row_index().filter(~pl.col("index").is_in(self._top_k_indices)).drop("index")

    @property
    def num_out_of_top_k(self) -> int:
        return self._valid_length - len(self._top_k_indices)

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @cached_property
    def top_k(self) -> float:
        return len(self._top_k_indices) / self._valid_length

    @property
    def macro_f1_score(self) -> float:
        return f1_score(self.labels[self.valid_idx], self.predictions[self.valid_idx], average="macro")  # type: ignore

    @cached_property
    def confusion_matrix(self) -> npt.NDArray[np.int_]:
        return confusion_matrix(self.labels[self.valid_idx], self.predictions[self.valid_idx])  # type: ignore

    def most_confused(self, n: int = 10) -> pl.DataFrame:
        cnf_matrix = self.confusion_matrix.copy()
        np.fill_diagonal(cnf_matrix, -1)
        top_indices = np.argsort(cnf_matrix.ravel())[-n:][::-1]
        class_names = [self.label_names[label_idx] for label_idx in self.unique_labels]

        data = []
        for flat_idx in top_indices:
            idx = np.unravel_index(flat_idx, cnf_matrix.shape)
            if cnf_matrix[idx] == 0:
                break

            data.append(
                {
                    "predicted": class_names[idx[1]],
                    "actual": class_names[idx[0]],
                    "amount": cnf_matrix[idx],
                    "reverse": cnf_matrix[idx[::-1]],
                }
            )

        return pl.DataFrame(data)

    def get_as_df(self) -> pl.DataFrame:
        return self._results_df.clone()

    def detailed_report(self) -> pl.DataFrame:
        """
        Returns a detailed classification report with per-class metrics
        """

        raw_report_dict: dict[str, dict[str, float]] = classification_report(
            self.labels[self.valid_idx], self.predictions[self.valid_idx], output_dict=True, zero_division=0
        )
        del raw_report_dict["accuracy"]
        del raw_report_dict["macro avg"]
        del raw_report_dict["weighted avg"]

        # Pre-compute row and column sums for the confusion matrix
        cm_row_sums = np.sum(self.confusion_matrix, axis=1)
        cm_col_sums = np.sum(self.confusion_matrix, axis=0)

        row_list = []
        for class_idx, metrics in raw_report_dict.items():
            class_num = int(class_idx)

            # Skip metrics on classes we did not predict
            if metrics["support"] == 0:
                continue

            # Cast to int
            metrics["support"] = int(metrics["support"])

            # Get label name
            label_name = self._label_names[class_num]

            # Calculate additional metrics
            item_index = np.asarray(self.unique_labels == class_num).nonzero()[0][0]
            false_negative = cm_row_sums[item_index] - self.confusion_matrix[item_index][item_index]
            false_positive = cm_col_sums[item_index] - self.confusion_matrix[item_index][item_index]

            # Save metrics
            row_list.append(
                {
                    "Class": class_num,
                    "Class name": label_name,
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-score": metrics["f1-score"],
                    "Samples": metrics["support"],
                    "False negative": false_negative,
                    "False positive": false_positive,
                }
            )

        report_df = pl.DataFrame(row_list)

        return report_df

    def log_short_report(self) -> None:
        """
        Log using the Python logging module a short metrics summary
        """

        report_df = self.detailed_report()
        lowest_precision = report_df[report_df["Precision"].arg_min()]  # type: ignore[index]
        lowest_recall = report_df[report_df["Recall"].arg_min()]  # type: ignore[index]
        highest_precision = report_df[report_df["Precision"].arg_max()]  # type: ignore[index]
        highest_recall = report_df[report_df["Recall"].arg_max()]  # type: ignore[index]

        logger.info(f"Accuracy {self.accuracy:.3f} on {self._valid_length} samples ({self._num_mistakes} mistakes)")
        logger.info(
            f"Top-{settings.TOP_K} accuracy {self.top_k:.3f} on {self._valid_length} samples "
            f"({self.num_out_of_top_k} samples out of top-{settings.TOP_K})"
        )

        logger.info(
            f"Lowest precision {lowest_precision['Precision'][0]:.3f} for '{lowest_precision['Class name'][0]}' "
            f"({lowest_precision['False negative'][0]} false negatives, "
            f"{lowest_precision['False positive'][0]} false positives)"
        )
        logger.info(
            f"Lowest recall {lowest_recall['Recall'][0]:.3f} for '{lowest_recall['Class name'][0]}' "
            f"({lowest_recall['False negative'][0]} false negatives, "
            f"{lowest_recall['False positive'][0]} false positives)"
        )

        logger.info(
            f"Highest precision {highest_precision['Precision'][0]:.3f} for '{highest_precision['Class name'][0]}' "
            f"({highest_precision['False negative'][0]} false negatives, "
            f"{highest_precision['False positive'][0]} false positives)"
        )
        logger.info(
            f"Highest recall {highest_recall['Recall'][0]:.3f} for '{highest_recall['Class name'][0]}' "
            f"({highest_recall['False negative'][0]} false negatives, "
            f"{highest_recall['False positive'][0]} false positives)"
        )
        if self.missing_labels is True:
            logger.warning(
                f"{len(self) - self._valid_length} of the samples did not have labels, metrics calculated only on "
                f"{self._valid_length} out of total {len(self)} samples"
            )

    def pretty_print(
        self,
        sort_by: Literal["class", "precision", "recall", "f1-score"] = "class",
        order: Literal["ascending", "descending"] = "ascending",
        n: Optional[int] = None,
    ) -> None:
        console = Console()

        table = Table(show_header=True, header_style="bold dark_magenta")
        table.add_column("Class")
        table.add_column("Class name", style="dim")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-score", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("False negative", justify="right")
        table.add_column("False positive", justify="right")

        report_df = self.detailed_report()
        report_df = report_df.sort(sort_by.capitalize(), descending=order == "descending")
        if n is not None:
            report_df = report_df[:n]

        fn_cutoff = report_df["False negative"].quantile(0.95)
        fp_cutoff = report_df["False positive"].quantile(0.95)

        for row in report_df.iter_rows(named=True):
            recall_msg = f"{row['Recall']:.3f}"
            if row["Recall"] < 0.75:
                recall_msg = "[red1]" + recall_msg + "[/red1]"

            elif row["Recall"] < 0.9:
                recall_msg = "[dark_orange]" + recall_msg + "[/dark_orange]"

            f1_msg = f"{row['F1-score']:.3f}"
            if row["F1-score"] == 1.0:
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

        console.print(table)

        accuracy_text = Text()
        accuracy_text.append(f"Accuracy {self.accuracy:.3f} on {self._valid_length} samples (")
        accuracy_text.append(f"{self._num_mistakes}", style="bold")
        accuracy_text.append(" mistakes)")

        top_k_text = Text()
        top_k_text.append(f"Top-{settings.TOP_K} accuracy {self.top_k:.3f} on {self._valid_length} samples (")
        top_k_text.append(f"{self.num_out_of_top_k}", style="bold")
        top_k_text.append(f" samples out of top-{settings.TOP_K})")

        console.print(accuracy_text)
        console.print(top_k_text)
        if self.missing_labels is True:
            console.print(
                "[bold][bright_red]NOTICE[/bright_red][/bold]: "
                f"{len(self) - self._valid_length} of the samples did not have labels, metrics calculated only on "
                f"{self._valid_length} out of total {len(self)} samples"
            )

    def save(self, name: str, append: bool = False) -> None:
        """
        Save results object to file

        Parameters
        ----------
        name
            Output file name.
        append
            Append result data to existing results file.
        """

        if settings.RESULTS_DIR.exists() is False:
            logger.info(f"Creating {settings.RESULTS_DIR} directory...")
            settings.RESULTS_DIR.mkdir(parents=True)

        results_path = settings.RESULTS_DIR.joinpath(name)
        if append is False:
            logger.info(f"Saving results at {results_path}")

            # Write label names list
            with open(results_path, "w", encoding="utf-8") as handle:
                handle.write("," * Results.num_desc_cols)
                handle.write(",".join(self._label_names))
                handle.write(os.linesep)

            # Write the data frame
            with open(results_path, "a", encoding="utf-8") as handle:
                self._results_df.write_csv(handle)
        else:
            logger.info(f"Adding results to {results_path}")
            with open(results_path, "a", encoding="utf-8") as handle:
                self._results_df.write_csv(handle, include_header=False)

    @staticmethod
    def load(path: str) -> "Results":
        """
        Load results object from file

        Parameters
        ----------
        path
            Path to load from.
        """

        # Read label names
        with open(path, "r", encoding="utf-8") as handle:
            label_names = handle.readline().rstrip(os.linesep).split(",")
            label_names = label_names[Results.num_desc_cols :]

        # Read the data frame
        results_df = pl.read_csv(path, skip_rows=1)
        return Results(
            results_df["sample"].to_list(),
            results_df["label"].to_list(),
            label_names,
            results_df[:, Results.num_desc_cols :].to_numpy(),
            results_df["prediction"].to_numpy(),
        )


def compare_results(results_dict: dict[str, Results]) -> pl.DataFrame:
    result_list = []
    for name, results in results_dict.items():
        result_list.append(
            {
                "File name": name,
                "Accuracy": results.accuracy,
                f"Top-{settings.TOP_K} accuracy": results.top_k,
                "Macro F1-score": results.macro_f1_score,
                "Samples": len(results),
                "Mistakes": results._num_mistakes,  # pylint: disable=protected-access
            }
        )

    return pl.DataFrame(result_list)
