import itertools
import logging
from functools import partial
from typing import Optional

import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from torchvision.datasets.folder import pil_loader

from birder.conf import settings
from birder.results.classification import Results

logger = logging.getLogger(__name__)


def show_top_k(
    image_path: str, out: npt.NDArray[np.float32], class_to_idx: dict[str, int], label: Optional[int | str] = None
) -> None:
    img = pil_loader(image_path)

    idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
    probabilities: list[float] = []
    predicted_class_names: list[str] = []

    logger.debug(f"'{image_path}'")
    for idx in np.argsort(out)[::-1][: settings.TOP_K]:
        probabilities.append(out[idx])
        predicted_class_names.append(idx_to_class[idx])
        logger.debug(f"{idx_to_class[idx]:<25}: {out[idx]:.4f}")

    logger.debug("---")

    fig = plt.figure(num=image_path)
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(settings.TOP_K)
    bars = ax.barh(y_pos, probabilities, alpha=0.4, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predicted_class_names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)

    if label is not None and label != -1:
        # 'label' could be any int-like object like NumPy, Torch, etc.
        if not isinstance(label, str):
            label = idx_to_class[label]

        for idx, class_name in enumerate(predicted_class_names):
            if label == class_name:
                bars[idx].set_color("green")

    plt.tight_layout()
    plt.show()


class ConfusionMatrix:
    def __init__(self, results: Results):
        self.results = results

    def save(self, path: str) -> None:
        """
        Save confusion matrix as a CSV file
        """

        cnf_df = pl.DataFrame(
            {
                "": self.results.label_names,
                **{name: self.results.confusion_matrix[:, i] for i, name in enumerate(self.results.label_names)},
            }
        )
        logger.info(f"Saving confusion matrix at {path}")
        cnf_df.write_csv(path)

    def show(self) -> None:
        """
        Show confusion matrix as matplotlib figure
        """

        # Define figure and axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)

        # Plot confusion matrix
        cnf_matrix = self.results.confusion_matrix
        cnf_ax = ax.imshow(cnf_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
        ax.set_title(f"Confusion matrix, accuracy {self.results.accuracy:.3f} on {len(self.results)} samples")
        plt.colorbar(cnf_ax)
        tick_marks = np.arange(len(self.results.unique_labels))
        class_names = [self.results.label_names[label_idx] for label_idx in self.results.unique_labels]

        # Set axis
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, minor=False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, minor=False)

        # Add text (matrix values)
        threshold = cnf_matrix.max() / 2.0
        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            if i == j and np.sum(cnf_matrix, axis=1)[i] == cnf_matrix[i, j]:
                text_color = "greenyellow"

            elif i == j and (np.sum(cnf_matrix, axis=1)[i] * 0.75) > cnf_matrix[i, j]:
                text_color = "red"

            elif i == j and (np.sum(cnf_matrix, axis=1)[i] * 0.9) > cnf_matrix[i, j]:
                text_color = "orange"

            elif cnf_matrix[i, j] > threshold:
                text_color = "white"

            else:
                text_color = "black"

            ax.text(
                j,
                i,
                cnf_matrix[i, j],
                verticalalignment="center",
                horizontalalignment="center",
                color=text_color,
                fontsize="small",
                clip_on=True,
            )

        offset = 0.5
        (height, width) = cnf_matrix.shape
        ax.hlines(
            y=np.arange(height + 1) - offset,
            xmin=-offset,
            xmax=width - offset,
            linestyles="dashed",
            colors="grey",
            linewidth=0.5,
        )
        ax.vlines(
            x=np.arange(width + 1) - offset,
            ymin=-offset,
            ymax=height - offset,
            linestyles="dashed",
            colors="grey",
            linewidth=0.5,
        )

        plt.tight_layout()
        plt.show()


class ROC:
    def __init__(self, result: Optional[Results] = None) -> None:
        self.results_dict: dict[str, Results] = {}
        if result is not None:
            self.results_dict["result"] = result

    def add_result(self, name: str, result: Results) -> None:
        self.results_dict[name] = result

    def show(self, roc_classes: Optional[list[str]] = None) -> None:
        """
        Show roc curve as matplotlib figure
        """

        if roc_classes is None:
            roc_classes = []

        # Define figure and axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)

        for name, results in self.results_dict.items():
            fpr = {}
            tpr = {}
            roc_auc = {}
            for i in results.unique_labels:
                binary_labels = results.labels == i
                (fpr[i], tpr[i], _) = roc_curve(binary_labels, results.output[:, i])
                if np.sum(binary_labels) == 0:
                    tpr[i] = np.zeros_like(fpr[i])

                roc_auc[i] = auc(fpr[i], tpr[i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in results.unique_labels]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in results.unique_labels:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= len(results.unique_labels)
            macro_auc = auc(all_fpr, mean_tpr)

            for cls in roc_classes:
                i = results.label_names.index(cls)
                ax.plot(fpr[i], tpr[i], label=f"{name} ROC curve for {cls} ({roc_auc[i]:.3f})")

            ax.plot(
                all_fpr,
                mean_tpr,
                linestyle=":",
                label=f"{name} Macro-average ROC curve ({macro_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


class PrecisionRecall:
    def __init__(self, result: Optional[Results] = None) -> None:
        self.results_dict: dict[str, Results] = {}
        if result is not None:
            self.results_dict["result"] = result

    def add_result(self, name: str, result: Results) -> None:
        self.results_dict[name] = result

    def show(self, pr_classes: Optional[list[str]] = None) -> None:
        """
        Show precision recall curve as matplotlib figure
        """

        if pr_classes is None:
            pr_classes = []

        # Define figure and axes
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)

        legend_lines = []
        legend_labels: list[str] = []
        for name, results in self.results_dict.items():
            labels = label_binarize(results.labels, classes=range(len(results.label_names)))

            # A "micro-average" quantifying score on all classes jointly
            (precision, recall, _) = precision_recall_curve(labels.ravel(), results.output.ravel())
            average_precision = average_precision_score(labels.ravel(), results.output.ravel(), average="micro")

            line = ax.step(recall, precision, linestyle=":", where="post")
            legend_lines.append(line[0])
            legend_labels.append(f"{name} micro-average precision-recall ({average_precision:.3f})")

            # Per selected class
            for cls in pr_classes:
                i = results.label_names.index(cls)
                (precision, recall, _) = precision_recall_curve(labels[:, i], results.output[:, i])
                average_precision = average_precision_score(labels[:, i], results.output[:, i])
                line = ax.plot(recall, precision, lw=2)
                legend_lines.append(line[0])
                legend_labels.append(f"{name} precision-recall for class {cls} ({average_precision:.3f})")

        # iso-f1 curves
        f_scores = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            line = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            ax.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

        legend_lines.append(line[0])
        legend_labels.append("iso-f1 curves")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim((0.0, 1.05))
        ax.set_xlim((0.0, 1.0))
        ax.set_title("Precision Recall Curves")
        plt.legend(legend_lines, legend_labels, loc="lower left")
        plt.tight_layout()
        plt.show()


class ProbabilityHistogram:
    def __init__(self, results: Results) -> None:
        self.results = results

    def show(self, cls_a: str, cls_b: str) -> None:
        results_df = self.results.get_as_df()
        hist = partial(np.histogram, bins=20, range=(0, 1), density=True)

        cls_a_df = results_df.filter(pl.col("label_name") == cls_a)
        cls_b_df = results_df.filter(pl.col("label_name") == cls_b)

        (cls_a_prob_a_counts, cls_a_prob_a_bins) = hist(cls_a_df[str(self.results.label_names.index(cls_a))])
        (cls_a_prob_b_counts, cls_a_prob_b_bins) = hist(cls_b_df[str(self.results.label_names.index(cls_a))])
        plt.subplot(2, 1, 1)
        plt.stairs(
            cls_a_prob_a_counts,
            cls_a_prob_a_bins,
            fill=True,
            alpha=0.4,
            label=f"{cls_a} prob. on {cls_a} label",
        )
        plt.stairs(
            cls_a_prob_b_counts,
            cls_a_prob_b_bins,
            fill=True,
            alpha=0.4,
            label=f"{cls_a} prob. on {cls_b} label",
        )
        plt.legend(loc="upper center")

        (cls_b_prob_a_counts, cls_b_prob_a_bins) = hist(cls_a_df[str(self.results.label_names.index(cls_b))])
        (cls_b_prob_b_counts, cls_b_prob_b_bins) = hist(cls_b_df[str(self.results.label_names.index(cls_b))])
        plt.subplot(2, 1, 2)
        plt.stairs(
            cls_b_prob_b_counts,
            cls_b_prob_b_bins,
            fill=True,
            alpha=0.4,
            label=f"{cls_b} prob. on {cls_b} label",
        )
        plt.stairs(
            cls_b_prob_a_counts,
            cls_b_prob_a_bins,
            fill=True,
            alpha=0.4,
            label=f"{cls_b} prob. on {cls_a} label",
        )

        plt.legend(loc="upper center")
        plt.tight_layout()
        plt.show()
