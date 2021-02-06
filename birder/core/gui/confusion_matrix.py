import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from birder.core import Results


class ConfusionMatrix:
    def __init__(self, results: Results):
        """
        TODO: Explain class_names
        """

        self.results = results

    def save(self, path: str) -> None:
        """
        Save confusion matrix as CSV
        """

        logging.info(f"Saving confusion matrix at {path}")
        class_names = [self.results.label_names[label_idx] for label_idx in self.results.unique_labels]
        cnf = pd.DataFrame(self.results.confusion_matrix, index=class_names, columns=class_names)
        cnf.to_csv(path)

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
        ax.set_title(f"Confusion matrix, accuracy {self.results.accuracy:.2f} on {len(self.results)} samples")
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
