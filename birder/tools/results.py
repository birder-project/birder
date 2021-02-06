import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from birder.common import util
from birder.core.gui import ConfusionMatrix
from birder.core.results import Results


def prep_results(results: Results, cnf_classes: List[int]) -> Results:
    if len(cnf_classes) > 0:
        results_df = results.df[results.df["label"].isin(cnf_classes)]
        results = Results([], [], results.label_names, [], results_df=results_df)

    return results


def plot_confusion_matrix(results: Results, cnf_classes: List[int]) -> None:
    results = prep_results(results, cnf_classes)
    cnf = ConfusionMatrix(results)
    cnf.show()


def save_confusion_matrix(results: Results, cnf_classes: List[int], results_file: str) -> None:
    results = prep_results(results, cnf_classes)
    cnf = ConfusionMatrix(results)

    filename = os.path.splitext(os.path.basename(results_file))[0]
    filename = f"{filename}_confusion_matrix.csv"
    cnf_path = os.path.join(os.path.dirname(results_file), filename)
    cnf.save(cnf_path)


def plot_roc(results: Results, roc_classes: List[int]) -> None:
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in results.unique_labels:
        (fpr[i], tpr[i], _) = roc_curve(results.labels == i, results.output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in results.unique_labels]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in results.unique_labels:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(results.unique_labels)
    macro_auc = auc(all_fpr, mean_tpr)

    for i in roc_classes:
        plt.plot(fpr[i], tpr[i], label=f"ROC curve for {results.label_names[i]} ({roc_auc[i]:.3f})")

    plt.plot(
        all_fpr,
        mean_tpr,
        linestyle=":",
        color="darkorange",
        label=f"Macro-average ROC curve ({macro_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


def set_parser(subparsers):
    subparser = subparsers.add_parser(
        "results", help="read results file", formatter_class=util.ArgumentHelpFormatter
    )
    subparser.add_argument(
        "--print",
        default=False,
        action="store_true",
        help="print results table",
    )
    subparser.add_argument("--cnf", default=False, action="store_true", help="plot confusion matrix")
    subparser.add_argument(
        "--cnf-save", default=False, action="store_true", help="save confusion matrix as csv"
    )
    subparser.add_argument(
        "--cnf-classes", default=[], type=int, nargs="+", help="classes to plot confusion matrix for"
    )
    subparser.add_argument("--roc", default=False, action="store_true", help="plot roc curve")
    subparser.add_argument("--roc-classes", default=[], type=int, nargs="+", help="classes to plot roc for")
    subparser.add_argument("results_file", type=str, help="result file to process")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    results = Results.load(args.results_file)

    if args.print is True:
        results.pretty_print()

    if args.cnf is True:
        plot_confusion_matrix(results, args.cnf_classes)

    if args.cnf_save is True:
        save_confusion_matrix(results, args.cnf_classes, args.results_file)

    if args.roc is True:
        plot_roc(results, args.roc_classes)
