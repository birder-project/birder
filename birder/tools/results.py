import argparse
import fnmatch
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.conf import settings
from birder.core.results.classification import Results
from birder.core.results.gui import ROC
from birder.core.results.gui import ConfusionMatrix
from birder.core.results.gui import PrecisionRecall
from birder.core.results.gui import ProbabilityHistogram


def print_report(results_dict: dict[str, Results], classes: list[str]) -> None:
    if len(results_dict) == 1:
        results = next(iter(results_dict.values()))
        results.pretty_print()
        return

    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File name")
    table.add_column("Accuracy", justify="right")
    table.add_column(f"Top-{settings.TOP_K} Accuracy", justify="right")
    table.add_column("Macro F1-Score", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Mistakes", justify="right")

    for name, results in results_dict.items():
        table.add_row(
            name,
            f"{results.accuracy:.3f}",
            f"{results.top_k:.3f}",
            f"{results.macro_f1_score:.3f}",
            f"{len(results)}",
            f"{results._num_mistakes}",  # pylint: disable=protected-access
        )

    console.print(table)
    console.print("\n")
    if len(classes) == 0:
        return

    # Expand classes according to shell-style wildcards
    all_classes = []
    for results in results_dict.values():
        for cls in classes:
            all_classes.extend(fnmatch.filter(results.label_names, cls))

    classes = sorted(list(set(all_classes)))

    # Per class
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File name")
    table.add_column("Class name", style="dim")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("False negative", justify="right")
    table.add_column("False positive", justify="right")

    for cls in classes:
        for name, results in results_dict.items():
            report_df = results.detailed_report()
            row = report_df[report_df["Class name"] == cls].squeeze()
            if row.empty is True:
                continue

            recall_msg = f"{row['Recall']:.3f}"
            if row["Recall"] < 0.75:
                recall_msg = "[red1]" + recall_msg + "[/red1]"

            elif row["Recall"] < 0.9:
                recall_msg = "[dark_orange]" + recall_msg + "[/dark_orange]"

            f1_msg = f"{row['F1-Score']:.3f}"
            if row["F1-Score"] == 1.0:
                f1_msg = "[green]" + f1_msg + "[/green]"

            table.add_row(
                name,
                row["Class name"],
                f"{row['Precision']:.3f}",
                recall_msg,
                f1_msg,
                f"{row['Samples']}",
                f"{row['False negative']}",
                f"{row['False positive']}",
            )

    console.print(table)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "results",
        help="read and process result files",
        description="read and process result files",
        epilog=(
            "Usage examples:\n"
            "python3 tool.py results results/vit_3_pretrained340_218_e0_448px_crop1.0_10883.csv "
            "--cnf --cnf-mistakes\n"
            'python3 tool.py results results/deit_2_* --print --classes "Lesser kestrel" "Common kestrel" "*swan"\n'
            "python3 tool.py results results/inception_resnet_v2_105_e100_299px_crop1.0_3150.csv "
            "--print --roc\n"
            "python3 tool.py results results/inception_resnet_v2_105_e100_299px_crop1.0_3150.csv "
            '--pr-curve --pr-classes "Common crane" "Demoiselle crane"\n'
            "python3 tool.py results results/densenet_121_105_e100_224px_crop1.0_3150.csv --prob-hist "
            '"Common kestrel" "Red-footed falcon"\n'
            "python3 tool.py results results/inception_resnet_v2_105_e100_299px_crop1.0_3150.csv --cnf "
            "--cnf-classes Mallard Unknown Wallcreeper\n"
            "python3 tool.py results results/maxvit_2_154_e0_288px_crop1.0_6286.csv "
            "results/inception_next_1_160_e0_384px_crop1.0_6762.csv --print\n"
            "python3 tool.py results results/convnext_v2_4_214_e0_448px_crop1.0_10682.csv "
            '--prob-hist "Common kestrel" "Lesser kestrel"\n'
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument(
        "--print",
        default=False,
        action="store_true",
        help="print results table",
    )
    subparser.add_argument("--classes", default=[], type=str, nargs="+", help="class name to compare (print)")
    subparser.add_argument("--list-mistakes", default=False, action="store_true", help="list all mistakes")
    subparser.add_argument("--list-out-of-k", default=False, action="store_true", help="list all samples not in top-k")
    subparser.add_argument("--cnf", default=False, action="store_true", help="plot confusion matrix")
    subparser.add_argument(
        "--cnf-mistakes",
        default=False,
        action="store_true",
        help="show only classes with mistakes at the confusion matrix",
    )
    subparser.add_argument("--cnf-save", default=False, action="store_true", help="save confusion matrix as csv")
    subparser.add_argument(
        "--cnf-classes", type=str, default=[], nargs="+", help="classes to plot confusion matrix for"
    )
    subparser.add_argument("--roc", default=False, action="store_true", help="plot roc curve")
    subparser.add_argument("--roc-classes", type=str, default=[], nargs="+", help="classes to plot roc for")
    subparser.add_argument("--pr-curve", default=False, action="store_true", help="plot precision recall curve")
    subparser.add_argument("--pr-classes", type=str, default=[], nargs="+", help="classes to plot pr for")
    subparser.add_argument(
        "--prob-hist", type=str, nargs=2, help="classes to plot probability histogram against each other"
    )
    subparser.add_argument("result_files", type=str, nargs="+", help="result files to process")
    subparser.set_defaults(func=main)


# pylint: disable=too-many-branches
def main(args: argparse.Namespace) -> None:
    results_dict: dict[str, Results] = {}
    for results_file in args.result_files:
        results = Results.load(results_file)
        results_dict[results_file] = results

    if args.print is True:
        print_report(results_dict, args.classes)

    if args.list_mistakes is True:
        for name, results in results_dict.items():
            print()
            mistakes = sorted(list(results.mistakes["sample"]))
            print("\n".join(mistakes))
            logging.info(f"{len(results.mistakes):,} mistakes found at {name}")

    if args.list_out_of_k is True:
        for name, results in results_dict.items():
            print()
            out_of_k = sorted(list(results.out_of_top_k["sample"]))
            print("\n".join(out_of_k))
            logging.info(f"{len(results.out_of_top_k):,} out of k found at {name}")

    if args.cnf is True:
        if len(results_dict) > 1:
            logging.warning("Cannot compare confusion matrix, processing only the first file")

        results = next(iter(results_dict.values()))
        if len(args.cnf_classes) > 0:
            results_df = results.get_as_df()[results.get_as_df()["label_name"].isin(args.cnf_classes)]
            cnf_results = Results(
                results_df["sample"],
                results_df["label"],
                results.label_names,
                results_df.iloc[:, Results.num_desc_cols :].values,
            )

        elif args.cnf_mistakes is True:
            classes_list = list(results.prediction_names.iloc[results.mistakes.index].unique())
            classes_list.extend(list(results.mistakes["label_name"].unique()))
            results_df = results.get_as_df()[results.get_as_df()["label_name"].isin(classes_list)]
            cnf_results = Results(
                results_df["sample"],
                results_df["label"],
                results.label_names,
                results_df.iloc[:, Results.num_desc_cols :].values,
            )

        else:
            cnf_results = results

        ConfusionMatrix(cnf_results).show()

    if args.cnf_save is True:
        for results_file, results in results_dict.items():
            filename = f"{results_file[:-4]}_confusion_matrix.csv"
            ConfusionMatrix(results).save(filename)

    if args.roc is True:
        roc = ROC()
        for name, results in results_dict.items():
            roc.add_result(Path(name).name, results)

        roc.show(args.roc_classes)

    if args.pr_curve is True:
        pr_curve = PrecisionRecall()
        for name, results in results_dict.items():
            pr_curve.add_result(Path(name).name, results)

        pr_curve.show(args.pr_classes)

    if args.prob_hist is not None:
        if len(results_dict) > 1:
            logging.warning("Cannot compare probability histograms, processing only the first file")

        results = next(iter(results_dict.values()))
        ProbabilityHistogram(results).show(*args.prob_hist)
