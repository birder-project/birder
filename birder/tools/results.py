import argparse
import fnmatch
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import polars.datatypes.classes
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.conf import settings
from birder.results.classification import Results
from birder.results.classification import compare_results
from birder.results.gui import ROC
from birder.results.gui import ConfusionMatrix
from birder.results.gui import PrecisionRecall
from birder.results.gui import ProbabilityHistogram

logger = logging.getLogger(__name__)


def print_per_class_report(results_dict: dict[str, Results], classes: list[str]) -> None:
    console = Console()

    # Expand classes according to shell-style wildcards
    all_classes = []
    for results in results_dict.values():
        for cls in classes:
            all_classes.extend(fnmatch.filter(results.label_names, cls))

    classes = sorted(list(set(all_classes)))

    # Per class
    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("File name")
    table.add_column("Class name", style="dim")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-score", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("False negative", justify="right")
    table.add_column("False positive", justify="right")

    for cls in classes:
        for name, results in results_dict.items():
            report_df = results.detailed_report()
            row = report_df.filter(pl.col("Class name") == cls)
            if row.is_empty() is True:
                continue

            recall_msg = f"{row['Recall'][0]:.4f}"
            if row["Recall"][0] < 0.75:
                recall_msg = "[red1]" + recall_msg + "[/red1]"
            elif row["Recall"][0] < 0.9:
                recall_msg = "[dark_orange]" + recall_msg + "[/dark_orange]"

            f1_msg = f"{row['F1-score'][0]:.4f}"
            if row["F1-score"][0] == 1.0:
                f1_msg = "[green]" + f1_msg + "[/green]"

            table.add_row(
                name,
                row["Class name"][0],
                f"{row['Precision'][0]:.4f}",
                recall_msg,
                f1_msg,
                f"{row['Samples'][0]}",
                f"{row['False negative'][0]}",
                f"{row['False positive'][0]}",
            )

    console.print(table)


def print_report(results_dict: dict[str, Results]) -> None:
    if len(results_dict) == 1:
        results = next(iter(results_dict.values()))
        results.pretty_print()
        return

    results_df = compare_results(results_dict)
    console = Console()
    table = Table(show_header=True, header_style="bold dark_magenta")
    for idx, column in enumerate(results_df.columns):
        if idx == 0:
            table.add_column(column)
        else:
            table.add_column(column, justify="right")

        if isinstance(results_df[column].dtype, polars.datatypes.classes.FloatType):
            results_df = results_df.with_columns(
                pl.col(column).map_elements(lambda x: f"{x:.4f}", return_dtype=pl.String)
            )
        else:
            results_df = results_df.with_columns(pl.col(column).cast(pl.String))

    for row in results_df.iter_rows():
        table.add_row(*row)

    console.print(table)
    console.print("\n")


def print_most_confused_pairs(results: Results) -> None:
    most_confused_df = results.most_confused(n=14)

    console = Console()

    table = Table(show_header=True, header_style="bold dark_magenta")
    for column in most_confused_df.columns:
        if isinstance(most_confused_df[column].dtype, polars.datatypes.classes.NumericType):
            table.add_column(column.capitalize(), justify="right")
        else:
            table.add_column(column.capitalize())

    for row in most_confused_df.iter_rows():
        table.add_row(row[0], row[1], str(row[2]), str(row[3]))

    console.print(table)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "results",
        allow_abbrev=False,
        help="read and process result files",
        description="read and process result files",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools results results/vit_l16_mim340_218_e0_448px_crop1.0_10883.csv "
            "--cnf --cnf-mistakes\n"
            'python -m birder.tools results results/deit_2_* --print --classes "Lesser kestrel" '
            '"Common kestrel" "*swan"\n'
            "python -m birder.tools results results/inception_resnet_v2_105_e100_299px_crop1.0_3150.csv "
            "--print --roc\n"
            "python -m birder.tools results results/inception_resnet_v2_105_e100_299px_crop1.0_3150.csv "
            '--pr-curve --classes "Common crane" "Demoiselle crane"\n'
            "python -m birder.tools results results/densenet_121_105_e100_224px_crop1.0_3150.csv --prob-hist "
            '"Common kestrel" "Red-footed falcon"\n'
            "python -m birder.tools results results/inception_resnet_v2_105_e100_299px_crop1.0_3150.csv --cnf "
            "--classes Mallard Unknown Wallcreeper\n"
            "python -m birder.tools results results/maxvit_2_154_e0_288px_crop1.0_6286.csv "
            "results/inception_next_1_160_e0_384px_crop1.0_6762.csv --print\n"
            "python -m birder.tools results results/convnext_v2_base_214_e0_448px_crop1.0_10682.csv "
            '--prob-hist "Common kestrel" "Lesser kestrel"\n'
            "python -m birder.tools results results/squeezenet_il-common_367_e0_259px_crop1.0_13029.csv "
            "--most-confused\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--print", default=False, action="store_true", help="print results table")
    subparser.add_argument("--short-print", default=False, action="store_true", help="print results")
    subparser.add_argument("--save-summary", default=False, action="store_true", help="save results summary as csv")
    subparser.add_argument("--summary-suffix", type=str, help="add suffix to summary file")
    subparser.add_argument(
        "--print-mistakes", default=False, action="store_true", help="print only classes with non-perfect f1-score"
    )
    subparser.add_argument("--classes", default=[], type=str, nargs="+", help="class names to compare")
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
    subparser.add_argument("--roc", default=False, action="store_true", help="plot roc curve")
    subparser.add_argument("--pr-curve", default=False, action="store_true", help="plot precision recall curve")
    subparser.add_argument(
        "--prob-hist", type=str, nargs=2, help="classes to plot probability histogram against each other"
    )
    subparser.add_argument("--most-confused", default=False, action="store_true", help="print most confused pairs")
    subparser.add_argument("result_files", type=str, nargs="+", help="result files to process")
    subparser.set_defaults(func=main)


# pylint: disable=too-many-branches
def main(args: argparse.Namespace) -> None:
    results_dict: dict[str, Results] = {}
    for results_file in args.result_files:
        results = Results.load(results_file)
        result_name = results_file.split("/")[-1]
        results_dict[result_name] = results

    if args.print is True:
        if args.print_mistakes is True and len(results_dict) > 1:
            logger.warning("Cannot print mistakes in compare mode. processing only the first file")

        if args.print_mistakes is True:
            (result_name, results) = next(iter(results_dict.items()))
            label_names_arr = np.array(results.label_names)
            classes_list: list[str] = label_names_arr[results.mistakes["prediction"].unique()].tolist()
            classes_list.extend(list(results.mistakes["label_name"].unique()))
            results_df = results.get_as_df().filter(pl.col("label_name").is_in(classes_list))

            results = Results(
                results_df["sample"].to_list(),
                results_df["label"].to_list(),
                results.label_names,
                results_df[:, Results.num_desc_cols :].to_numpy(),
            )
            results_dict = {result_name: results}

        print_report(results_dict)
        if len(args.classes) > 0:
            print_per_class_report(results_dict, args.classes)

    if args.short_print is True:
        for name, results in results_dict.items():
            print(f"{name}: {results}\n")

    if args.save_summary is True:
        if args.summary_suffix is not None:
            summary_path = settings.RESULTS_DIR.joinpath(f"summary_{args.summary_suffix}.csv")
        else:
            summary_path = settings.RESULTS_DIR.joinpath("summary.csv")

        if summary_path.exists() is True:
            logger.warning(f"Summary already exists '{summary_path}', skipping...")
        else:
            logger.info(f"Writing results summary at '{summary_path}...")
            results_df = compare_results(results_dict)
            results_df.write_csv(summary_path)

    if args.list_mistakes is True:
        for name, results in results_dict.items():
            mistakes = sorted(list(results.mistakes["sample"]))
            print("\n".join(mistakes))
            logger.info(f"{len(results.mistakes):,} mistakes found at {name}")

    if args.list_out_of_k is True:
        for name, results in results_dict.items():
            out_of_k = sorted(list(results.out_of_top_k["sample"]))
            print("\n".join(out_of_k))
            logger.info(f"{len(results.out_of_top_k):,} out of k found at {name}")

    if args.cnf is True:
        if len(results_dict) > 1:
            logger.warning("Cannot compare confusion matrix, processing only the first file")

        results = next(iter(results_dict.values()))
        if len(args.classes) > 0:
            results_df = results.get_as_df().filter(pl.col("label_name").is_in(args.classes))
            cnf_results = Results(
                results_df["sample"].to_list(),
                results_df["label"].to_list(),
                results.label_names,
                results_df[:, Results.num_desc_cols :].to_numpy(),
            )

        elif args.cnf_mistakes is True:
            label_names_arr = np.array(results.label_names)
            classes_list: list[str] = label_names_arr[results.mistakes["prediction"].unique()].tolist()  # type: ignore
            classes_list.extend(list(results.mistakes["label_name"].unique()))
            results_df = results.get_as_df().filter(pl.col("label_name").is_in(classes_list))
            cnf_results = Results(
                results_df["sample"].to_list(),
                results_df["label"].to_list(),
                results.label_names,
                results_df[:, Results.num_desc_cols :].to_numpy(),
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

        roc.show(args.classes)

    if args.pr_curve is True:
        pr_curve = PrecisionRecall()
        for name, results in results_dict.items():
            pr_curve.add_result(Path(name).name, results)

        pr_curve.show(args.classes)

    if args.prob_hist is not None:
        if len(results_dict) > 1:
            logger.warning("Cannot compare probability histograms, processing only the first file")

        results = next(iter(results_dict.values()))
        ProbabilityHistogram(results).show(*args.prob_hist)

    if args.most_confused is True:
        if len(results_dict) > 1:
            logger.warning("Cannot compare, processing only the first file")

        results = next(iter(results_dict.values()))
        print_most_confused_pairs(results)
