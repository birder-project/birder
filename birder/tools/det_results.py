import argparse
import fnmatch
import logging
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.datatypes.classes
from rich.console import Console
from rich.table import Table

from birder.common import cli
from birder.conf import settings
from birder.results.detection import Results
from birder.results.gui import ConfusionMatrix
from birder.tools.results import print_most_confused_pairs

logger = logging.getLogger(__name__)


def compare_results(results_dict: dict[str, Results]) -> pl.DataFrame:
    result_list = []
    for name, results in results_dict.items():
        report_df = results.detailed_report()
        total_objects = int(report_df["Objects"].sum()) if report_df.is_empty() is False else 0
        result_entry = {
            "File name": name,
            "mAP": results.map,
            "mAP_50": results.metrics_dict["map_50"],
            "mAP_75": results.metrics_dict["map_75"],
            "mAP_small": results.metrics_dict["map_small"],
            "mAP_medium": results.metrics_dict["map_medium"],
            "mAP_large": results.metrics_dict["map_large"],
            "mAR_1": results.metrics_dict["mar_1"],
            "mAR_10": results.metrics_dict["mar_10"],
            "mAR_100": results.metrics_dict["mar_100"],
            "mAR_small": results.metrics_dict["mar_small"],
            "mAR_medium": results.metrics_dict["mar_medium"],
            "mAR_large": results.metrics_dict["mar_large"],
            "Samples": len(results),
            "Objects": total_objects,
        }
        result_list.append(result_entry)

    return pl.DataFrame(result_list)


def print_per_class_report(results_dict: dict[str, Results], classes: list[str]) -> None:
    console = Console()

    all_classes = []
    for results in results_dict.values():
        for cls in classes:
            all_classes.extend(fnmatch.filter(results.label_names, cls))

    classes = sorted(list(set(all_classes)))

    table = Table(show_header=True, header_style="bold dark_magenta")
    table.add_column("File name")
    table.add_column("Class name", style="dim")
    table.add_column("mAP", justify="right")
    table.add_column("mAR 100", justify="right")
    table.add_column("Objects", justify="right")

    per_class_mar = {
        name: dict(zip(results.metrics_dict["classes"], results.metrics_dict["mar_100_per_class"]))
        for name, results in results_dict.items()
    }

    for cls in classes:
        for name, results in results_dict.items():
            report_df = results.detailed_report()
            row = report_df.filter(pl.col("Class name") == cls)
            if row.is_empty() is True:
                continue

            class_num = int(row["Class"][0])
            mar_value = per_class_mar[name].get(class_num)
            mar_msg = "n/a" if mar_value is None else f"{mar_value:.4f}"

            map_msg = f"{row['mAP'][0]:.4f}"
            if row["mAP"][0] < 0.4:
                map_msg = "[red1]" + map_msg + "[/red1]"
            elif row["mAP"][0] < 0.5:
                map_msg = "[dark_orange]" + map_msg + "[/dark_orange]"

            table.add_row(
                name,
                row["Class name"][0],
                map_msg,
                mar_msg,
                f"{row['Objects'][0]}",
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


def confusion_matrix_data(
    results: Results,
    score_threshold: float,
    iou_threshold: float,
    classes: list[str],
    errors_only: bool,
) -> tuple[npt.NDArray[np.int_], list[str]]:
    cnf_matrix = results.confusion_matrix(score_threshold=score_threshold, iou_threshold=iou_threshold)
    label_names = results.label_names

    selected_indices: Optional[list[int]] = None
    if len(classes) > 0:
        selected_indices = []
        for pattern in classes:
            selected_indices.extend([idx for idx, name in enumerate(label_names) if fnmatch.fnmatch(name, pattern)])

        selected_indices = np.unique(selected_indices).tolist()

    elif errors_only is True:
        row_sums = cnf_matrix.sum(axis=1)
        col_sums = cnf_matrix.sum(axis=0)
        diag = np.diag(cnf_matrix)
        selected_indices = np.where((row_sums != diag) | (col_sums != diag))[0].tolist()
        if len(selected_indices) == 0:
            logger.warning("No mistakes found, showing full confusion matrix")
            return (cnf_matrix, label_names)

    if selected_indices is None:
        return (cnf_matrix, label_names)

    if 0 not in selected_indices:
        selected_indices.insert(0, 0)

    selected_indices = sorted(list(set(selected_indices)))

    cnf_matrix = cnf_matrix[np.ix_(selected_indices, selected_indices)]
    label_names = [label_names[idx] for idx in selected_indices]

    return (cnf_matrix, label_names)


def set_parser(subparsers: Any) -> None:
    subparser = subparsers.add_parser(
        "det-results",
        allow_abbrev=False,
        help="read and process detection result files",
        description="read and process detection result files",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools det-results "
            "results/faster_rcnn_coco_csp_resnet_50_imagenet1k_91_e0_640px_5000.json --print\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparser.add_argument("--print", default=False, action="store_true", help="print results table")
    subparser.add_argument("--short-print", default=False, action="store_true", help="print results")
    subparser.add_argument("--save-summary", default=False, action="store_true", help="save results summary as csv")
    subparser.add_argument("--summary-suffix", type=str, help="add suffix to summary file")
    subparser.add_argument("--classes", default=[], type=str, nargs="+", help="class names to compare")
    subparser.add_argument("--cnf", default=False, action="store_true", help="plot confusion matrix")
    subparser.add_argument(
        "--cnf-errors-only",
        default=False,
        action="store_true",
        help="show only classes with mistakes at the confusion matrix",
    )
    subparser.add_argument("--cnf-save", default=False, action="store_true", help="save confusion matrix as csv")
    subparser.add_argument(
        "--cnf-score-threshold", type=float, default=0.5, help="detection score threshold for confusion matrix"
    )
    subparser.add_argument(
        "--cnf-iou-threshold", type=float, default=0.5, help="iou threshold for confusion matrix matching"
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

    if args.cnf is True:
        if len(results_dict) > 1:
            logger.warning("Cannot compare confusion matrix, processing only the first file")

        results = next(iter(results_dict.values()))
        (cnf_matrix, label_names) = confusion_matrix_data(
            results, args.cnf_score_threshold, args.cnf_iou_threshold, args.classes, args.cnf_errors_only
        )
        title = f"Confusion matrix (score >= {args.cnf_score_threshold:.2f}, IoU >= {args.cnf_iou_threshold:.2f})"
        ConfusionMatrix(cnf_matrix, label_names, title=title).show()

    if args.cnf_save is True:
        for results_file, results in results_dict.items():
            filename = f"{results_file.rsplit('.', 1)[0]}_confusion_matrix.csv"
            cnf_matrix = results.confusion_matrix(
                score_threshold=args.cnf_score_threshold, iou_threshold=args.cnf_iou_threshold
            )
            ConfusionMatrix(cnf_matrix, results.label_names).save(filename)

    if args.most_confused is True:
        if len(results_dict) > 1:
            logger.warning("Cannot compare, processing only the first file")

        results = next(iter(results_dict.values()))
        most_confused_df = results.most_confused_pairs(
            n=10, score_threshold=args.cnf_score_threshold, iou_threshold=args.cnf_iou_threshold
        )
        print_most_confused_pairs(most_confused_df)
