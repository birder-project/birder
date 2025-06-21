import argparse
import logging
from typing import Any

from birder.common import cli
from birder.conf import settings
from birder.results.detection import Results

logger = logging.getLogger(__name__)


def print_report(results_dict: dict[str, Results]) -> None:
    if len(results_dict) == 1:
        results = next(iter(results_dict.values()))
        results.pretty_print()
        return

    raise NotADirectoryError


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
    subparser.add_argument("--save-summary", default=False, action="store_true", help="save results summary as csv")
    subparser.add_argument("--summary-suffix", type=str, help="add suffix to summary file")
    subparser.add_argument("result_files", type=str, nargs="+", help="result files to process")
    subparser.set_defaults(func=main)


def main(args: argparse.Namespace) -> None:
    results_dict: dict[str, Results] = {}
    for results_file in args.result_files:
        results = Results.load(results_file)
        result_name = results_file.split("/")[-1]
        results_dict[result_name] = results

    if args.print is True:
        print_report(results_dict)

    if args.save_summary is True:
        if args.summary_suffix is not None:
            summary_path = settings.RESULTS_DIR.joinpath(f"summary_{args.summary_suffix}.csv")
        else:
            summary_path = settings.RESULTS_DIR.joinpath("summary.csv")

        if summary_path.exists() is True:
            logger.warning(f"Summary already exists '{summary_path}', skipping...")
        else:
            logger.info(f"Writing results summary at '{summary_path}...")
            raise NotImplementedError
