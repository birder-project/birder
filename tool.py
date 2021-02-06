import argparse

from birder.common import util
from birder.tools import aug
from birder.tools import avg_model
from birder.tools import collect_model
from birder.tools import ensemble_model
from birder.tools import export_onnx
from birder.tools import fgsm
from birder.tools import results
from birder.tools import show_iterator
from birder.tools import stats


def main() -> None:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Tool to run auxiliary commands",
        epilog=(
            "Usage examples:\n"
            "python3 tool.py aug --ratio 0.1 -j 4\n"
            "python3 tool.py avg-model --network shufflenet_v2_2.0 --size 224 --epochs 80 100 120\n"
            "python3 tool.py collect-model --network mobilenet_v1_1.0 --epoch 120\n"
            "python3 tool.py ensemble-model --network mobilenet_v2_1.25 mobilenet_v2_1.25\n"
            "python3 tool.py export-onnx --network inception_v4 --epoch 80 --size 299\n"
            "python3 tool.py fgsm --network inception_resnet_v2 --ratio 0.05\n"
            "python3 tool.py results --show-cnf results/xception_94_e0_299px_2820.csv\n"
            "python3 tool.py show-iterator --iterator ImageRecordIterMixup\n"
            "python3 tool.py stats --class-graph-only\n"
        ),
        formatter_class=util.ArgumentHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    aug.set_parser(subparsers)
    avg_model.set_parser(subparsers)
    collect_model.set_parser(subparsers)
    ensemble_model.set_parser(subparsers)
    export_onnx.set_parser(subparsers)
    fgsm.set_parser(subparsers)
    results.set_parser(subparsers)
    show_iterator.set_parser(subparsers)
    stats.set_parser(subparsers)
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
