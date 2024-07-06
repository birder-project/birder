import argparse

from birder.common import cli
from birder.tools import adversarial
from birder.tools import avg_model
from birder.tools import ensemble_model
from birder.tools import fetch_model
from birder.tools import introspection
from birder.tools import labelme_to_coco
from birder.tools import list_models
from birder.tools import model_info
from birder.tools import pack
from birder.tools import quantize_model
from birder.tools import results
from birder.tools import script_model
from birder.tools import show_det_iterator
from birder.tools import show_iterator
from birder.tools import similarity
from birder.tools import stats
from birder.tools import verify_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m birder.tools",
        allow_abbrev=False,
        description="Tool to run auxiliary commands",
        epilog=(
            "Usage examples:\n"
            "python tool.py adversarial --method fgsm -n swin_transformer_v1 -p 2 -e 0 "
            "--image 'data/training/Mallard/000112.jpeg'\n"
            "python tool.py avg-model --network resnet_v2 --net-param 50 --epochs 95 95 100\n"
            "python tool.py ensemble-model --network convnext_v2_4_0 focalnet_3_0\n"
            "python tool.py fetch-model mobilenet_v3_1_0\n"
            "python tool.py introspection --method gradcam --network efficientnet_v2 --net-param 1 "
            "--epoch 200 --image 'data/validation/Mallard/000003.jpeg'\n"
            "python tool.py labelme-to-coco data/detection_data\n"
            "python tool.py list-models --registry\n"
            "python tool.py model-info -n deit -p 2 -t intermediate -e 0\n"
            "python tool.py pack data/training\n"
            "python tool.py quantize-model -n convnext_v2 -p 4 -e 0 --qbackend x86\n"
            "python tool.py results results/inception_resnet_v2_105_e100_3150.csv --print --pr-curve\n"
            "python tool.py script-model --network shufflenet_v2 --net-param 2 --epoch 100\n"
            "python tool.py show-det-iterator --mode inference --size 640 --batch\n"
            "python tool.py show-iterator --mode training --size 256 --aug-level 3\n"
            "python tool.py similarity -n efficientnet_v2 -p 1 -e 0 --limit 15 data/*/*crane\n"
            "python tool.py stats --class-graph\n"
            "python tool.py verify-directory data/testing\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    adversarial.set_parser(subparsers)
    avg_model.set_parser(subparsers)
    ensemble_model.set_parser(subparsers)
    fetch_model.set_parser(subparsers)
    introspection.set_parser(subparsers)
    labelme_to_coco.set_parser(subparsers)
    list_models.set_parser(subparsers)
    model_info.set_parser(subparsers)
    pack.set_parser(subparsers)
    quantize_model.set_parser(subparsers)
    results.set_parser(subparsers)
    script_model.set_parser(subparsers)
    show_det_iterator.set_parser(subparsers)
    show_iterator.set_parser(subparsers)
    similarity.set_parser(subparsers)
    stats.set_parser(subparsers)
    verify_directory.set_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
