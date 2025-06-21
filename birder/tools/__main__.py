import argparse

from birder.common import cli
from birder.tools import adversarial
from birder.tools import avg_model
from birder.tools import convert_model
from birder.tools import det_results
from birder.tools import download_model
from birder.tools import ensemble_model
from birder.tools import introspection
from birder.tools import labelme_to_coco
from birder.tools import list_models
from birder.tools import model_info
from birder.tools import pack
from birder.tools import quantize_model
from birder.tools import results
from birder.tools import show_det_iterator
from birder.tools import show_iterator
from birder.tools import similarity
from birder.tools import stats
from birder.tools import verify_coco
from birder.tools import verify_directory


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m birder.tools",
        allow_abbrev=False,
        description="Tool to run auxiliary commands",
        epilog=(
            "Usage examples:\n"
            "python -m birder.tools adversarial --method fgsm -n swin_transformer_v1 -p 2 -e 0 "
            "--image 'data/training/Mallard/000112.jpeg'\n"
            "python -m birder.tools avg-model --network resnet_v2 --net-param 50 --epochs 95 95 100\n"
            "python -m birder.tools convert-model --network convnext_v2_base --epoch 0 --pt2\n"
            "python -m birder.tools det-results "
            "results/faster_rcnn_coco_csp_resnet_50_imagenet1k_91_e0_640px_5000.json --print\n"
            "python -m birder.tools download-model mobilenet_v3_large_1_0\n"
            "python -m birder.tools ensemble-model --network convnext_v2_4_0 focalnet_3_0 --pts\n"
            "python -m birder.tools introspection --method gradcam --network efficientnet_v2 --net-param 1 "
            "--epoch 200 --image 'data/validation/Mallard/000003.jpeg'\n"
            "python -m birder.tools labelme-to-coco data/detection_data\n"
            "python -m birder.tools list-models --pretrained\n"
            "python -m birder.tools model-info -n deit -p 2 -t intermediate -e 0\n"
            "python -m birder.tools pack data/training\n"
            "python -m birder.tools quantize-model -n convnext_v2 -p 4 -e 0 --qbackend x86\n"
            "python -m birder.tools results results/inception_resnet_v2_105_e100_3150.csv --print --pr-curve\n"
            "python -m birder.tools show-det-iterator --mode inference --size 640 --batch\n"
            "python -m birder.tools show-iterator --mode training --size 256 320 --aug-level 5\n"
            "python -m birder.tools similarity -n efficientnet_v2_l -e 0 --limit 15 data/*/*crane\n"
            "python -m birder.tools stats --class-graph\n"
            "python -m birder.tools verify-coco --coco-json-path "
            "~/Datasets/Objects365-2020/train/zhiyuan_objv2_train.json --data-path ~/Datasets/Objects365-2020/train\n"
            "python -m birder.tools verify-directory data/testing\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    adversarial.set_parser(subparsers)
    avg_model.set_parser(subparsers)
    convert_model.set_parser(subparsers)
    det_results.set_parser(subparsers)
    ensemble_model.set_parser(subparsers)
    download_model.set_parser(subparsers)
    introspection.set_parser(subparsers)
    labelme_to_coco.set_parser(subparsers)
    list_models.set_parser(subparsers)
    model_info.set_parser(subparsers)
    pack.set_parser(subparsers)
    quantize_model.set_parser(subparsers)
    results.set_parser(subparsers)
    show_det_iterator.set_parser(subparsers)
    show_iterator.set_parser(subparsers)
    similarity.set_parser(subparsers)
    stats.set_parser(subparsers)
    verify_coco.set_parser(subparsers)
    verify_directory.set_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
