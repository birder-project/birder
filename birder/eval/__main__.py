import argparse

from birder.common import cli
from birder.eval import adversarial
from birder.eval import classification
from birder.eval.benchmarks import awa2
from birder.eval.benchmarks import bioscan5m
from birder.eval.benchmarks import fishnet
from birder.eval.benchmarks import flowers102
from birder.eval.benchmarks import fungiclef
from birder.eval.benchmarks import nabirds
from birder.eval.benchmarks import newt
from birder.eval.benchmarks import plankton
from birder.eval.benchmarks import plantdoc
from birder.eval.benchmarks import plantnet


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m birder.eval",
        allow_abbrev=False,
        description="Evaluation module",
        epilog=(
            "Usage examples:\n"
            "python -m birder.eval adversarial -n resnet_v2_50 -t il-all -e 100 --method pgd "
            "--gpu data/validation_il-all_packed\n"
            "python -m birder.eval classification --filter '*il-all*' --gpu data/validation_il-all\n"
            "---\n"
            "python -m birder.eval awa2 --embeddings "
            "results/awa2/*.parquet --dataset-path ~/Datasets/Animals_with_Attributes2 --gpu\n"
            "python -m birder.eval bioscan5m --embeddings "
            "results/bioscan5m/*.parquet --data-path ~/Datasets/BIOSCAN-5M/species/testing_unseen\n"
            "python -m birder.eval fishnet --embeddings "
            "results/vit_b16_224px_embeddings.parquet --dataset-path ~/Datasets/fishnet --gpu\n"
            "python -m birder.eval flowers102 --embeddings "
            "results/flowers102_rope_i_vit_s16_pn_aps_c1_pe-core_0_384px_crop1.0_8189_sc_embeddings.parquet "
            "--dataset-path ~/Datasets/Flowers102\n"
            "python -m birder.eval fungiclef --embeddings "
            "results/fungiclef/*.parquet --dataset-path ~/Datasets/FungiCLEF2023 --gpu\n"
            "python -m birder.eval nabirds --embeddings "
            "results/vit_b16_224px_crop1.0_48562_embeddings.parquet --dataset-path ~/Datasets/nabirds --gpu\n"
            "python -m birder.eval newt --embeddings "
            "results/vit_reg4_so150m_p14_ls_dino-v2-bio_0_e45_224px_crop1.0_36032_output.parquet "
            "--dataset-path ~/Datasets/NeWT\n"
            "python -m birder.eval plankton --embeddings "
            "results/plankton/*.parquet --dataset-path ~/Datasets/plankton --gpu\n"
            "python -m birder.eval plantdoc --embeddings "
            "results/plantdoc_embeddings.parquet --dataset-path ~/Datasets/PlantDoc\n"
            "python -m birder.eval plantnet --embeddings "
            "results/plantnet_embeddings.parquet --dataset-path ~/Datasets/plantnet_300K\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    adversarial.set_parser(subparsers)
    classification.set_parser(subparsers)

    awa2.set_parser(subparsers)
    bioscan5m.set_parser(subparsers)
    fishnet.set_parser(subparsers)
    flowers102.set_parser(subparsers)
    fungiclef.set_parser(subparsers)
    nabirds.set_parser(subparsers)
    newt.set_parser(subparsers)
    plankton.set_parser(subparsers)
    plantdoc.set_parser(subparsers)
    plantnet.set_parser(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
