# type: ignore
import pathlib
import time

import polars as pl
import torch
from invoke import Exit
from invoke import task
from jinja2 import Template

from birder.common import cli
from birder.common import fs_ops
from birder.common import lib
from birder.conf import settings
from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.tools.pack import CustomImageFolder
from birder.tools.stats import detection_object_count
from birder.tools.stats import directory_label_count

if pathlib.Path(__file__).parent != pathlib.Path().resolve():
    print("Can only run from the root directory, aborting...")
    raise SystemExit

COLOR_GRAY = 37
COLOR_GREEN = 32
COLOR_RED = 91
DEFAULT_COLOR = COLOR_GRAY

PROJECT_DIR = "birder"
HF_DOCS_DIR = "docs/internal/hf_model_cards"
HF_MODEL_CARD_TEMPLATE = "docs/internal/model_card_template.md.j2"


def echo(msg: str, color: int = DEFAULT_COLOR) -> None:
    print(f"\033[1;{color}m{msg}\033[0m")


def _class_list() -> list[str]:
    return sorted([x.name for x in settings.TRAINING_DATA_PATH.iterdir() if x.is_dir()])


#####################
# Linting and testing
#####################


@task
def ci(ctx, coverage=False):
    """
    Run all linters and tests, set return code 0 only if everything succeeded.
    """

    tic = time.time()

    return_code = 0

    if pylint(ctx) != 0:
        return_code = 1

    if sec(ctx) != 0:
        return_code = 1

    if pytest(ctx, coverage) != 0:
        return_code = 1

    echo("")
    toc = time.time()
    echo(f"CI took {(toc - tic):.1f}s")
    if return_code == 0:
        echo("CI Passed", color=COLOR_GREEN)

    else:
        echo("CI Failed", color=COLOR_RED)

    raise Exit(code=return_code)


@task
def pylint(ctx):
    """
    Run pylint & flake8 on all Python files, type check and formatting check
    """

    return_code = 0

    # pylint
    result = ctx.run(
        f"python -m pylint *.py tests tests_flow {PROJECT_DIR}",
        echo=True,
        pty=True,
        warn=True,
    )
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # flake8
    result = ctx.run("python -m flake8 .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # mypy type checking
    result = ctx.run(
        "python -m mypy --pretty --show-error-codes .",
        echo=True,
        pty=True,
        warn=True,
    )
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # Format check, black
    result = ctx.run("python -m black --check .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    # Import check, isort
    result = ctx.run("python -m isort --check-only .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def sec(ctx):
    """
    Run security related analysis
    """

    return_code = 0

    result = ctx.run("python -m bandit -r .", echo=True, pty=True, warn=True)

    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def pytest(ctx, coverage=False):
    """
    Run Python tests
    """

    return_code = 0

    if coverage is True:
        result = ctx.run(
            f"python -m coverage run --source={PROJECT_DIR} -m unittest discover -s tests -v",
            echo=True,
            pty=True,
            warn=True,
        )
        ctx.run("python -m coverage report", echo=True, pty=True, warn=True)
    else:
        result = ctx.run("python -m unittest discover -s tests -v", echo=True, pty=True, warn=True)

    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)
    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code


###############
# Data handling
###############


@task
def update_annotation_table(_ctx):
    """
    Add new classes to the annotations status table
    """

    (training_detection_count, _) = detection_object_count(settings.TRAINING_DETECTION_ANNOTATIONS_PATH)
    (validation_detection_count, _) = detection_object_count(settings.VALIDATION_DETECTION_ANNOTATIONS_PATH)

    class_list = _class_list()
    column_class = "class"
    classes = pl.Series(column_class, class_list)

    annotations_status = pl.read_csv(settings.DATA_DIR.joinpath("annotations_status.csv"))
    new_classes = classes.filter(~classes.is_in(annotations_status[column_class]))
    if len(new_classes) == 0:
        echo("No new species")
    else:
        new_annotations_status = new_classes.to_frame()
        for column in annotations_status.columns:
            if column == column_class:
                continue

            new_annotations_status = new_annotations_status.with_columns(
                pl.lit(0).cast(annotations_status[column].dtype).alias(column)
            )

        annotations_status = pl.concat([annotations_status, new_annotations_status])
        annotations_status = annotations_status.sort(column_class, descending=False)

    # Update sample count
    training_count = directory_label_count(settings.TRAINING_DATA_PATH)
    validation_count = directory_label_count(settings.VALIDATION_DATA_PATH)
    testing_count = directory_label_count(settings.TESTING_DATA_PATH)
    annotations_status = annotations_status.with_columns(
        pl.col(column_class).replace_strict(training_count, default=0).alias("training_samples")
    )
    annotations_status = annotations_status.with_columns(
        pl.col(column_class).replace_strict(validation_count, default=0).alias("validation_samples")
    )
    annotations_status = annotations_status.with_columns(
        pl.col(column_class).replace_strict(testing_count, default=0).alias("testing_samples")
    )
    annotations_status = annotations_status.with_columns(
        pl.col(column_class).replace_strict(training_detection_count, default=0).alias("training_detection_samples")
    )
    annotations_status = annotations_status.with_columns(
        pl.col(column_class).replace_strict(validation_detection_count, default=0).alias("validation_detection_samples")
    )

    # Save
    annotations_status.write_csv(settings.DATA_DIR.joinpath("annotations_status.csv"))
    echo(f"Done, added {len(new_classes)} new classes")


@task
def gen_android_classes(_ctx):
    """
    Generates Android values.xml
    """

    class_list = _class_list()

    doc = ""
    doc += "<resources>\n"
    for key in class_list:
        doc += f'    <string name="{key.lower().replace(" ", "_").replace("-", "_")}">{key}</string>\n'

    doc += "</resources>\n"
    echo("Writing android/species.xml")
    with open("android/species.xml", "w", encoding="utf-8") as handle:
        handle.write(doc)

    echo("Done !")


@task
def gen_classes_file(_ctx):
    """
    Generates data/detection_data/classes.txt
    """

    if settings.DETECTION_DATA_PATH.exists() is False:
        settings.DETECTION_DATA_PATH.mkdir(parents=True)

    class_list = _class_list()
    class_to_idx = fs_ops.read_class_file(settings.DETECTION_DATA_PATH.joinpath(settings.CLASS_LIST_NAME))
    if class_list == list(class_to_idx.keys()):
        echo("No new species")
    else:
        # Remove "Unknown" class (not required in detection)
        class_list.remove("Unknown")
        doc = "\n".join(class_list)

        echo("data/detection_data/classes.txt")
        with open("data/detection_data/classes.txt", "w", encoding="utf-8") as handle:
            handle.write(doc)

    echo(f"Done, added {len(class_list) - len(class_to_idx)} new classes")


@task
def convert_to_coco(ctx, class_file=None):
    """
    Convert labelme annotations to coco format for both training and validation sets
    """

    if class_file is not None:
        stmt = f" --class-file {class_file} "
    else:
        stmt = " "

    ctx.run(
        f"python tool.py labelme-to-coco{stmt}data/detection_data/training_annotations",
        echo=True,
        pty=True,
        warn=True,
    )
    ctx.run(
        f"python tool.py labelme-to-coco{stmt}data/detection_data/validation_annotations",
        echo=True,
        pty=True,
        warn=True,
    )


@task
def pack_intermediate(ctx, jobs=12, size=384, suffix="intermediate"):
    """
    Pack data for intermediate training
    """

    ctx.run(
        f"python tool.py pack -j {jobs} --shuffle --suffix {suffix} --target-path data/{suffix} --size {size} "
        "data/training data/raw_data",
        echo=True,
        pty=True,
        warn=True,
    )
    ctx.run(
        f"python tool.py pack -j {jobs} --suffix {suffix} --target-path data/{suffix} --size {size} --split validation "
        "--max-size 200 --append data/validation",
        echo=True,
        pty=True,
        warn=True,
    )


@task
def pack_class_file(ctx, cls, size=384):
    """
    Pack a class file in directory format
    """

    ctx.run(
        f"python tool.py pack --type directory -j 8 --suffix {cls}_packed --size {size} --format jpeg "
        f"--class-file data/{cls}_classes.txt data/training",
        echo=True,
        pty=True,
        warn=True,
    )
    ctx.run(
        f"python tool.py pack --type directory -j 8 --suffix {cls}_packed --size {size} --format jpeg "
        f"--class-file data/{cls}_classes.txt data/validation",
        echo=True,
        pty=True,
        warn=True,
    )


@task
def print_datasets_stats(_ctx):
    """
    Print stats for all datasets based on class files at the main data directory
    """

    datasets_stats = []
    for class_file in settings.DATA_DIR.glob("*_classes.txt"):
        class_to_idx = fs_ops.read_class_file(class_file)
        dataset_name = class_file.stem.removesuffix("_classes")

        training_dataset = CustomImageFolder(settings.TRAINING_DATA_PATH, class_to_idx=class_to_idx)
        validation_dataset = CustomImageFolder(settings.VALIDATION_DATA_PATH, class_to_idx=class_to_idx)
        datasets_stats.append(
            {
                "Name": dataset_name,
                "Training samples": len(training_dataset),
                "Validation samples": len(validation_dataset),
                "Classes": len(class_to_idx),
            }
        )

    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        thousands_separator=True,
    ):
        print(pl.DataFrame(datasets_stats).sort(by="Name"))


######
# Misc
######


@task
def benchmark_append(ctx, fn, suffix, gpu_id=0):
    """
    Append models to benchmark
    """

    # CPU
    ctx.run(
        f"python benchmark.py --filter '{fn}' --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # CPU single thread
    ctx.run(
        f"python benchmark.py --filter '{fn}' --repeats 2 --bench-iter 60 --single-thread "
        f"--suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # Compiled CPU
    ctx.run(
        f"python benchmark.py --filter '{fn}' --compile --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # Compiled CPU with AMP
    ctx.run(
        f"python benchmark.py --filter '{fn}' --compile --amp --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # CUDA
    ctx.run(
        f"python benchmark.py --filter '{fn}' --bench-iter 50 --max-batch-size 512 "
        f"--gpu --gpu-id {gpu_id} --fast-matmul --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # Compiled CUDA
    ctx.run(
        f"python benchmark.py --filter '{fn}' --bench-iter 50 --max-batch-size 512 "
        f"--compile --gpu --gpu-id {gpu_id} --fast-matmul --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # Compiled CUDA with AMP
    ctx.run(
        f"python benchmark.py --filter '{fn}' --bench-iter 50 --max-batch-size 512 "
        f"--compile --gpu --gpu-id {gpu_id} --amp --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )

    # CUDA Memory
    ctx.run(
        f"python benchmark.py --filter '{fn}' --max-batch-size 1 "
        f"--gpu --gpu-id {gpu_id} --fast-matmul --memory --suffix {suffix} --append",
        echo=True,
        pty=True,
        warn=True,
    )


@task
def model_pre_publish(_ctx, model, net_param=None, tag=None, epoch=None, reparameterized=False, hf=True):
    """
    Generate data required for publishing a model
    """

    if net_param is not None:
        net_param = float(net_param)

    network_name = fs_ops.get_network_name(model, net_param, tag)

    (net, model_info) = fs_ops.load_model(
        torch.device("cpu"),
        model,
        net_param=net_param,
        tag=tag,
        epoch=epoch,
        inference=True,
        reparameterized=reparameterized,
    )
    num_params = sum(p.numel() for p in net.parameters())
    num_params = round(num_params / 1_000_000, 1)
    size = lib.get_size_from_signature(model_info.signature)
    num_outputs = lib.get_num_labels_from_signature(model_info.signature)

    # Check if model already in manifest
    if registry.pretrained_exists(network_name) is False:
        echo("Model not in manifest, generating ModelMetadata information")
        path = fs_ops.model_path(network_name, epoch=epoch)
        file_size = pathlib.Path(path).stat().st_size
        file_size = round(file_size / 1024 / 1024, 1)
        sha256 = cli.calc_sha256(path)

        print(f'"resolution": {size}')
        print(f'"file_size": {file_size}')
        print(f'"sha256": "{sha256}"')

    # Check if HF model card already exist
    if hf is True:
        model_card_path = pathlib.Path(HF_DOCS_DIR).joinpath(f"{network_name}.md")
        if model_card_path.exists() is False:
            echo("Model card does not exist, creating from template")
            with open(HF_MODEL_CARD_TEMPLATE, mode="r", encoding="utf-8") as handle:
                template_str = handle.read()

            template = Template(template_str)
            if isinstance(net, DetectorBackbone):
                detector_backbone = True
                out = net.detection_features(torch.rand((1, 3, *size)))
                feature_map_shapes = [(k, v.size()) for k, v in out.items()]
            else:
                detector_backbone = False
                print("Feature maps not supported")
                feature_map_shapes = []

            model_card = template.render(
                model_name=network_name,
                num_params=num_params,
                size=size,
                num_outputs=num_outputs,
                embedding_size=net.embedding_size,
                detector_backbone=detector_backbone,
                feature_map_shapes=feature_map_shapes,
            )

            echo(f"Writing model card at {model_card_path}...")
            with open(model_card_path, mode="w", encoding="utf-8") as handle:
                handle.write(model_card)


@task
def sam_from_vit(_ctx, network, tag=None, epoch=None):
    """
    Transform vanilla ViT to ViT SAM
    """

    # Assuming network is vit_{b, l, h}16 or vitregN_{b, l, h}16
    if "reg" in network:
        sam_network = network[0:3] + "_sam" + network[7:]
    else:
        sam_network = network[0:3] + "_sam" + network[3:]

    if tag is not None:
        sam_network_tagged = sam_network + f"_{tag}"
    else:
        sam_network_tagged = sam_network

    path = fs_ops.model_path(sam_network_tagged, epoch=epoch)
    if path.exists() is True:
        echo(f"{path} already exists")
        echo("Aborting", color=COLOR_RED)
        raise Exit(code=1)

    # Load model
    device = torch.device("cpu")
    (net, model_info) = fs_ops.load_model(device, network, tag=tag, epoch=epoch, inference=False)
    size = lib.get_size_from_signature(model_info.signature)
    channels = lib.get_channels_from_signature(model_info.signature)

    sam = registry.net_factory(sam_network, channels, len(model_info.class_to_idx), size=size)
    sam.load_vit_weights(net.state_dict())

    # Save model
    fs_ops.checkpoint_model(
        sam_network_tagged,
        epoch,
        sam,
        model_info.signature,
        model_info.class_to_idx,
        model_info.rgb_stats,
        optimizer=None,
        scheduler=None,
        scaler=None,
        model_base=None,
    )


@task
def hieradet_from_hiera(_ctx, network, tag=None, epoch=None):
    """
    Transform vanilla Hiera to HieraDet
    """

    if "abswin" not in network:
        raise ValueError("Only abswin variant is supported")

    hieradet_network = network[0:5] + "det" + network[12:]
    if tag is not None:
        hieradet_network_tagged = hieradet_network + f"_{tag}"
    else:
        hieradet_network_tagged = hieradet_network

    path = fs_ops.model_path(hieradet_network_tagged, epoch=epoch)
    if path.exists() is True:
        echo(f"{path} already exists")
        echo("Aborting", color=COLOR_RED)
        raise Exit(code=1)

    # Load model
    device = torch.device("cpu")
    (net, model_info) = fs_ops.load_model(device, network, tag=tag, epoch=epoch, inference=False)
    size = lib.get_size_from_signature(model_info.signature)
    channels = lib.get_channels_from_signature(model_info.signature)

    hieradet = registry.net_factory(hieradet_network, channels, len(model_info.class_to_idx), size=size)
    hieradet.load_hiera_weights(net.state_dict())

    # Save model
    fs_ops.checkpoint_model(
        hieradet_network_tagged,
        epoch,
        hieradet,
        model_info.signature,
        model_info.class_to_idx,
        model_info.rgb_stats,
        optimizer=None,
        scheduler=None,
        scaler=None,
        model_base=None,
    )
