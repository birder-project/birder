# type: ignore
import pathlib
import time

import polars as pl
from invoke import Exit
from invoke import task

from birder.common import fs_ops
from birder.conf import settings
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
        f"python -m pylint *.py tests {PROJECT_DIR}",
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

    annotations_status = pl.read_csv("annotations_status.csv")
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
    annotations_status.write_csv("annotations_status.csv")
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
        doc = "\n".join(class_list)

        echo("data/detection_data/classes.txt")
        with open("data/detection_data/classes.txt", "w", encoding="utf-8") as handle:
            handle.write(doc)

    echo(f"Done, added {len(class_list) - len(class_to_idx)} new classes")


@task
def convert_to_coco(ctx):
    """
    Convert labelme annotations to coco format for both training and validation sets
    """

    ctx.run(
        "python tool.py labelme-to-coco data/detection_data/training_annotations",
        echo=True,
        pty=True,
        warn=True,
    )
    ctx.run(
        "python tool.py labelme-to-coco data/detection_data/validation_annotations",
        echo=True,
        pty=True,
        warn=True,
    )


@task
def pack_intermediate(ctx, size=384, suffix=settings.PACK_PATH_SUFFIX):
    """
    Pack data for intermediate training
    """

    ctx.run(
        f"python tool.py pack -j 12 --shuffle --suffix {suffix} --size {size} data/training data/raw_data",
        echo=True,
        pty=True,
        warn=True,
    )
    ctx.run(
        f"python tool.py pack -j 12 --suffix {suffix} --size {size} --max-size 200 --class-file "
        "data/training_packed/classes.txt data/validation data/raw_data_validation",
        echo=True,
        pty=True,
        warn=True,
    )


@task
def pack_il_common(ctx, size=384):
    """
    Pack il-common dataset
    """

    ctx.run(
        f"python tool.py pack --type directory -j 8 --suffix il-common_packed --size {size} --format jpeg "
        "--class-file data/il-common_classes.txt data/training",
        echo=True,
        pty=True,
        warn=True,
    )
    ctx.run(
        f"python tool.py pack --type directory -j 8 --suffix il-common_packed --size {size} --format jpeg "
        "--class-file data/il-common_classes.txt data/validation",
        echo=True,
        pty=True,
        warn=True,
    )
