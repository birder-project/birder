import os
import pathlib
import time

import numpy as np
import pandas as pd
from invoke import Exit
from invoke import task

from birder.common import util
from birder.conf import settings
from birder.core.net.base import get_net_class
from frontend.server.server import settings as server_settings

if pathlib.Path(__file__).parent != pathlib.Path().resolve():
    print("Can only run from the root directory, aborting...")
    raise SystemExit

COLOR_GRAY = 37
COLOR_GREEN = 32
COLOR_RED = 91
DEFAULT_COLOR = COLOR_GRAY

BIRDER_DIR = "birder"
BIRDER_TEST_DIR = "tests"
FRONTEND_SERVER_DIR = "frontend/server"
FRONTEND_CLIENT_DIR = "frontend/client"

VERSION = "0.1.0"
CONTAINER_REGISTRY = "192.168.122.11:5000"


def echo(msg, color=DEFAULT_COLOR):
    print(f"\033[1;{color}m{msg}\033[0m")


def is_venv():
    if os.getenv("VIRTUAL_ENV"):
        return True

    return False


@task
def setup(ctx):
    """
    Install all requirements (Python and JS)
    """

    if is_venv() is True:
        ctx.run("pip3 install --upgrade pip", echo=True, pty=True)
        ctx.run("pip3 install -r requirements/requirements-dev.txt", echo=True, pty=True)
        ctx.run("pip3 install 'mxnet~=1.7.0'", echo=True, pty=True)
        ctx.run("pip3 install opencv-python opencv-contrib-python", echo=True, pty=True)

    else:
        echo("venv not active, skipping pip steps", color=COLOR_RED)

    if pathlib.Path(FRONTEND_CLIENT_DIR).resolve().joinpath("node_modules").is_dir() is True:
        echo("node_modules already exists, skipping npm steps", color=COLOR_RED)

    else:
        with ctx.cd(FRONTEND_CLIENT_DIR):
            ctx.run("npm install && npm dedupe", echo=True, pty=True)


@task
def ci(ctx):  # pylint: disable=invalid-name
    """
    Run all linters and tests, set return code 0 only if everything succeeded.
    """

    tic = time.time()

    return_code = 0
    if csslint(ctx) != 0:
        return_code = 1

    if jslint(ctx) != 0:
        return_code = 1

    if jstest(ctx) != 0:
        return_code = 1

    if pylint(ctx) != 0:
        return_code = 1

    if pytest(ctx) != 0:
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
def csslint(ctx):
    """
    Run style lint on all relevant files
    """

    return_code = 0

    with ctx.cd(FRONTEND_CLIENT_DIR):
        result = ctx.run("npm run lint:css", echo=True, pty=True, warn=True)
        if result.exited != 0:
            return_code = 1
            echo("Failed", color=COLOR_RED)

        else:
            echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def jslint(ctx):
    """
    Run eslint and formatting check on the client frontend
    """

    return_code = 0

    with ctx.cd(FRONTEND_CLIENT_DIR):
        result = ctx.run("npm run lint:format", echo=True, pty=True, warn=True)
        if result.exited != 0:
            return_code = 1
            echo("Failed", color=COLOR_RED)

        else:
            echo("Passed", color=COLOR_GREEN)

    with ctx.cd(FRONTEND_CLIENT_DIR):
        result = ctx.run("npm run lint", echo=True, pty=True, warn=True)
        if result.exited != 0:
            return_code = 1
            echo("Failed", color=COLOR_RED)

        else:
            echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def jstest(ctx):
    """
    Run javascript tests
    """

    return_code = 0

    with ctx.cd(FRONTEND_CLIENT_DIR):
        result = ctx.run("npm run test:unit", echo=True, pty=True, warn=True)
        if result.exited != 0:
            return_code = 1
            echo("Failed", color=COLOR_RED)

        else:
            echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def pylint(ctx):
    """
    Run pylint & flake8 on all Python files, type check and formatting check
    """

    return_code = 0

    # pylint
    result = ctx.run(
        f"python3 -m pylint *.py {BIRDER_DIR} {BIRDER_TEST_DIR} "
        f"{FRONTEND_SERVER_DIR}/*.py {FRONTEND_SERVER_DIR}/*/",
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
    result = ctx.run("python3 -m flake8 .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    else:
        echo("Passed", color=COLOR_GREEN)

    # mypy type checking
    result = ctx.run(
        f"python3 -m mypy --pretty . {FRONTEND_SERVER_DIR}",
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
    result = ctx.run("python3 -m black --check .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    else:
        echo("Passed", color=COLOR_GREEN)

    # Import check, isort
    result = ctx.run("python3 -m isort --check-only .", echo=True, pty=True, warn=True)
    if result.exited != 0:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    else:
        echo("Passed", color=COLOR_GREEN)

    return return_code


@task
def pytest(ctx):
    """
    Run python tests
    """

    return_code = 0

    result = ctx.run("python3 -m unittest discover -s tests -v", echo=True, pty=True, warn=True)

    if result.exited == 0:
        echo("Passed", color=COLOR_GREEN)

    else:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    with ctx.cd(FRONTEND_SERVER_DIR):
        result = ctx.run("python3 manage.py test -v 2", echo=True, pty=True, warn=True)

    if result.exited == 0:
        echo("Passed", color=COLOR_GREEN)

    else:
        return_code = 1
        echo("Failed", color=COLOR_RED)

    return return_code


@task
def gen_classes_doc(_ctx):
    """
    Generates docs/classes.md from synset.txt and synset.py for i18n
    """

    classes = util.read_synset()

    doc = ""
    doc += "# Classes\n\n"
    doc += f"Currently supporting the following {len(classes)} classes:\n\n"

    for key in classes:
        doc += f"* {key}\n"

    echo("Writing docs/classes.md")
    with open("docs/classes.md", "w") as handle:
        handle.write(doc)

    doc = ""
    doc += "from django.utils.translation import gettext_lazy as _\n\n"
    doc += "SYNSET = [\n"
    for key in classes:
        doc += f'    _("{key}"),\n'

    doc += "]\n"

    echo(f"Writing {FRONTEND_SERVER_DIR}/inference/synset.py")
    with open(f"{FRONTEND_SERVER_DIR}/inference/synset.py", "w") as handle:
        handle.write(doc)

    echo("Done !", color=COLOR_GREEN)


@task
def update_annotation_table(_ctx):
    """
    Add new classes to the annotations status table
    """

    class_list = util.read_synset_as_list()
    classes = pd.Series(class_list)
    column_class = "class"

    annotations_status = pd.read_csv("annotations_status.csv")  # type: pd.DataFrame
    new_classes = classes[~classes.isin(annotations_status[column_class])].values
    if len(new_classes) == 0:
        echo("Nothing to do")
        return

    new_annotations_status = pd.DataFrame(new_classes, columns=[column_class])
    for column in annotations_status:
        if column == column_class:
            continue

        new_annotations_status[column] = np.zeros_like(new_classes)

    annotations_status = annotations_status.append(new_annotations_status)
    annotations_status.sort_values(by=column_class, inplace=True)
    annotations_status.reset_index(drop=True, inplace=True)
    annotations_status.to_csv("annotations_status.csv", index=False)

    echo(f"Done, added {len(new_classes)} new classes")


@task
def update_classes(ctx):
    """
    Convenient that generate synset and runs gen-classes-doc and update-annotation-table
    """

    ctx.run("python3 pack.py --only-write-synset data/", echo=True, pty=True)
    ctx.run("inv gen-classes-doc update-annotation-table", echo=True, pty=True)


@task(help={"val-size": "image size for the validation set"})
def pack(ctx, val_size=224, skip_val=False):
    """
    Runs pack both on training and validation sets
    """

    ctx.run(
        f"python3 pack.py --size 360 --jobs -1 --shuffle --write-synset {settings.DATA_DIR}",
        echo=True,
        pty=True,
    )

    if skip_val is False:
        ctx.run(f"python3 pack.py --size {val_size} --jobs -1 {settings.VAL_DATA_DIR}", echo=True, pty=True)


@task
def aug(ctx):
    """
    Run default augmentation on the data directory
    """

    ctx.run("python3 tool.py aug --jobs -1", echo=True, pty=True)


@task
def clean(ctx, no_aug=False, no_adv=False, no_rec=False, no_val=False):
    """
    Deletes all augmentation and rec files from the data directory
    """

    if no_rec is False:
        # Remove rec files
        ctx.run(f"rm -f {settings.DATA_PATH}", echo=True, pty=True, warn=True)
        ctx.run(f"rm -f {settings.DATA_PATH[:-3] + 'idx'}", echo=True, pty=True, warn=True)

        if no_val is False:
            ctx.run(f"rm -f {settings.VAL_PATH}", echo=True, pty=True, warn=True)
            ctx.run(f"rm -f {settings.VAL_PATH[:-3] + 'idx'}", echo=True, pty=True, warn=True)

    if no_aug is False:
        # Remove augmentations
        ctx.run(f'find {settings.DATA_DIR} -name "aug_*.jpeg" -delete', echo=True, pty=True, warn=True)

    if no_adv is False:
        # Remove augmentations
        ctx.run(f'find {settings.DATA_DIR} -name "adv_*.jpeg" -delete', echo=True, pty=True, warn=True)

    # Remove model archive staging
    ctx.run(f"rm -f {settings.MODEL_STAGING_DIR}/*", echo=True, pty=True, warn=True)


@task(help={"network": "the neural network to train"})
def train(ctx, network, num_layers=None):
    """
    Train network end to end

    Example:
        inv train inception_resnet_v2
        inv train shufflenet_v2 --num-layers 2.0
    """

    net_class = get_net_class(network, 0, 0)  # Used only to get the default size
    val_size = net_class.default_size

    extra_arg = ""
    network_final_name = network
    if num_layers is not None:
        extra_arg = f"--num-layers {num_layers}"
        network_final_name = f"{network}_{num_layers}"

    ctx.run(f"inv clean aug pack --val-size {val_size}", echo=True, pty=True)
    ctx.run(
        f"python3 train.py --network {network} {extra_arg} --mixup-beta 0.2 --smooth-alpha 0.1",
        echo=True,
        pty=True,
    )
    last_epoch = settings.NUM_EPOCHS
    for num_epoch in settings.RESTART_EPOCHS:
        ctx.run("inv clean --no-val aug pack --skip-val", echo=True, pty=True)
        ctx.run(
            f"python3 train.py --network {network_final_name} "
            f"--resume-epoch {last_epoch} --num-epochs {num_epoch} "
            f"--lr 0.05 --final-lr 0.001 --smooth-alpha 0.1",
            echo=True,
            pty=True,
        )

        last_epoch = num_epoch

    ctx.run(
        f"python3 predict.py --network {network_final_name} {settings.VAL_DATA_DIR}/*/* --gpu",
        echo=True,
        pty=True,
    )


@task(help={"network": "the neural network to archive (e.g. shufflenet_v2_2.0)"})
def build_mar(ctx, network):
    """
    Collects and archives a model
    """

    classes = util.read_synset()

    ctx.run(f"python3 tool.py collect-model --network {network}", echo=True, pty=True)
    ctx.run(
        f"model-archiver --model-name {network}_{len(classes)} --model-path {settings.MODEL_STAGING_DIR} "
        f"--handler classification:handle --export-path docker/mar",
        echo=True,
        pty=True,
    )
    ctx.run("inv clean --no-adv --no-aug --no-rec", echo=True, pty=True)

    echo("Done, archive can be found at 'docker/mar'")


@task
def make_messages(ctx):
    """
    Generate po files for the frontend application server
    """

    locale_path = pathlib.Path(f"{FRONTEND_SERVER_DIR}/inference/locale")
    if locale_path.is_dir() is False:
        locale_path.mkdir()

    for lang_code, _ in server_settings.LANGUAGES:
        locale_path.joinpath(lang_code).mkdir(exist_ok=True)

    with ctx.cd(FRONTEND_SERVER_DIR):
        with ctx.cd("inference"):
            ctx.run("python3 ../manage.py makemessages --all", echo=True, pty=True)


@task
def compile_messages(ctx):
    """
    Compile translation files
    """

    with ctx.cd(FRONTEND_SERVER_DIR):
        with ctx.cd("inference"):
            ctx.run("python3 ../manage.py compilemessages", echo=True, pty=True)


@task
def missing_messages(ctx, lang):
    """
    Show string with missing translation for language
    """

    with ctx.cd(FRONTEND_SERVER_DIR):
        ctx.run(
            f"msgattrib --untranslated inference/locale/{lang}/LC_MESSAGES/django.po", echo=True, pty=True
        )


@task
def destroy(ctx):
    """
    Destroy all images and containers
    """

    ctx.run("docker-compose down --rmi all", echo=True, pty=True)


@task
def prod_build(ctx):
    """
    Build all production containers
    """

    if pathlib.Path(f"{FRONTEND_SERVER_DIR}/config.json").exists() is False:
        echo("Frontend config.json not found, aborting...", color=COLOR_RED)
        raise SystemExit

    ctx.run("docker-compose build", echo=True, pty=True)
    ctx.run("docker-compose up --no-start", echo=True, pty=True)


@task
def push_images(ctx):
    """
    Push all containers to defined registry
    """

    for container in ["birder-api-http", "birder-api", "birder-app", "birder-mms"]:
        ctx.run(
            f"docker tag {container}:{VERSION} {CONTAINER_REGISTRY}/{container}:{VERSION}",
            echo=True,
            pty=True,
        )
        ctx.run(f"docker push {CONTAINER_REGISTRY}/{container}:{VERSION}", echo=True, pty=True)
        ctx.run(f"docker image rm {CONTAINER_REGISTRY}/{container}:{VERSION}", echo=True, pty=True)


@task
def pull_images(ctx, registry=None):
    """
    Pull all containers from defined registry
    """

    if registry is None:
        registry = CONTAINER_REGISTRY

    for container in ["birder-api-http", "birder-api", "birder-app", "birder-mms"]:
        ctx.run(
            f"docker pull {registry}/{container}:{VERSION}",
            echo=True,
            pty=True,
        )
        ctx.run(
            f"docker tag {registry}/{container}:{VERSION} {container}:{VERSION}",
            echo=True,
            pty=True,
        )
        ctx.run(f"docker image rm {registry}/{container}:{VERSION}", echo=True, pty=True)


@task
def dev_build(ctx):
    """
    Build all development containers
    """

    ctx.run("docker-compose -f docker-compose.yml -f docker-compose.dev.yml build", echo=True, pty=True)
    ctx.run(
        "docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --no-start", echo=True, pty=True
    )


@task(
    iterable=["service"],
    help={"service": "service to start, can be called multiple times, default is all services"},
)
def dev_start(ctx, service):
    """
    Run dev containers
    """

    ctx.run(
        f"docker-compose -f docker-compose.yml -f docker-compose.dev.yml start {' '.join(service)}",
        echo=True,
        pty=True,
    )


@task(
    iterable=["service"],
    help={"service": "service to stop, can be called multiple times, default is all services"},
)
def dev_stop(ctx, service):
    """
    Stop dev containers
    """

    ctx.run(
        f"docker-compose -f docker-compose.yml -f docker-compose.dev.yml stop {' '.join(service)}",
        echo=True,
        pty=True,
    )


@task
def dev_status(ctx):
    """
    Show dev images and containers
    """

    ctx.run("docker-compose -f docker-compose.yml -f docker-compose.dev.yml images", echo=True, pty=True)
    echo("\n")
    ctx.run("docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps", echo=True, pty=True)


@task
def serve_api(ctx):
    """
    Run Django dev server
    """

    with ctx.cd(FRONTEND_SERVER_DIR):
        ctx.run("python3 manage.py runserver", echo=True, pty=True)


@task
def serve_app(ctx):
    """
    Run Vue dev server
    """

    with ctx.cd(FRONTEND_CLIENT_DIR):
        ctx.run("npm run serve", echo=True, pty=True)
