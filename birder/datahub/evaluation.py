from pathlib import Path

from birder.datahub._lib import download_url
from birder.datahub._lib import extract_archive


class AwA2:
    """
    Name: AwA2 (Animals with Attributes 2)
    Link: https://cvml.ista.ac.at/AwA2/
    Size: 50 animal classes, 37,322 images, 85 binary attributes per class
    """

    attribute_names = [
        "black",
        "white",
        "blue",
        "brown",
        "gray",
        "orange",
        "red",
        "yellow",
        "patches",
        "spots",
        "stripes",
        "furry",
        "hairless",
        "toughskin",
        "big",
        "small",
        "bulbous",
        "lean",
        "flippers",
        "hands",
        "hooves",
        "pads",
        "paws",
        "longleg",
        "longneck",
        "tail",
        "chewteeth",
        "meatteeth",
        "buckteeth",
        "strainteeth",
        "horns",
        "claws",
        "tusks",
        "smelly",
        "flys",
        "hops",
        "swims",
        "tunnels",
        "walks",
        "fast",
        "slow",
        "strong",
        "weak",
        "muscle",
        "bipedal",
        "quadrapedal",
        "active",
        "inactive",
        "nocturnal",
        "hibernate",
        "agility",
        "fish",
        "meat",
        "plankton",
        "vegetation",
        "insects",
        "forager",
        "grazer",
        "hunter",
        "scavenger",
        "skimmer",
        "stalker",
        "newworld",
        "oldworld",
        "arctic",
        "coastal",
        "desert",
        "bush",
        "plains",
        "forest",
        "fields",
        "jungle",
        "mountains",
        "ocean",
        "ground",
        "water",
        "tree",
        "cave",
        "fierce",
        "timid",
        "smart",
        "group",
        "solitary",
        "nestspot",
        "domestic",
    ]

    def __init__(self, root: str | Path, download: bool = False, progress_bar: bool = True) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        if download is True:
            archive_path = root.parent.joinpath("AwA2-data.zip")
            downloaded = download_url(
                "https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
                archive_path,
                sha256="cc5a849879165acaa2b52f1de3f146ffcd1c475f6ef85bab0152c763e573744f",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(archive_path, root.parent)

        else:
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            if self.images_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: JPEGImages directory not found")

            if self.predicate_matrix_binary_path.exists() is False:
                raise RuntimeError("Dataset seems corrupted: predicate-matrix-binary.txt not found")

    @property
    def images_dir(self) -> Path:
        return self._root.joinpath("JPEGImages")

    @property
    def classes_path(self) -> Path:
        return self._root.joinpath("classes.txt")

    @property
    def predicates_path(self) -> Path:
        return self._root.joinpath("predicates.txt")

    @property
    def predicate_matrix_binary_path(self) -> Path:
        return self._root.joinpath("predicate-matrix-binary.txt")

    @property
    def predicate_matrix_continuous_path(self) -> Path:
        return self._root.joinpath("predicate-matrix-continuous.txt")

    @property
    def trainclasses_path(self) -> Path:
        return self._root.joinpath("trainclasses.txt")

    @property
    def testclasses_path(self) -> Path:
        return self._root.joinpath("testclasses.txt")


class FishNet:
    """
    Name: FishNet
    Link: https://fishnet-2023.github.io/
    Size: 94,532 images, 17,357 aquatic species, 9 binary traits

    Traits:
        - FeedingPath (benthic=0, pelagic=1)
        - Tropical, Temperate, Subtropical, Boreal, Polar (habitat, 0/1)
        - freshwater, saltwater, brackish (water type, 0/1)

    Note: This dataset requires manual download from Google Drive.
    """

    trait_columns = [
        "FeedingPath",
        "Tropical",
        "Temperate",
        "Subtropical",
        "Boreal",
        "Polar",
        "freshwater",
        "saltwater",
        "brackish",
    ]

    def __init__(self, root: str | Path) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        # Verify dataset exists
        if self._root.exists() is False or self._root.is_dir() is False:
            raise RuntimeError(f"Dataset not found at {self._root}. Download it from https://fishnet-2023.github.io/")

        if self.images_dir.exists() is False:
            raise RuntimeError("Dataset seems corrupted: images directory not found")

        if self.train_csv.exists() is False:
            raise RuntimeError("Dataset seems corrupted: train.csv not found")

        if self.test_csv.exists() is False:
            raise RuntimeError("Dataset seems corrupted: test.csv not found")

    @property
    def images_dir(self) -> Path:
        return self._root.joinpath("images")

    @property
    def train_csv(self) -> Path:
        return self._root.joinpath("train.csv")

    @property
    def test_csv(self) -> Path:
        return self._root.joinpath("test.csv")


class FungiCLEF2023:
    """
    Name: FungiCLEF2023
    Link: https://www.imageclef.org/FungiCLEF2023
    Size: 1,604 species, ~417K images (train + val + test)
    """

    def __init__(self, root: str | Path, download: bool = False, progress_bar: bool = True) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        if download is True:
            self._root.mkdir(parents=True, exist_ok=True)

            train_images_archive = self._root.joinpath("DF20-300px.tar.gz")
            downloaded_train_images = download_url(
                "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF20-300px.tar.gz",
                train_images_archive,
                sha256="b7b572179c3e99dfdfaed4b75872cb6cc59ad8d7dccab331906687ca6bce3b5a",
                progress_bar=progress_bar,
            )
            if downloaded_train_images is True or self.train_images_dir.exists() is False:
                extract_archive(train_images_archive, self._root)

            val_test_images_archive = self._root.joinpath("DF21_300px.tar.gz")
            downloaded_val_test_images = download_url(
                "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/DF21_300px.tar.gz",
                val_test_images_archive,
                sha256="c0194d3314370a22fb01fb0800330c2e18c90d83f97def55dea84cb5abc2fc3e",
                progress_bar=progress_bar,
            )
            if downloaded_val_test_images is True or self.val_test_images_dir.exists() is False:
                extract_archive(val_test_images_archive, self._root)

            download_url(
                "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/FungiCLEF2023_train_metadata_PRODUCTION.csv",
                self.train_metadata_path,
                sha256="dc17fc1ab48f0876947402965ee9c25e437c1622f134edab5c7da6c9b853d907",
                progress_bar=progress_bar,
            )
            download_url(
                "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/FungiCLEF2023_val_metadata_PRODUCTION.csv",
                self.val_metadata_path,
                sha256="9573102de721bc93f36e5e03e878cd50cc7f6031a7a3bc82ed0642ec4c691c2a",
                progress_bar=progress_bar,
            )
            download_url(
                "http://ptak.felk.cvut.cz/plants/DanishFungiDataset/FungiCLEF2023_public_test_metadata_PRODUCTION.csv",
                self.test_metadata_path,
                sha256="56ae171d5abf2a99a3ccf8cd96cb685d0f96a7bda055a37afd2fda3e943d991c",
                progress_bar=progress_bar,
            )

        else:
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            if self.train_images_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: DF20_300 directory not found")

            if self.val_test_images_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: DF21_300 directory not found")

            if self.train_metadata_path.exists() is False:
                raise RuntimeError("Dataset seems corrupted: train metadata CSV not found")

            if self.val_metadata_path.exists() is False:
                raise RuntimeError("Dataset seems corrupted: validation metadata CSV not found")

            if self.test_metadata_path.exists() is False:
                raise RuntimeError("Dataset seems corrupted: test metadata CSV not found")

    @property
    def train_images_dir(self) -> Path:
        return self._root.joinpath("DF20_300")

    @property
    def val_test_images_dir(self) -> Path:
        return self._root.joinpath("DF21_300")

    @property
    def val_images_dir(self) -> Path:
        return self.val_test_images_dir

    @property
    def test_images_dir(self) -> Path:
        return self.val_test_images_dir

    @property
    def train_metadata_path(self) -> Path:
        return self._root.joinpath("FungiCLEF2023_train_metadata_PRODUCTION.csv")

    @property
    def val_metadata_path(self) -> Path:
        return self._root.joinpath("FungiCLEF2023_val_metadata_PRODUCTION.csv")

    @property
    def test_metadata_path(self) -> Path:
        return self._root.joinpath("FungiCLEF2023_public_test_metadata_PRODUCTION.csv")


class NABirds:
    """
    Name: NABirds
    Link: https://dl.allaboutbirds.org/nabirds
    Size: 555 visual categories, ~48K images

    Note: This dataset requires manual download. Visit the link above.
    """

    def __init__(self, root: str | Path) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        # Verify dataset exists
        if self._root.exists() is False or self._root.is_dir() is False:
            raise RuntimeError(
                f"Dataset not found at {self._root}. Download it from https://dl.allaboutbirds.org/nabirds"
            )

        if self.images_dir.exists() is False:
            raise RuntimeError("Dataset seems corrupted: images directory not found")

    @property
    def images_dir(self) -> Path:
        return self._root.joinpath("images")

    @property
    def images_path(self) -> Path:
        return self._root.joinpath("images.txt")

    @property
    def classes_path(self) -> Path:
        return self._root.joinpath("classes.txt")

    @property
    def labels_path(self) -> Path:
        return self._root.joinpath("image_class_labels.txt")

    @property
    def train_test_split_path(self) -> Path:
        return self._root.joinpath("train_test_split.txt")

    @property
    def hierarchy_path(self) -> Path:
        return self._root.joinpath("hierarchy.txt")


class NeWT:
    """
    Name: NeWT (Natural World Tasks)
    Link: https://github.com/visipedia/newt
    Size: 164 binary classification tasks, ~36K images
    """

    def __init__(self, root: str | Path, download: bool = False, progress_bar: bool = True) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        if download is True:
            self._root.mkdir(parents=True, exist_ok=True)

            # Download images
            images_src = root.parent.joinpath("newt2021_images.tar.gz")
            downloaded_images = download_url(
                "https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_images.tar.gz",
                images_src,
                sha256="8d40958a867c1296f92b5e125f1f1d8ddaa59f249315740fc366fc606995c055",
                progress_bar=progress_bar,
            )
            if downloaded_images is True or self.images_dir.exists() is False:
                extract_archive(images_src, self._root)

            # Download labels
            labels_src = root.parent.joinpath("newt2021_labels.csv.tar.gz")
            downloaded_labels = download_url(
                "https://ml-inat-competition-datasets.s3.amazonaws.com/newt/newt2021_labels.csv.tar.gz",
                labels_src,
                sha256="e09807842485ef49ccf51d74ac9f6072c599fc16cf5ee755fdf4064f2e4c3828",
                progress_bar=progress_bar,
            )
            if downloaded_labels is True or self.labels_path.exists() is False:
                extract_archive(labels_src, self._root)

        else:
            # Some sanity checks
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            if self.images_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: images directory not found")

            if self.labels_path.exists() is False:
                raise RuntimeError("Dataset seems corrupted: labels CSV not found")

    @property
    def images_dir(self) -> Path:
        return self._root.joinpath("newt2021_images")

    @property
    def labels_path(self) -> Path:
        return self._root.joinpath("newt2021_labels.csv")


class Plankton:
    """
    Name: SYKE-plankton_IFCB_2022
    Link: https://b2share.eudat.eu/records/xvnrp-7ga56
    Size: 50 phytoplankton classes, ~214K images (train + val)
    """

    def __init__(self, root: str | Path, download: bool = False, progress_bar: bool = True) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        if download is True:
            self._root.mkdir(parents=True, exist_ok=True)

            train_archive = root.parent.joinpath("phytoplankton_labeled.zip")
            downloaded_train = download_url(
                "https://b2share.eudat.eu/records/xvnrp-7ga56/files/phytoplankton_labeled.zip",
                train_archive,
                sha256="0c47acd8dfad46829fe42758a6c24adcdb5e6f2456be4ced975cbb9de9644704",
                progress_bar=progress_bar,
            )
            if downloaded_train is True or self.train_dir.exists() is False:
                extract_archive(train_archive, self._root)

            val_archive = root.parent.joinpath("phytoplankton_Uto_2021_labeled.zip")
            downloaded_val = download_url(
                "https://b2share.eudat.eu/records/w7y96-6jd66/files/phytoplankton_Ut%C3%B6_2021_labeled.zip",
                val_archive,
                sha256="b017809515c3d58171ecbfd196d6725239e9380c2a22ae880ac56e878bbfcfa4",
                progress_bar=progress_bar,
            )
            if downloaded_val is True or self.val_dir.exists() is False:
                extract_archive(val_archive, self._root)
                # Rename extracted directory to avoid non-ASCII character
                extracted_dir = self._root.joinpath("phytoplankton_Utö_2021_labeled")
                if extracted_dir.exists():
                    extracted_dir.rename(self.val_dir)

        else:
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            if self.train_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: train directory not found")

            if self.val_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: val directory not found")

    @property
    def train_dir(self) -> Path:
        return self._root.joinpath("labeled_20201020")

    @property
    def val_dir(self) -> Path:
        return self._root.joinpath("phytoplankton_Uto_2021_labeled")


class PlantDoc:
    """
    Name: PlantDoc
    Link: https://github.com/pratikkayal/PlantDoc-Dataset
    Paper: https://arxiv.org/abs/1911.10317
    Size: 27 classes (13 plant species, 17 disease categories), 2,598 images
    """

    _archive_dir_name = "PlantDoc-Dataset-5467f6012d78d1c446145d5f582da6096f852ae8"

    def __init__(self, root: str | Path, download: bool = False, progress_bar: bool = True) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        if download is True:
            archive_path = root.parent.joinpath("plantdoc.zip")
            downloaded = download_url(
                "https://github.com/pratikkayal/PlantDoc-Dataset/archive/"
                "5467f6012d78d1c446145d5f582da6096f852ae8.zip",
                archive_path,
                sha256="94e2b99a500a63efbd48923ed48588fbb01f9b1db66a2d3b5c24eed6466da20f",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(archive_path, root.parent)
                # Rename extracted directory from commit hash name to friendly name
                extracted_dir = root.parent.joinpath(self._archive_dir_name)
                extracted_dir.rename(self._root)

        else:
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            if self.train_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: train directory not found")

            if self.test_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: test directory not found")

    @property
    def train_dir(self) -> Path:
        return self._root.joinpath("train")

    @property
    def test_dir(self) -> Path:
        return self._root.joinpath("test")


class PlantNet:
    """
    Name: PlantNet-300K
    Link: https://plantnet.org/en/2021/03/30/a-plntnet-dataset-for-machine-learning-researchers/
    Size: 1081 species, ~300K images
    """

    def __init__(self, root: str | Path, download: bool = False, progress_bar: bool = True) -> None:
        if isinstance(root, str):
            root = Path(root)

        self._root = root

        if download is True:
            archive_path = root.parent.joinpath("plantnet_300K.zip")
            downloaded = download_url(
                "https://zenodo.org/records/5645731/files/plantnet_300K.zip?download=1",
                archive_path,
                sha256="3a079076c8ad4476beac54d89ea344958256a999428937eba47ec352dadce00d",
                progress_bar=progress_bar,
            )
            if downloaded is True or self._root.exists() is False:
                extract_archive(archive_path, root.parent)

        else:
            if self._root.exists() is False or self._root.is_dir() is False:
                raise RuntimeError("Dataset not found, try download=True to download it")

            if self.images_dir.exists() is False:
                raise RuntimeError("Dataset seems corrupted: images directory not found")

            if self.species_id_to_name_path.exists() is False:
                raise RuntimeError("Dataset seems corrupted: species_id_2_name.json not found")

    @property
    def images_dir(self) -> Path:
        return self._root.joinpath("images")

    @property
    def train_dir(self) -> Path:
        return self.images_dir.joinpath("train")

    @property
    def val_dir(self) -> Path:
        return self.images_dir.joinpath("val")

    @property
    def test_dir(self) -> Path:
        return self.images_dir.joinpath("test")

    @property
    def species_id_to_name_path(self) -> Path:
        return self._root.joinpath("plantnet300K_species_id_2_name.json")

    @property
    def metadata_path(self) -> Path:
        return self._root.joinpath("plantnet300K_metadata.json")
