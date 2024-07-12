# Pack

The `pack` tool allows you to convert a directory of images into WebDataset format, which is optimized for efficient data loading during training. This tool is essential for preparing large datasets for use with Birder.

## Usage

```sh
python -m birder.tools pack [OPTIONS] DATA_PATH...
```

To see all available options and get detailed help, run:

```sh
python -m birder.tools pack --help
```

This will display a comprehensive list of all options and their descriptions, ensuring you have access to the full functionality of the `pack` tool.

## Description

The pack tool provides several key features for dataset preparation:

* **WebDataset conversion**: Transforms image directories into the WebDataset format
* **Multi-processing**: Utilizes multiple CPU cores for faster processing
* **Image resizing**: Option to resize images to a specified maximum dimension while maintaining aspect ratio
* **Class list generation**: Automatically generates a class list file or uses an existing one
* **Random shuffling**: Shuffles the dataset during packing for better training dynamics

If no resize is specified, files are archived as-is. If resize is defined, all files are re-encoded to WebP format.

## Output

The tool creates a new directory with the name of the first input directory plus the specified suffix. This directory will contain:

* WebDataset shards (.tar files)
* A classes.txt file listing all classes in the dataset
