# Pack

The `pack` tool allows you to convert a directory of images into either WebDataset format or a structured directory format, which are optimized for efficient data loading during training. This tool is essential for preparing large datasets for use with Birder.

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

- **Multiple output formats**:
    - WebDataset format (default): Transforms image directories into the WebDataset format
    - Directory format: Creates a structured directory with class subdirectories
- **Multi-processing**: Utilizes multiple CPU cores for faster processing
- **Image resizing**: Option to resize images to a specified maximum dimension while maintaining aspect ratio
- **Image format conversion**: Ability to convert images to WebP, PNG, or JPEG formats when resizing
- **Class list generation**: Automatically generates a class list file or uses an existing one
- **Random shuffling**: Option to shuffle the dataset during packing for better training dynamics

If no resize is specified, files are archived as-is. If resize is defined, all files are re-encoded to the specified format (WebP by default).

When a class file is provided, only images from known classes will be packed.

## Output

The tool creates a new directory with the name of the first input directory plus the specified suffix. This directory will contain:

- For WebDataset format:
    - WebDataset shards (.tar files)
    - A classes.txt file listing all classes in the dataset
- For Directory format:
    - Subdirectories for each class
    - A classes.txt file listing all classes in the dataset

For more detailed information about each option and its usage, refer to the help output of the tool.
