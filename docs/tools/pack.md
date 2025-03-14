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
- **Class list handling**: Supports multiple approaches:
    - Auto-generation of class list from input directories
    - Using an existing class file
    - Reading classes from an existing pack when appending
- **Split management**: Support for creating and appending different dataset splits (e.g., training, validation)
- **Random shuffling**: Option to shuffle the dataset during packing for better training dynamics

If no resize is specified, files are archived as-is if the format matches, or converted to the specified format otherwise. When resize is defined, all images are resized and re-encoded to the specified format.

When a class file is provided, only images from known classes will be packed.

## WebDataset Features

- **Shard size control**: Limit the maximum size of individual tar files
- **Metadata tracking**: Automatically generates a `_info.json` file with information about shards and samples
- **Append capability**: Add new splits to existing WebDataset packs while maintaining class consistency

## Output

The tool creates a new directory with:

- For WebDataset format:
    - WebDataset shards (.tar files) with a naming pattern based on suffix and split
    - A classes.txt file listing all classes in the dataset
    - A _info.json file with metadata about the dataset
- For Directory format:
    - Subdirectories for each class
    - A classes.txt file listing all classes in the dataset

For detailed information about specific options and their usage, refer to the help output of the tool.
