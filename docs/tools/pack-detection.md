# Pack Detection

The `pack-detection` tool allows you to convert a COCO-style detection dataset into WebDataset format, which is optimized for efficient data loading during training. It reads images from a dataset root, reads annotations from a COCO JSON file, and writes image samples with packed detection metadata.

## Usage

```sh
python -m birder.tools pack-detection [OPTIONS]
```

To see all available options and get detailed help, run:

```sh
python -m birder.tools pack-detection --help
```

This will display a comprehensive list of all options and their descriptions, ensuring you have access to the full functionality of the `pack-detection` tool.

## Description

- **COCO-style input**: Reads images from `--data-path` and annotations from `--coco-json-path`
- **WebDataset output**: Writes `.tar` shards with ordered sample keys and split-aware `_info.json`
- **Multi-processing**: Utilizes multiple CPU cores for faster processing
- **Image resizing**: Option to resize images to a specified maximum dimension while maintaining aspect ratio
- **Image format conversion**: Ability to convert images to WebP, PNG, or JPEG formats when resizing
- **Split management**: Support for creating and appending different dataset splits (e.g., training, validation)
- **Empty-image filtering**: Option to skip images without annotations via `--drop-empty`

If no resize is specified, images are archived as-is when the format already matches, or converted to the requested format otherwise. When resize is defined, images are resized and re-encoded to the requested format.

## Packed Sample Format

Each WebDataset sample contains:

- An image payload under the selected image extension (`webp`, `png` or `jpeg`)
- A `json` payload with:
    - `image_id`
    - `file_name`
    - `width`, `height`
    - `boxes` in `xyxy` format
    - `labels` (1-indexed)
    - `area`
    - `iscrowd`
    - `annotation_ids`
    - `boxes_format`

By default, labels are remapped to contiguous 1-indexed values (reserving 0 for background). When `--class-file` is provided, the original COCO `category_id` values are preserved instead.

## Output

The tool creates a new directory with:

- WebDataset shards (.tar files) with a naming pattern based on suffix and split
- A classes.txt file listing all classes in the dataset
- A _info.json file with metadata about the dataset

For detailed information about specific options and their usage, refer to the help output of the tool.
