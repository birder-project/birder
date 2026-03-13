---
license: cc-by-4.0
task_categories:
- image-classification
pretty_name: Butterflies & Moths Austria
size_categories:
- 100K<n<1M
---

# Butterflies & Moths Austria

## Dataset Summary

This is a repackaged version of the Austria butterflies and moths dataset in PyTorch `ImageFolder` format.

**Note**: All credit goes to the original authors.

Compared to the original release, this upload:

- Converts all images to WebP
- Resizes each image so that the total number of pixels < 589,824 (=768×768), preserving aspect ratio
- Pre-splits the dataset into train/validation/test using a 70:20:10 split
- Packs the data in an `ImageFolder` directory layout

The original dataset contains 541,677 images of 185 butterfly and moth species from Austria.
Classes with fewer than 10 total images (6 classes) are placed in a separate `misc` split,
the remaining 179 classes are split into `train`, `validation`, and `test`.

- Dataset: <https://figshare.com/s/e79493adf7d26352f0c7>
- Paper: <https://www.nature.com/articles/s41597-025-05708-z>

## Citation

```bibtex
@article{Barkmann2025,
         title={Machine learning training data: over 500,000 images of butterflies and moths (Lepidoptera) with species labels},
         author={Barkmann, Friederike and Lindner, Andreas and W{\"u}rflinger, Ronald and H{\"o}ttinger, Helmut and R{\"u}disser, Johannes},
         journal={Scientific Data},
         volume={12},
         number={1},
         pages={1369},
         year={2025},
         doi={10.1038/s41597-025-05708-z},
         url={https://doi.org/10.1038/s41597-025-05708-z},
}
```
