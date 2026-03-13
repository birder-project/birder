---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: mit
base_model:
- BGLab/BioTrove-CLIP
datasets:
- BGLab/BioTrove-Train
---

# Model Card for vit_b16_pn_biotrove-clip-o

A ViT b16 image encoder from BioTrove by Yang et al., converted to the Birder format for image feature extraction.
This version preserves the original model weights and architecture.
Trained on the large-scale BioTrove dataset, it serves as a powerful foundation for downstream computer vision tasks.
The model excels at understanding biological imagery across diverse taxonomic groups.

See: <https://huggingface.co/BGLab/BioTrove-CLIP> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 86.2
    - Input image size: 224 x 224
- **Dataset:** Trained on the BioTrove dataset

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - BioTrove: A Large Curated Image Dataset Enabling AI for Biodiversity: <https://arxiv.org/abs/2406.17720>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
(net, model_info) = birder.load_pretrained_model("vit_b16_pn_biotrove-clip-o", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_b16_pn_biotrove-clip-o", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_b16_pn_biotrove-clip-o", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 768, 14, 14]))]
```

## Citation

```bibtex
@misc{dosovitskiy2021imageworth16x16words,
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
      year={2021},
      eprint={2010.11929},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2010.11929},
}

@misc{yang2025biotrovelargecuratedimage,
      title={BioTrove: A Large Curated Image Dataset Enabling AI for Biodiversity},
      author={Chih-Hsuan Yang and Benjamin Feuer and Zaki Jubery and Zi K. Deng and Andre Nakkab and Md Zahid Hasan and Shivani Chiranjeevi and Kelly Marshall and Nirmal Baishnab and Asheesh K Singh and Arti Singh and Soumik Sarkar and Nirav Merchant and Chinmay Hegde and Baskar Ganapathysubramanian},
      year={2025},
      eprint={2406.17720},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.17720},
}

@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title={OpenCLIP},
  year={2021},
  doi={10.5281/zenodo.5143773},
}
```
