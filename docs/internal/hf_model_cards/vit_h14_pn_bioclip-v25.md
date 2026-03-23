---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: mit
base_model:
- imageomics/bioclip-2.5-vith14
datasets:
- imageomics/TreeOfLife-200M
---

# Model Card for vit_h14_pn_bioclip-v25

A ViT h14 image encoder from BioCLIP-2 by Gu et al., converted to the Birder format for image feature extraction.
This version preserves the original model weights and architecture.
Trained on the large-scale TreeOfLife-200M dataset, it serves as a powerful foundation for downstream computer vision tasks.
The model excels at understanding biological imagery across diverse taxonomic groups.

See: <https://huggingface.co/imageomics/bioclip-2.5-vith14> for further details.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 632.1
    - Input image size: 224 x 224
- **Dataset:** Trained on the TreeOfLife-200M dataset

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning: <https://arxiv.org/abs/2505.23883>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
(net, model_info) = birder.load_pretrained_model("vit_h14_pn_bioclip-v25", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_h14_pn_bioclip-v25", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 1280)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info, transform) = birder.load_pretrained_model_and_transform("vit_h14_pn_bioclip-v25", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 1280, 16, 16]))]
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

@misc{gu2025bioclip2emergentproperties,
      title={BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
      author={Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
      year={2025},
      eprint={2505.23883},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.23883},
}

@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title={OpenCLIP},
  year={2021},
  doi={10.5281/zenodo.5143773},
}
```
