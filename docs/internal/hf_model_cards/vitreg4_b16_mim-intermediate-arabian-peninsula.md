---
tags:
- image-classification
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for vitreg4_b16_mim-intermediate-arabian-peninsula

A ViTReg4 image classification model. The model follows a three-stage training process: first, masked image modeling, next intermediate training on a large-scale dataset containing diverse bird species from around the world, finally fine-tuned specifically on the `arabian-peninsula` dataset.

The species list is derived from data available at <https://avibase.bsc-eoc.org/checklist.jsp?region=ARA>.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 86.7
    - Input image size: 384 x 384
- **Dataset:** arabian-peninsula (735 classes)
    - Intermediate training involved ~5000 species from asia, europe and eastern africa
    - Epoch 200 checkpoint of [vitreg4_b16_mim](https://huggingface.co/birder-project/vitreg4_b16_mim)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - Masked Autoencoders Are Scalable Vision Learners: <https://arxiv.org/abs/2111.06377>

## Model Usage

### Image Classification

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("vitreg4_b16_mim-intermediate-arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(signature)

# Create an inference transform
transform = birder.classification_transform(size, rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)
# out is a NumPy array with shape of (1, num_classes), representing class probabilities.
```

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, class_to_idx, signature, rgb_stats) = birder.load_pretrained_model("vitreg4_b16_mim-intermediate-arabian-peninsula", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(signature)

# Create an inference transform
transform = birder.classification_transform(size, rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, embedding_size)
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

@misc{darcet2024visiontransformersneedregisters,
      title={Vision Transformers Need Registers},
      author={Timothée Darcet and Maxime Oquab and Julien Mairal and Piotr Bojanowski},
      year={2024},
      eprint={2309.16588},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.16588},
}

@misc{he2021maskedautoencodersscalablevision,
      title={Masked Autoencoders Are Scalable Vision Learners},
      author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Dollár and Ross Girshick},
      year={2021},
      eprint={2111.06377},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2111.06377},
}
```
