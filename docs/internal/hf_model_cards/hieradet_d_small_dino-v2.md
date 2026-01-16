---
tags:
- image-feature-extraction
- birder
- pytorch
library_name: birder
license: apache-2.0
---

# Model Card for hieradet_d_small_dino-v2

HieraDet (dynamic window size) small image encoder pre-trained using DINOv2. This model has *not* been fine-tuned for a specific classification task and is intended to be used as a general-purpose feature extractor or a backbone for downstream tasks like object detection, segmentation, or custom classification.

## Model Details

- **Model Type:** Image classification and detection backbone
- **Model Stats:**
    - Params (M): 33.9
    - Input image size: 224 x 224
- **Dataset:** Trained on a diverse dataset of approximately 21M images, including:
    - iNaturalist 2021 (~2.7M)
    - imagenet-w21-webp-wds (~2.4M random subset)
    - Objects365-2020 (~1.8M)
    - WebVision-2.0 (~1.5M random subset)
    - imagenet-1k-webp (~1.3M)
    - BIOSCAN-1M (~1.1M)
    - GLDv2 (~820K random subset of 100 chunks)
    - VMMR (~285K)
    - SA-1B (~220K random subset of 20 chunks)
    - TreeOfLife-10M (~200K random subset)
    - Places365 (~170K random subset)
    - Open Images V4 (~160K random subset)
    - COCO (~120K) x2
    - SUN397 (~100K)
    - Food-101 (~100K)
    - IP102 v1.1 (~75K)
    - NABirds (~48K)
    - Birdsnap v1.1 (~44K)
    - Country211 (~31K)
    - ADE20K 2016 (~22K) x2
    - VOC2012 (~17K)
    - IndoorCVPR 2009 (~15K)
    - VisDrone2019-DET (~10K) x4
    - BLIP3o-Pretrain-Short-Caption (~10K random subset)
    - comma10k (~10K) x2
    - FGVC-Aircraft 2013 (~10K) x2
    - Oxford-IIIT Pet (~7K) x4
    - CUB-200 2011 (~6K)
    - Flowers102 (~1K) x16
    - The Birder dataset (~8M, private dataset)

- **Papers:**
    - Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles: <https://arxiv.org/abs/2306.00989>
    - SAM 2: Segment Anything in Images and Videos: <https://arxiv.org/abs/2408.00714>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

(net, model_info) = birder.load_pretrained_model("hieradet_d_small_dino-v2", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "path/to/image.jpeg"  # or a PIL image
(out, embedding) = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 768)
```

### Detection Feature Map

```python
from PIL import Image
import birder

(net, model_info) = birder.load_pretrained_model("hieradet_d_small_dino-v2", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 96, 56, 56])),
#  ('stage2', torch.Size([1, 192, 28, 28])),
#  ('stage3', torch.Size([1, 384, 14, 14])),
#  ('stage4', torch.Size([1, 768, 7, 7]))]
```

## Citation

```bibtex
@misc{ryali2023hierahierarchicalvisiontransformer,
      title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
      author={Chaitanya Ryali and Yuan-Ting Hu and Daniel Bolya and Chen Wei and Haoqi Fan and Po-Yao Huang and Vaibhav Aggarwal and Arkabandhu Chowdhury and Omid Poursaeed and Judy Hoffman and Jitendra Malik and Yanghao Li and Christoph Feichtenhofer},
      year={2023},
      eprint={2306.00989},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2306.00989},
}

@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos},
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714},
}

@misc{oquab2024dinov2learningrobustvisual,
      title={DINOv2: Learning Robust Visual Features without Supervision},
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2024},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.07193},
}
```
