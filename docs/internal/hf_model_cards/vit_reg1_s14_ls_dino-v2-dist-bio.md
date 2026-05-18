---
tags:
- image-feature-extraction
- birder
- pytorch
- biology
library_name: birder
license: apache-2.0
base_model:
- birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio
---

# Model Card for vit_reg1_s14_ls_dino-v2-dist-bio

`vit_reg1_s14_ls_dino-v2-dist-bio` is a compact Bio-DINO image encoder distilled from the larger [Bio-DINO SoViT-150M/14 model](https://huggingface.co/birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio).
It keeps the same natural-photography biodiversity scope as the teacher model, but uses a much smaller ViT-S/14-style student with 21.7M backbone parameters and 384-dimensional embeddings.

For the training-data description, intended scope, limitations, broader evaluation design and background on the Bio-DINO release, see the [teacher model card](https://huggingface.co/birder-project/vit_reg4_so150m_p14_ls_dino-v2-bio).
This card focuses only on what is specific to the distilled model.

## Model Details

- **Model Type:** Image encoder and detection backbone
- **Model Stats:**
    - Params (M): 21.7
    - Input image size: 252 x 252
- **Dataset:** Trained on a diverse dataset of approximately 31M images, including:
    - TreeOfLife-10M-EOL-NaturalImages
    - iNaturalist 2021
    - BIOSCAN-5M (pretrain split)
    - TreeOfLife-200M (subset)
    - IP102 v1.1
    - iWildCam 2022 (subset)
    - The Birder dataset (private dataset)

- **Papers:**
    - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: <https://arxiv.org/abs/2010.11929>
    - Vision Transformers Need Registers: <https://arxiv.org/abs/2309.16588>
    - DINOv2: Learning Robust Visual Features without Supervision: <https://arxiv.org/abs/2304.07193>

## Distillation

The student was distilled on the same dataset used for Bio-DINO.

The distillation run used the Birder `train_dino_v2_dist` workflow with the Bio-DINO 252px teacher checkpoint:

The launch/configuration file is available here: [distillation Slurm file](train_distributed_dino-v2_bio-dist.sh).

## Selected Results

The iNaturalist 2021 result uses a fuller linear-probing setup than the frozen-embedding evaluations above: the backbone is frozen, but the classification head is trained for multiple epochs with light image augmentation.

| Model                         | Resolution | Params (M) | Accuracy | Top-3 accuracy | Macro F1 |
| ----------------------------- | ---------: | ---------: | -------: | -------------: | -------: |
| Bio-DINO 224px                |        224 |      142.6 |   0.8572 |         0.9420 |   0.8564 |
| Bio-DINO 252px                |        252 |      142.6 |   0.8709 |         0.9510 |   0.8702 |
| Bio-DINO 336px                |        336 |      142.6 |   0.8807 |         0.9567 |   0.8800 |
| Bio-DINO S/14 (dist)          |        252 |       25.5 |   0.8010 |         0.9055 |   0.8000 |
| BioCLIP v1 ViT-B/16           |        224 |       93.5 |   0.7890 |         0.9005 |   0.7874 |
| BioCLIP v2 ViT-L/14           |        224 |      313.4 |   0.9169 |         0.9745 |   0.9164 |
| BioTrove-CLIP-O ViT-B/16      |        224 |       93.5 |   0.8351 |         0.9290 |   0.8334 |
| DINOv2 ViT-B/14               |        224 |       93.4 |   0.7492 |         0.8645 |   0.7457 |
| DINOv3 RoPE ViT-B/16          |        256 |       93.4 |   0.7826 |         0.8889 |   0.7795 |
| DINOv3 RoPE ViT-L/16          |        256 |      313.4 |   0.8333 |         0.9235 |   0.8309 |
| PE-Core RoPE ViT-B/16         |        224 |      100.6 |   0.6831 |         0.8299 |   0.6788 |
| ConvNeXt v1 L ImageNet-22K    |        224 |      211.6 |   0.7317 |         0.8647 |   0.7289 |

For the iNaturalist 2021 linear-probe table, parameter counts include the frozen image encoder plus the trained 10k-class linear classification head.

![iNaturalist 2021 accuracy vs CPU latency](img/performance-inat21.png)

## Model Usage

### Image Embeddings

```python
import birder
from birder.inference.classification import infer_image

# Option 1: manual setup (more control over preprocessing)
net, model_info = birder.load_pretrained_model("vit_reg1_s14_ls_dino-v2-dist-bio", inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

# Option 2: helper (quick start with default preprocessing)
net, model_info, transform = birder.load_pretrained_model_and_transform("vit_reg1_s14_ls_dino-v2-dist-bio", inference=True)

image = "path/to/image.jpeg"  # or a PIL image
out, embedding = infer_image(net, image, transform, return_embedding=True)
# embedding is a NumPy array with shape of (1, 384)
```

### Detection Feature Map

```python
from PIL import Image
import birder

net, model_info, transform = birder.load_pretrained_model_and_transform("vit_reg1_s14_ls_dino-v2-dist-bio", inference=True)

image = Image.open("path/to/image.jpeg")
features = net.detection_features(transform(image).unsqueeze(0))
# features is a dict (stage name -> torch.Tensor)
print([(k, v.size()) for k, v in features.items()])
# Output example:
# [('stage1', torch.Size([1, 384, 18, 18]))]
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
