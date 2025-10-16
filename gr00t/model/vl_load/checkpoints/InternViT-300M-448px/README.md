---
license: mit
datasets:
  - laion/laion2B-en
  - laion/laion-coco
  - laion/laion2B-multi
  - kakaobrain/coyo-700m
  - conceptual_captions
  - wanng/wukong100m
pipeline_tag: image-feature-extraction
library_name: transformers
new_version: OpenGVLab/InternViT-300M-448px-V2_5
---

# InternViT-300M-448px

[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVL)  [\[📜 InternVL 1.0\]](https://huggingface.co/papers/2312.14238)  [\[📜 InternVL 1.5\]](https://huggingface.co/papers/2404.16821)  [\[📜 Mini-InternVL\]](https://arxiv.org/abs/2410.16261)  [\[📜 InternVL 2.5\]](https://huggingface.co/papers/2412.05271)

[\[🆕 Blog\]](https://internvl.github.io/blog/)  [\[🗨️ Chat Demo\]](https://internvl.opengvlab.com/)  [\[🤗 HF Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL)  [\[🚀 Quick Start\]](#quick-start)  [\[📖 Documents\]](https://internvl.readthedocs.io/en/latest/)

<div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
</div>

This update primarily focuses on enhancing the efficiency of the vision foundation model. We developed InternViT-300M-448px by distilling knowledge from the robust vision foundation model, [InternViT-6B-448px-V1-5](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5). Like its predecessor, InternViT-300M-448px features a dynamic input resolution of 448×448, with a basic tile size of 448×448. During training, it allows for 1 to 12 tiles, and expands to 1 to 40 tiles during testing. Additionally, it inherits the powerful robustness, OCR capability, and high-resolution processing capacity from InternViT-6B-448px-V1-5.

## Model Details

- **Model Type:** vision foundation model, feature backbone
- **Model Stats:**
  - Params (M): 304
  - Image size: 448 x 448, training with 1 - 12 tiles
- **Pretrain Dataset:** LAION-en, LAION-zh, COYO, GRIT, COCO, TextCaps, Objects365, OpenImages, All-Seeing, Wukong-OCR, LaionCOCO-OCR, and other OCR-related datasets.
  To enhance the OCR capability of the model, we have incorporated additional OCR data alongside the general caption datasets. Specifically, we utilized PaddleOCR to perform Chinese OCR on images from Wukong and English OCR on images from LAION-COCO.

## Quick Start

> \[!Warning\]
> 🚨 Note: In our experience, the InternViT V2.5 series is better suited for building MLLMs than traditional computer vision tasks.

```python
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-300M-448px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image = Image.open('./examples/image1.jpg').convert('RGB')

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
```

## License

This project is released under the MIT License.

## Citation

If you find this project useful in your research, please consider citing:

```BibTeX
@article{chen2024expanding,
  title={Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling},
  author={Chen, Zhe and Wang, Weiyun and Cao, Yue and Liu, Yangzhou and Gao, Zhangwei and Cui, Erfei and Zhu, Jinguo and Ye, Shenglong and Tian, Hao and Liu, Zhaoyang and others},
  journal={arXiv preprint arXiv:2412.05271},
  year={2024}
}
@article{gao2024mini,
  title={Mini-internvl: A flexible-transfer pocket multimodal model with 5\% parameters and 90\% performance},
  author={Gao, Zhangwei and Chen, Zhe and Cui, Erfei and Ren, Yiming and Wang, Weiyun and Zhu, Jinguo and Tian, Hao and Ye, Shenglong and He, Junjun and Zhu, Xizhou and others},
  journal={arXiv preprint arXiv:2410.16261},
  year={2024}
}
@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
@inproceedings{chen2024internvl,
  title={Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24185--24198},
  year={2024}
}
```