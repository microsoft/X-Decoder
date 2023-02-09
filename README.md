# X-Decoder: Generalized Decoding for Pixel, Image, and Language
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-decoding-for-pixel-image-and/referring-expression-segmentation-on-refcocog)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refcocog?p=generalized-decoding-for-pixel-image-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-decoding-for-pixel-image-and/semantic-segmentation-on-coco-1)](https://paperswithcode.com/sota/semantic-segmentation-on-coco-1?p=generalized-decoding-for-pixel-image-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-decoding-for-pixel-image-and/panoptic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-ade20k-val?p=generalized-decoding-for-pixel-image-and) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/generalized-decoding-for-pixel-image-and/instance-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/instance-segmentation-on-ade20k-val?p=generalized-decoding-for-pixel-image-and)

\[[Project Page](https://x-decoder-vl.github.io/)\]   \[[Paper](https://arxiv.org/pdf/2212.11270.pdf)\]    \[[HuggingFace All-in-One Demo](https://huggingface.co/spaces/xdecoder/Demo)\] \[[HuggingFace Instruct Demo](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder)\]  \[[Video](https://youtu.be/nZZTkYM0kd0)\]

by [Xueyan Zou*](https://maureenzou.github.io/), [Zi-Yi Dou*](https://zdou0830.github.io/), [Jianwei Yang*](https://jwyang.github.io/),  [Zhe Gan](https://zhegan27.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Chunyuan Li](https://chunyuan.li/), [Xiyang Dai](https://sites.google.com/site/xiyangdai/), [Harkirat Behl](https://harkiratbehl.github.io/), [Jianfeng Wang](https://scholar.google.com/citations?user=vJWEw_8AAAAJ&hl=en), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en), [Nanyun Peng](https://vnpeng.net/), [Lijuan Wang](https://scholar.google.com/citations?user=cDcWXuIAAAAJ&hl=zh-CN), [Yong Jae Lee^](https://pages.cs.wisc.edu/~yongjaelee/), [Jianfeng Gao^](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F).

## :fire: News
* **[2022.02.07]** We combine <ins>X-Decoder</ins> (strong image understanding), <ins>GPT-3</ins> (strong language understanding) and <ins>Stable Diffusion</ins> (strong image generation) to make an [instructional image editing demo](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder), check it out!
* **[2022.12.21]** We release inference code of X-Decoder.
* **[2022.12.21]** We release Focal-T pretrained checkpoint.
* **[2022.12.21]** We release open-vocabulary segmentation benchmark.

## :notes: Introduction

![github_figure](https://user-images.githubusercontent.com/11957155/210801832-c9143c42-ef65-4501-95a5-0d54749dcc52.gif)

X-Decoder is a generalized decoding model that can generate **pixel-level segmentation** and **token-level texts** seamlessly!

**It achieves:**

* State-of-the-art results on open-vocabulary segmentation and referring segmentation on eight datasets; 
* Better or competitive finetuned performance to generalist and specialist models on segmentation and VL tasks; 
* Friendly for efficient finetuning and flexible for novel task composition.

**It supports:** 

* **One suite of parameters** pretrained for Semantic/Instance/Panoptic Segmentation, Referring Segmentation, Image Captioning, and Image-Text Retrieval;
* **One model architecture** finetuned for Semantic/Instance/Panoptic Segmentation, Referring Segmentation, Image Captioning, Image-Text Retrieval and Visual Question Answering (with an extra cls head);
* **Zero-shot task composition** for Region Retrieval, Referring Captioning, Image Editing.

## :snowflake: TODO
- [ ] Release Training and Prompt Tuning code
- [ ] Release Finetuned model
- [ ] Release Base and Large model

## Getting Started

### Installation
```sh
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
```

To prepare the dataset: [DATASET.md](./DATASET.md)

## Open Vocabulary Segmentation
```sh
mpirun -n 8 python eval.py evaluate --conf_files configs/xdecoder/svlp_focalt_lang.yaml  --overrides WEIGHT /pth/to/ckpt
```
Note: Due to zero-padding, filling a single gpu with multiple images may decrease the performance.

## Inference Demo
```sh
# For Segmentation Tasks
python demo/demo_semseg.py evaluate --conf_files configs/xdecoder/svlp_focalt_lang.yaml  --overrides WEIGHT /pth/to/xdecoder_focalt_best_openseg.pt
# For VL Tasks
python demo/demo_captioning.py evaluate --conf_files configs/xdecoder/svlp_focalt_lang.yaml  --overrides WEIGHT /pth/to/xdecoder_focalt_last_novg.pt
```


## Model Zoo
|           |         | ADE  |      |      | ADE-full | SUN  | SCAN |      | SCAN40 | Cityscape |      |      | BDD  |      |
|-----------|---------|------|------|------|----------|------|------|------|--------|-----------|------|------|------|------|
| model     | ckpt    | PQ   | AP   | mIoU | mIoU     | mIoU | PQ   | mIoU | mIoU   | PQ        | mAP  | mIoU | PQ   | mIoU |
| X-Decoder | [BestSeg Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_best_openseg.pt) | 19.1 | 10.1 | 25.1 | 6.2      | 35.7 | 30.3 | 38.4 | 22.4   | 37.7      | 18.5 | 50.2 | 16.9 | 47.6 |
<!---
| X-Decoder | [Last Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last.pt) |  |  |  |       |  |  |  |    |       |  |  |  |  |
| X-Decoder | [NoVG Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last_novg.pt) |  |  |  |       |  |  |  |    |       |  |  |  | |
-->

* X-Decoder [NoVG Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last_novg.pt)
* X-Decoder [Last Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last.pt)

## Additional Results

* Finetuned ADE 150 (32 epochs)

| Model                           | Task    | Log | PQ   | mAP  | mIoU |
|---------------------------------|---------|-----|------|------|------|
| X-Decoder (davit-d5,Deformable) | PanoSeg |  [log](https://projects4jw.blob.core.windows.net/x-decoder/release/ade20k_finetune_davitd5_deform_32epoch_log.txt)   | 52.4 | 38.7 | 59.1 |

## Acknowledgement
* We appreciate the contructive dicussion with [Haotian Zhang](https://haotian-zhang.github.io/), and inspiration from [GLIP](https://github.com/microsoft/GLIP)! 
* We thank the solid codebase of [Mask2Former](https://github.com/facebookresearch/Mask2Former)
* It is really generous that HuggingFace :hugs: sponsors GPU for our Demo!

## Citation
```
@article{zou2022xdecoder,
  author      = {Zou, Xueyan and Dou, Zi-Yi and Yang, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee, Yong Jae and Gao, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
```
