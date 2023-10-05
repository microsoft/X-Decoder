# X-Decoder: Generalized Decoding for Pixel, Image, and Language

\[[Project Page](https://x-decoder-vl.github.io/)\]   \[[Paper](https://arxiv.org/pdf/2212.11270.pdf)\]    \[[HuggingFace All-in-One Demo](https://huggingface.co/spaces/xdecoder/Demo)\] \[[HuggingFace Instruct Demo](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder)\]  \[[Video](https://youtu.be/nZZTkYM0kd0)\]

by [Xueyan Zou*](https://maureenzou.github.io/), [Zi-Yi Dou*](https://zdou0830.github.io/), [Jianwei Yang*](https://jwyang.github.io/),  [Zhe Gan](https://zhegan27.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Chunyuan Li](https://chunyuan.li/), [Xiyang Dai](https://sites.google.com/site/xiyangdai/), [Harkirat Behl](https://harkiratbehl.github.io/), [Jianfeng Wang](https://scholar.google.com/citations?user=vJWEw_8AAAAJ&hl=en), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en), [Nanyun Peng](https://vnpeng.net/), [Lijuan Wang](https://scholar.google.com/citations?user=cDcWXuIAAAAJ&hl=zh-CN), [Yong Jae Lee^](https://pages.cs.wisc.edu/~yongjaelee/), [Jianfeng Gao^](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F) in **CVPR 2023**.


## :hot_pepper: Getting Started
We release the following contents for **both SEEM and X-Decoder**:exclamation:
- [x] Demo Code
- [x] Model Checkpoint
- [x] Comprehensive User Guide
- [x] Training Code
- [x] Evaluation Code

:point_right: **One-Line SEEM Demo with Linux:**
```sh
git clone git@github.com:UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git && sh aasets/scripts/run_demo.sh
```

:round_pushpin: *[New]* **Getting Started:**

* [INSTALL.md](assets/readmes/INSTALL.md) <br>
* [DATASET.md](assets/readmes/DATASET.md) <br>
* [TRAIN.md](assets/readmes/TRAIN.md) <br>
* [EVAL.md](assets/readmes/EVAL.md) <br>
* [INFERENCE.md](assets/readmes/INFERENCE.md) <br>

:round_pushpin: *[New]* **Latest Checkpoints and Numbers:**
|                 |                                                                                      |          | COCO |      |      | Ref-COCOg |      |      | VOC   |       | SBD   |       |
|-----------------|---------------------------------------------------------------------------------------------|----------|------|------|------|-----------|------|------|-------|-------|-------|-------|
| Method          |  Checkpoint                                                                                  | backbone | PQ &uarr;  | mAP &uarr; | mIoU &uarr; | cIoU  &uarr; | mIoU &uarr; | AP50 &uarr; | NoC85 &darr; | NoC90 &darr;| NoC85 &darr;| NoC90 &darr;|
| X-Decoder       |  [ckpt](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt) | Focal-T  | 50.8 | 39.5 | 62.4 | 57.6      | 63.2 | 71.6 | -     | -     | -     | -     |
| X-Decoder-oq201 |  [ckpt](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focall_last.pt) | Focal-L  | 56.5 | 46.7 | 67.2 | 62.8      | 67.5 | 76.3 | -     | -     | -     | -     |
| SEEM_v0            | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt)      | Focal-T  | 50.6 | 39.4 | 60.9 | 58.5      | 63.5 | 71.6 | 3.54  | 4.59  | *     | *     |
| SEEM_v0            |  -                                                                                           | Davit-d3 | 56.2 | 46.8 | 65.3 | 63.2      | 68.3 | 76.6 | 2.99  | 3.89  | 5.93  | 9.23  |
| SEEM_v0      | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt)       | Focal-L  | 56.2 | 46.4 | 65.5 | 62.8      | 67.7 | 76.2 | 3.04  | 3.85  | *     | *     |
| SEEM_v1      | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v1.pt)       | Focal-T  | 50.8 | 39.4 | 60.7 |   58.5    |  63.7 | 72.0 | 3.19  | 4.13  | *     | *     |
| SEEM_v1      | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitb_v1.pt)       | SAM-ViT-B  | 52.0 | 43.5 | 60.2 | 54.1      | 62.2 | 69.3 | 2.53  | 3.23  | *     | *     |
| SEEM_v1       | [ckpt](https://huggingface.co/xdecoder/SEEM/resolve/main/seem_samvitl_v1.pt)       | SAM-ViT-L  | 49.0 | 41.6 | 58.2 | 53.8      | 62.2 | 69.5 | 2.40  | 2.96  | *     | *     |

**SEEM_v0:** Supporting Single Interactive object training and inference <br>
**SEEM_v1:** Supporting Multiple Interactive objects training and inference


## :fire: News
* **[2023.10.04]** We are excited to release :white_check_mark: [training/evaluation/demo code](https://github.com/microsoft/X-Decoder/edit/v2.0/README.md#hot_pepper-getting-started), :white_check_mark: [new checkpoints](https://github.com/microsoft/X-Decoder/edit/v2.0/README.md#hot_pepper-getting-started), and :white_check_mark: [comprehensive readmes](https://github.com/microsoft/X-Decoder/edit/v2.0/README.md#hot_pepper-getting-started) for ***both X-Decoder and SEEM***!
* **[2023.09.24]** We are providing new demo command/code for inference ([DEMO.md](asset/DEMO.md))!
* **[2023.07.19]** :roller_coaster: We are excited to release the x-decoder training code ([INSTALL.md](asset/INSTALL.md), [DATASET.md](asset/DATASET.md), [TRAIN.md](asset/TRAIN.md), [EVALUATION.md](asset/EVALUATION.md))!
* **[2023.07.10]** We release [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM), a universal image segmentation model to enable segment and recognize anything at any desired granularity. Code and checkpoint are available!
* **[2023.04.14]** We are releasing [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once), a new universal interactive interface for image segmentation! You can use it for any segmentation tasks, way beyond what X-Decoder can do!

<p align="center">
  <img src="inference/images/teaser_new.png" width="90%" height="90%">
</p>

* **[2023.03.20]** As an aspiration of our X-Decoder, we developed OpenSeeD ([[Paper](https://arxiv.org/pdf/2303.08131.pdf)][[Code](https://github.com/IDEA-Research/OpenSeeD)]) to enable open-vocabulary segmentation and detection with a single model, Check it out! 
* **[2023.03.14]** We release [X-GPT](https://github.com/microsoft/X-Decoder/tree/xgpt) which is an conversational version of our X-Decoder through GPT-3 langchain!
* **[2023.03.01]** The [Segmentation in the Wild Challenge](https://eval.ai/web/challenges/challenge-page/1931/overview) had been launched and ready for submitting results!
* **[2023.02.28]** We released the [SGinW benchmark](https://github.com/microsoft/X-Decoder/tree/seginw) for our challenge. Welcome to build your own models on the benchmark!
* **[2023.02.27]** Our X-Decoder has been accepted by CVPR 2023!
* **[2023.02.07]** We combine <ins>X-Decoder</ins> (strong image understanding), <ins>GPT-3</ins> (strong language understanding) and <ins>Stable Diffusion</ins> (strong image generation) to make an [instructional image editing demo](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder), check it out!
* **[2022.12.21]** We release inference code of X-Decoder.
* **[2022.12.21]** We release Focal-T pretrained checkpoint.
* **[2022.12.21]** We release open-vocabulary segmentation benchmark.

## :paintbrush: DEMO
:blueberries: [[X-GPT]](https://github.com/microsoft/X-Decoder/tree/xgpt) &ensp; :strawberry:[[Instruct X-Decoder](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder)]

![demo](https://user-images.githubusercontent.com/11957155/225728214-0523bd30-31f7-472d-be7e-12a049c25cbd.gif)

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

<!-- ## Getting Started

### Installation
```sh
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
sh install_cococapeval.sh
export DATASET=/pth/to/dataset
```

Here is the new link to download [coco_caption.zip](https://drive.google.com/file/d/1FHEQNkW7zHvSd-R8CQPC1gIuigC9w8Ff/view?usp=sharing).

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
| X-Decoder | [BestSeg Tiny](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_best_openseg.pt) | 19.1 | 10.1 | 25.1 | 6.2      | 35.7 | 30.3 | 38.4 | 22.4   | 37.7      | 18.5 | 50.2 | 16.9 | 47.6 |
<!---
| X-Decoder | [Last Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last.pt) |  |  |  |       |  |  |  |    |       |  |  |  |  |
| X-Decoder | [NoVG Tiny](https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last_novg.pt) |  |  |  |       |  |  |  |    |       |  |  |  | |
-->

<!-- * X-Decoder [NoVG Tiny](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last_novg.pt)
* X-Decoder [Last Tiny](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt)

## Additional Results
* Finetuned ADE 150 (32 epochs)

| Model                           | Task    | Log | PQ   | mAP  | mIoU |
|---------------------------------|---------|-----|------|------|------|
| X-Decoder (davit-d5,Deformable) | PanoSeg |  [log](https://projects4jw.blob.core.windows.net/x-decoder/release/ade20k_finetune_davitd5_deform_32epoch_log.txt)   | 52.4 | 38.7 | 59.1 | -->

## Acknowledgement
* We appreciate the contructive dicussion with [Haotian Zhang](https://haotian-zhang.github.io/) 
* We build our work on top of [Mask2Former](https://github.com/facebookresearch/Mask2Former)
* We build our demos on [HuggingFace :hugs:](https://huggingface.co/) with sponsored GPUs
* We appreciate the discussion with Xiaoyu Xiang during rebuttal

## Citation
```
@article{zou2022xdecoder,
  author      = {Zou*, Xueyan and Dou*, Zi-Yi and Yang*, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee*, Yong Jae and Gao*, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
```
