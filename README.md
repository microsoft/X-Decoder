# X-Decoder: Generalized Decoding for Pixel, Image, and Language

\[[Project Page](https://x-decoder-vl.github.io/)\]   \[[Paper](https://arxiv.org/pdf/2212.11270.pdf)\]    \[[HuggingFace All-in-One Demo](https://huggingface.co/spaces/xdecoder/Demo)\] \[[HuggingFace Instruct Demo](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder)\]  \[[Video](https://youtu.be/nZZTkYM0kd0)\]

by [Xueyan Zou*](https://maureenzou.github.io/), [Zi-Yi Dou*](https://zdou0830.github.io/), [Jianwei Yang*](https://jwyang.github.io/),  [Zhe Gan](https://zhegan27.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Chunyuan Li](https://chunyuan.li/), [Xiyang Dai](https://sites.google.com/site/xiyangdai/), [Harkirat Behl](https://harkiratbehl.github.io/), [Jianfeng Wang](https://scholar.google.com/citations?user=vJWEw_8AAAAJ&hl=en), [Lu Yuan](https://scholar.google.com/citations?user=k9TsUVsAAAAJ&hl=en), [Nanyun Peng](https://vnpeng.net/), [Lijuan Wang](https://scholar.google.com/citations?user=cDcWXuIAAAAJ&hl=zh-CN), [Yong Jae Lee^](https://pages.cs.wisc.edu/~yongjaelee/), [Jianfeng Gao^](https://www.microsoft.com/en-us/research/people/jfgao/?from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fpeople%2Fjfgao%2F) in CVPR 2023.


## :hot_pepper: Getting Started

<!-- :point_right: *[New]* **One-Line Getting Started:**
```sh
sh asset/train.sh # training
sh aaset/eval.sh # evaluation
``` -->

:point_right: *[New]* **Latest Checkpoints and Numbers:**
|          |            |     | COCO |      |      | ADE |     |      | Ref-COCO | COCO-Karpathy |      |       |
|----------|------------|-----|------|------|------|-----|-----|------|----------|---------------|------|-------|
| Backbone | Checkpoint | Log | PQ   | mAP  | mIoU | PQ  | mAP | mIoU | mIoU     | ir@1          | tr@1 | CIDEr |
| Focal-T  |  [last](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_last.pt)  |  Running   | 50.8 | 39.5 | 62.4 |     |  9.6   |  23.9    | 63.2   |   30.0  |   48.3   |   83.3    |
| Focal-T  |  [best_seg](https://huggingface.co/xdecoder/X-Decoder/resolve/main/xdecoder_focalt_best_openseg.pt)  |  Log   |  48.8   |   37.0   |   60.2   |     |  10.1   |  29.1    |    61.6      |     30.2    |    48.36  |       |
| Focal-L  |  [last](https://huggingface.co/xdecoder/X-Decoder/blob/main/xdecoder_focall_last.pt) |  Log   |  56.2  |  46.4    |   65.5   |     |  11.5  |  23.6  |  67.7  |  34.9     |   54.4   |       |
| Focal-L  |  [best_seg](https://huggingface.co/xdecoder/X-Decoder/blob/main/xdecoder_focall_bestseg.pt) |  Log   | 51.5   |   41.3   |   64.1   |     |  11.7   |  29.4    |  61.5  |  30.7  |  50.1  |       |

Note the number in Table 1 in main paper is after task specific finetuning.

:point_right: *[New]* **Installation, Training, Evaluation, Dataset, and Demo Guide**
* [DATASET.md](asset/DATASET.md)
* [INSTALL.md](asset/INSTALL.md)
* [TRAIN.md](asset/TRAIN.md)
* [EVALUATION.md](asset/EVALUATION.md)
* [DEMO.md](asset/DEMO.md)

## :fire: News

* **[2023.07.19]** :roller_coaster: We are excited to release the x-decoder training code ([INSTALL.md](asset/INSTALL.md), [DATASET.md](asset/DATASET.md), [TRAIN.md](asset/TRAIN.md), [EVALUATION.md](asset/EVALUATION.md))!
* **[2023.07.10]** We release [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM), a universal image segmentation model to enable segment and recognize anything at any desired granularity. Code and checkpoint are available!
* **[2023.04.14]** We are releasing [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once), a new universal interactive interface for image segmentation! You can use it for any segmentation tasks, way beyond what X-Decoder can do!

<p align="center">
  <img src="inference_demo/images/teaser_new.png" width="90%" height="90%">
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
