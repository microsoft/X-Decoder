# X-Decoder Suite for Segmentation In the Wild
Organizers: 

## :fire: News
* **[2023.03.01]** The [Segmentation in the Wild Challenge](https://eval.ai/web/challenges/challenge-page/1931/overview) had been launched and ready for submitting results!
* **[2023.02.28]** We release Segmentation In the Wild dataset and evaluation code.

## :notes: Introduction
![seginw_allfig](https://user-images.githubusercontent.com/11957155/221871274-a46da377-5c25-4642-80ef-edf150d31418.png)

**The branch includes:** 

* **Dataset** The download link for SGinW dataset;
* **Evaluation** On the fly evaluation code supported X-Decoder evaluation.

## Getting Started

### Installation
```sh
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
```

### Download
Please download the dataset [HERE](https://projects4jw.blob.core.windows.net/x-decoder/release/seginw.zip)!

```
.
└── seginw/
    ├── Airplane-Parts/
    │   ├── train/
    │   │   ├── *.jpg
    │   │   └── _annotations_min1cat.coco.json
    │   ├── train_10shot/
    │   │   └── ...
    │   └── valid/
    │       └── ...
    ├── Bottles/
    │   └── ...
    └── ...
```

## Evaluation

* Evaluate under Framework X-Decoder
```sh
mpirun -n 8 python eval.py evaluate --conf_files configs/xdecoder/svlp_focalt_lang.yaml  --overrides WEIGHT /pth/to/ckpt
```
Note: Due to zero-padding, filling a single gpu with multiple images may decrease the performance.

* Evaluate Using Json File



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


## Dataset Statistics
<img width="1240" alt="Screenshot 2023-02-28 at 8 46 56 AM" src="https://user-images.githubusercontent.com/11957155/221888498-f0e332ae-516f-405b-b3ee-faea5db5dc57.png">

## Submission Format
Please refer to detailed format [HERE](https://github.com/microsoft/X-Decoder/blob/seginw/eval_with_json/submission.zip) : )

```
.
├── ade.json/
│   └── {"ADE150-mIoU": "x", "ADE150-PQ": "x", "ADE150-mAP": "x", "ADE847-mIoU": "x"} 
├── seginw_Airplane-Parts_val.json/
│   └── coco format
├── seginw_Bottles_val.json
├── seginw_Brain-Tumor_val.json
├── seginw_Chicken_val.json
└── ...
```


## Citation
```
@article{zou2022xdecoder,
  author      = {Zou, Xueyan and Dou, Zi-Yi and Yang, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee, Yong Jae and Gao, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
```
