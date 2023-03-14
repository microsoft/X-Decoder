# X-Chat: Multi-modal Interactive X-Decoder

## :fire: News

* **[2023.03.14]** We build X-Chat, a multi-modal interactive X-Decoder!

## :notes: Introduction

## Getting Started

### Installation
```sh
# set up environment
conda create -n xchat python=3.8
conda activate xchat

# install dependencies
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
sh install_cococapeval.sh

# download x-decoder tiny model
wget https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last_novg.pt

# create a folder for image uploading and image retrieval
mkdir image & mkdir image_pool
```

## Run Demo
```sh
# For Segmentation Tasks
python demo/demo_semseg.py evaluate --conf_files configs/xdecoder/svlp_focalt_lang.yaml  --overrides WEIGHT /pth/to/xdecoder_focalt_best_openseg.pt
# For VL Tasks
python demo/demo_captioning.py evaluate --conf_files configs/xdecoder/svlp_focalt_lang.yaml  --overrides WEIGHT /pth/to/xdecoder_focalt_last_novg.pt
```

## Acknowledgement
* We are highly inspired by [visual-chatgpt](https://github.com/microsoft/visual-chatgpt) in the usage of langchain, thanks for the great work!

## Citation
```
@article{zou2022xdecoder,
  author      = {Zou, Xueyan and Dou, Zi-Yi and Yang, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee, Yong Jae and Gao, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
```
