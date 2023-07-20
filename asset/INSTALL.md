### Installation

```sh
pip3 install torch torchvision
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/openai/CLIP.git
python -m pip install -r requirements.txt
sh install_cococapeval.sh
export DATASET=/pth/to/dataset
wget https://huggingface.co/xdecoder/X-Decoder/blob/main/coco_caption.zip
export PATH=$PATH:/pth/to/coco_caption/jre1.8.0_321/bin
export PYTHONPATH=$PYTHONPATH:/pth/to/coco_caption
```