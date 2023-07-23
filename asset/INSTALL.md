### Installation

```sh
pip3 install torch torchvision
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/openai/CLIP.git
python -m pip install -r requirements.txt
mkdir ../xdecoder_data
wget -P ../xdecoder_data https://huggingface.co/xdecoder/X-Decoder/resolve/main/coco_caption.zip
unzip ../xdecoder_data/coco_caption.zip -d ../xdecoder_data
export PYTHONPATH=$PYTHONPATH:../xdecoder_data/coco_caption
export DATASET=../xdecoder_data
export PATH=$PATH:../xdecoder_data/coco_caption/jre1.8.0_321/bin
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```
