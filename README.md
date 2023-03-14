# X-Chat: X-Decoder for a Chat

## :fire: News

* **[2023.03.14]** We build X-Chat, a multi-modal conversational X-Decoder through langchain!

## :notes: Introduction

### Why are we unique?

This specific project was started with our previous two demos, [all-in-one](https://huggingface.co/spaces/xdecoder/Demo) and [instruct x-decoder](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder). Then we were inspired by [visual-chatgpt](https://github.com/microsoft/visual-chatgpt) developed by our MSRA collegues to use the [langchain](https://github.com/hwchase17/langchain) to empower a conversational X-Decoder and encompassing all the capacities of our single X-Decoder model.

Our **X-Chat** has several unique and new features:

* **It uses a SINGLE X-Decoder model to support a wide range of vision and vision-language tasks. As such, you do not need separate models for individual tasks!**

* **It delivers the state-of-the-art segmentation performance. It is much better than CLIPSeg or other existing open-vocabulary segmentation systems!**

* **It also supports text-to-image retrieval. You can choose find a real image from your own pool or ask for generating a new image!**

In the next, we will:

* **Have visual question answering added to our pretrained X-Decoder model. Our model shows good VQA performance, but was not added to pretraining, though no barrier at all.**

* **You may notice we developed a more grounded instructPix2Pix with the support of our X-Decoder. Next we integrate it to our X-Chat!**

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

# export openai key, please follow https://langchain.readthedocs.io/en/latest/modules/llms/integrations/azure_openai_example.html
export OPENAI_API_TYPE=azure
export OPENAI_API_VERSION=2022-12-01
export OPENAI_API_BASE=https://your-resource-name.openai.azure.com
export OPENAI_API_KEY=<your Azure OpenAI API key>

```

## Run Demo
```sh
# Simply run this single line and enjoy it!!!
python xchat.py
```

## Acknowledgement

We are highly inspired by [visual-chatgpt](https://github.com/microsoft/visual-chatgpt) in the usage of langchain, and build on top of many great open-sourced projects: [HuggingFace Transformers](https://github.com/huggingface), [LangChain](https://github.com/hwchase17/langchain), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix).

## Citation
```
@article{zou2022xdecoder,
  author      = {Zou, Xueyan and Dou, Zi-Yi and Yang, Jianwei and Gan, Zhe and Li, Linjie and Li, Chunyuan and Dai, Xiyang and Wang, Jianfeng and Yuan, Lu and Peng, Nanyun and Wang, Lijuan and Lee, Yong Jae and Gao, Jianfeng},
  title       = {Generalized Decoding for Pixel, Image and Language},
  publisher   = {arXiv},
  year        = {2022},
}
```

## Contact Information

For issues to use our X-Chat, please submit a GitHub issue or contact Jianwei Yang (jianwyan@microsoft.com).
