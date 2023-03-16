# X-GPT: Connecting generalist X-Decoder with GPT-3

![demo](https://user-images.githubusercontent.com/11957155/225451614-62c6f129-5c4a-4706-971c-0c90024a2bfa.gif)

## :fire: News

* **[2023.03.14]** We build X-GPT, a multi-modal conversational demo that is built on X-Decoder using GPT-3 and langchain!
* **[2023.03.14]** Feel free to explore the [Full resolution demo](https://youtu.be/GopwIdLb6GU)!

## :notes: Introduction

![intro](https://user-images.githubusercontent.com/11957155/225476626-80ba3c57-f831-41dd-8a66-6516b93e7bc9.png)

### What are the uniques?

This specific project was started with our previous two demos, [all-in-one](https://huggingface.co/spaces/xdecoder/Demo) and [instruct x-decoder](https://huggingface.co/spaces/xdecoder/Instruct-X-Decoder). Then we were inspired by [visual-chatgpt](https://github.com/microsoft/visual-chatgpt) developed by our MSRA collegues to use the [langchain](https://github.com/hwchase17/langchain) to empower a conversational X-Decoder and encompassing all the capacities of our single X-Decoder model.

Our **X-GPT** has several unique and new features:

* **It uses a SINGLE X-Decoder model to support a wide range of vision and vision-language tasks. As such, you do not need separate models for individual tasks!**

* **It delivers the state-of-the-art segmentation performance. It is much better than CLIPSeg or other existing open-vocabulary segmentation systems!**

* **It also supports text-to-image retrieval. You can choose find a real image from your own pool or ask for generating a new image!**

In the next, we will:

* **You may notice we developed a more grounded instructPix2Pix with the support of our X-Decoder. Next we integrate it to our X-GPT!**

## Getting Started

### Installation
```sh
# set up environment
conda create -n xgpt python=3.8
conda activate xgpt

# install dependencies
pip3 install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install -r requirements.txt
sh install_cococapeval.sh

# download x-decoder tiny model
wget https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_last_novg.pt
wget https://projects4jw.blob.core.windows.net/x-decoder/release/xdecoder_focalt_vqa.pt

# create a folder for image uploading and image retrieval, note image_folder is for image retrieval, image is for cache
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
python xgpt.py
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

For issues to use our X-GPT, please submit a GitHub issue or contact Jianwei Yang (jianwyan@microsoft.com).
