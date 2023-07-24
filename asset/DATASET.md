# Preparing Dataset
Our dataloader follows [Detectron2](https://github.com/facebookresearch/detectron2) contains (1) A dataset registrator. (2) A dataset mapper. (3) A dataset loader. We modify the dataset registrator and mapper for different datasets.

## Training Dataset

### COCO
```sh
Prepare panoptic_train2017, panoptic_semseg_train2017 exactly the same as [Mask2Fomer](https://github.com/facebookresearch/Mask2Former/tree/main/datasets)

# Download .pth and .json file
wget -P ../xdecoder_data https://github.com/microsoft/X-Decoder/releases/download/coco/caption_class_similarity.pth
wget -P ../xdecoder_data https://huggingface.co/xdecoder/X-Decoder/blob/main/captions_train2017_filtrefgumdval_filtvlp.json
wget -P ../xdecoder_data https://huggingface.co/xdecoder/X-Decoder/blob/main/grounding_train2017_filtrefgumdval_filtvlp.json
wget -P ../xdecoder_data https://huggingface.co/xdecoder/X-Decoder/blob/main/panoptic_train2017_filtrefgumdval_filtvlp.json
```

```
.xdecoder_data
└── coco/
    ├── train2017/
    ├── val2017/
    ├── panoptic_train2017/
    ├── panoptic_semseg_train2017/
    └── annotations/
        ├── caption_class_similarity.pth
        ├── panoptic_train2017_filtrefgumdval_filtvlp.json
        └── grounding_train2017_filtrefgumdval_filtvlp.json
```

### 4M Image Text Pairs
We follow the exact data preparation for the image text pair data using https://github.com/dandelin/ViLT/blob/master/DATA.md
```
# The pretrained arrow file are put under ../xdecoder_data/pretrain_arrows_code224 with the following list of files.
["filtcoco2017val_caption_karpathy_train.arrow", "filtcoco2017val_caption_karpathy_val.arrow", "filtcoco2017val_caption_karpathy_restval.arrow"] + ["code224_vg.arrow"] + [f"code224_sbu_{i}.arrow" for i in range(9)] + [f"code224_conceptual_caption_train_{i}.arrow" for i in range(31)]
Please filter out coco2017 validation set from the karpathy training split or delete coco trainin data from ./datasets/register_vlp_datasets.py
```

```
.xdecoder_data
└── pretrain_arrows_code224/
    ├── filtcoco2017val_caption_karpathy_train.arrow
    ├── ...
    ├── code224_vg.arrow
    ├── code224_sbu_0.arrow
    ├── ...
    └── code224_conceptual_caption_train_0.arrow
```


### Note
<img src="https://user-images.githubusercontent.com/11957155/226159078-7f817452-76f8-44f4-af7a-9f13f3e02554.png" width="500">
There are overlap between COCO2017, COCO-Karpathy and REF-COCO dataset, and ref-coco is all overalp with the COCO2017 training data, we have exclude the refcocog-umd validation, coco-karpathy test split during training.

## Evaluation Dataset
### ADE20K, Cityscapes
Please Refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets).

### BDD100K
Please download the 10k split of BDD100k at https://doc.bdd100k.com/download.html#id1

#### Expected dataset structure for cityscapes:
```
.xdecoder_data
└── bdd100k/
    ├── images/
    │   └── 10k/
    │       ├── test
    │       ├── train
    │       └── val
    └── labels/
        ├── ins_seg
        ├── pan_seg
        └── sem_seg
```

### RefCOCO
Please download the original refcoco datasets at https://github.com/lichengunc/refer.

#### Expected dataset structure for refcoco:
```
.xdecoder_data
└── refcocoseg/
    └── refcocog/
        ├── instances.json
        ├── refs(google).p
        └── refs(umd).p
```

Also download the coco dataset at https://cocodataset.org/#home:
#### Expected dataset structure for coco:
```
.xdecoder_data
└── coco/
    ├── annotations
    ├── train2017
    └── val2017
```

After preparing the dataset, run the following command:

```sh
# NOTE: Please modify coco_root and ref_root
python3 refcoco2json.py
```

### SUN-RGBD


### SCAN-Net


