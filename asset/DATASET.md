# Preparing Dataset
Our dataloader follows [Detectron2](https://github.com/facebookresearch/detectron2) contains (1) A dataset registrator. (2) A dataset mapper. (3) A dataset loader. We modify the dataset registrator and mapper for different datasets.

## Training Dataset
<img src="https://user-images.githubusercontent.com/11957155/226159078-7f817452-76f8-44f4-af7a-9f13f3e02554.png" width="500">
There are overlap between COCO2017, COCO-Karpathy and REF-COCO dataset, and ref-coco is all overalp with the COCO2017 training data, we have exclude the refcocog-umd validation, coco-karpathy test split during training.


## Evaluation Dataset
### ADE20K, Cityscapes, COCO
Please Refer to [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main/datasets).

### BDD100K
Please download the 10k split of BDD100k at https://doc.bdd100k.com/download.html#id1

#### Expected dataset structure for cityscapes:
```
.
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
.
└── refcocoseg/
    └── refcocog/
        ├── instances.json
        ├── refs(google).p
        └── refs(umd).p
```

Also download the coco dataset at https://cocodataset.org/#home:
#### Expected dataset structure for coco:
```
.
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


