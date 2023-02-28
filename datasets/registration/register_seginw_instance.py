# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager

_CATEGORIES = ['Elephants', 'Hand-Metal', 'Watermelon', 'House-Parts', 'HouseHold-Items', 'Strawberry', 'Fruits', 'Nutterfly-Squireel', 
                'Hand', 'Garbage', 'Chicken', 'Rail', 'Airplane-Parts', 'Brain-Tumor', 'Poles', 'Electric-Shaver', 'Bottles', 
                'Toolkits', 'Trash', 'Salmon-Fillet', 'Puppies', 'Tablets', 'Phones', 'Cows', 'Ginger-Garlic']

_PREDEFINED_SPLITS_SEGINW = {
    "seginw_{}_val".format(cat): (
        "valid",
        "seginw/{}".format(cat), # image_root
        "_annotations_min1cat.coco.json", # annot_root
    ) for cat in _CATEGORIES
}
_PREDEFINED_SPLITS_SEGINW.update({
    "seginw_{}_train".format(cat): (
        "train",
        "seginw/{}".format(cat), # image_root
        "_annotations_min1cat.coco.json", # annot_root
    ) for cat in _CATEGORIES
})


def get_metadata():
    # meta = {"thing_dataset_id_to_contiguous_id": {}}
    meta = {}
    return meta


def load_seginw_json(name, image_root, annot_json, metadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(annot_json) as f:
        json_info = json.load(f)
        
    # build dictionary for grounding
    grd_dict = collections.defaultdict(list)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    ret = []
    for image in json_info["images"]:
        image_id = int(image["id"])
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]

        if 'train' in name and len(grounding_anno) == 0:
            continue

        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "inst_info": grounding_anno,
            }
        )

    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_seginw(
    name, metadata, image_root, annot_json):
    DatasetCatalog.register(
        name,
        lambda: load_seginw_json(name, image_root, annot_json, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="seginw",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_seginw(root):
    for (
        prefix,
        (split, folder_name, annot_name),
    ) in _PREDEFINED_SPLITS_SEGINW.items():
        register_seginw(
            prefix,
            get_metadata(),
            os.path.join(root, folder_name, split),
            os.path.join(root, folder_name, split, annot_name),
        )


_root = os.getenv("DATASET", "datasets")
register_all_seginw(_root)
