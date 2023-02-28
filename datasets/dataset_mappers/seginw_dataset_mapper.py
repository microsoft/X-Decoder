# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import random

import scipy.io
import numpy as np
import torch
from PIL import Image

from torchvision import transforms

from pycocotools import mask
from detectron2.structures import BitMasks, Boxes, Instances
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from xdecoder.utils import configurable

__all__ = ["SeginWDatasetMapper"]

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    min_scale = cfg_input['MIN_SCALE']
    max_scale = cfg_input['MAX_SCALE']

    augmentation = []


    if cfg_input['RANDOM_FLIP'] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                vertical=cfg_input['RANDOM_FLIP'] == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])
    
    return augmentation


# This is specifically designed for the COCO dataset.
class SeginWDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        tfm_gens=None,
        img_format=None,
        min_size_test=None,
        max_size_test=None,
        mean=None,
        std=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.img_format = img_format

        self.is_train = is_train
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.pixel_mean = torch.tensor(mean)[:,None,None]
        self.pixel_std = torch.tensor(std)[:,None,None]

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            tfm_gens = build_transform_gen(cfg, is_train)
        else:
            tfm_gens = None

        ret = { "is_train": is_train,
                "tfm_gens": tfm_gens,
                "min_size_test": cfg['INPUT']['MIN_SIZE_TEST'],
                "max_size_test": cfg['INPUT']['MAX_SIZE_TEST'],
                "mean": cfg['INPUT']['PIXEL_MEAN'],
                "std": cfg['INPUT']['PIXEL_STD'],
                "img_format": cfg['INPUT']['FORMAT']}
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if self.is_train == False:
            file_name = dataset_dict['file_name']
            image = Image.open(file_name).convert('RGB')
            dataset_dict['width'] = image.size[0]
            dataset_dict['height'] = image.size[1]
            image = self.transform(image)
            image = torch.from_numpy(np.asarray(image).copy())
            image = image.permute(2,0,1)
            dataset_dict['image'] = image
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            grounding_anno = dataset_dict['inst_info']
            # assert len(grounding_anno) > 0
            masks_grd = []
            class_grd = []
            for ann in grounding_anno:
                rle = mask.frPyObjects(
                    ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
                m = mask.decode(rle)
                # sometimes there are multiple binary map (corresponding to multiple segs)
                m = np.sum(m, axis=2)
                m = m.astype(np.uint8)  # convert to np.uint8
                m = transforms.apply_segmentation(m[:,:,None])[:,:,0]
                masks_grd += [m]
                class_grd.append(ann['category_id'])

            is_things = [1 for idx in range(len(class_grd))]
            instances = Instances(image_shape)

            if len(masks_grd) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, image.shape[0], image.shape[1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks_grd])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            instances.gt_classes = torch.tensor(class_grd, dtype=torch.int64)
            instances.is_things = torch.tensor(is_things, dtype=torch.int64)

            dataset_dict["instances"] = instances

        return dataset_dict