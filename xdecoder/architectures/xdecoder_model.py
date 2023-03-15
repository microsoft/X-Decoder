# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random

from timm.models.layers import trunc_normal_
from nltk.stem.lancaster import LancasterStemmer
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .registry import register_model
from ..utils import configurable, get_class_names
from ..backbone import build_backbone, Backbone
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, bbox_postprocess
from ..language import build_language_encoder
from ..language.misc import vl_similarity
from utils.prompt_engineering import prompt_engineering
from utils.constants import COCO_PANOPTIC_CLASSES

st = LancasterStemmer()


class GeneralizedXdecoder(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        phrase_prob: float,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        train_dataset_name: str,
        retrieval_emsemble: bool,
        backbone_dim: int,
        dim_proj: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        # caption argument
        self.task_switch = task_switch
        self.phrase_prob = phrase_prob

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = get_class_names(train_dataset_name)

        self.retrieval_emsemble = retrieval_emsemble
        # backbone itc loss
        if task_switch['retrieval'] and retrieval_emsemble:
            self.backbone_proj = nn.Parameter(torch.empty(backbone_dim, dim_proj))
            trunc_normal_(self.backbone_proj, std=.02)

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # TODO: Training config to be release
        no_object_weight = None
        phrase_prob = None
        deep_supervision = None
        criterion = None
        train_dataset_name = None
        matcher = None
        top_x_layers = None
        loss_weights = None
        losses = None

        # loss weights, switcher for task, and top layers to compute loss
        task_switch = {'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg.get('MASK', True),
                       'caption': dec_cfg['CAPTION'].get('ENABLED', False),
                       'captioning': dec_cfg['CAPTIONING'].get('ENABLED', False),
                       'retrieval': dec_cfg['RETRIEVAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
                       'vqa': dec_cfg['VQA'].get('ENABLED', False),}

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)        
        sem_seg_head = build_xdecoder_head(cfg, backbone.output_shape(), lang_encoder, extra)


        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": None,
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "retrieval_emsemble": dec_cfg['RETRIEVAL']['ENSEMBLE'],
            "backbone_dim": cfg['MODEL']['BACKBONE_DIM'],
            "dim_proj": cfg['MODEL']['DIM_PROJ'],
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            assert False, "Currently not support training X-Decoder"
        else:
            if mode == 'retrieval':
                return self.evaluate_retrieval(batched_inputs)
            elif mode == 'captioning':
                return self.evaluate_captioning(batched_inputs)
            elif mode == 'classification':
                return self.evaluate_classification(batched_inputs)
            elif mode == 'grounding_refcoco':
                return self.evaluate_grounding(batched_inputs, mode)
            else:
                return self.evaluate(batched_inputs)

    def evaluate(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        box_pred_results = outputs["pred_boxes"] if self.task_switch['bbox'] else [None for i in range(len(mask_pred_results))]
        caption_pred_results = outputs["pred_captions"] if self.task_switch['caption'] else [None for i in range(len(mask_pred_results))]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bicubic",
            align_corners=False,
            antialias=True
        )

        input_size = mask_pred_results.shape[-2:]
        keep_sem_bgd = self.metadata.keep_sem_bgd if hasattr(self.metadata, 'keep_sem_bgd') else False
        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, box_pred_result, caption_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, box_pred_results, caption_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, keep_sem_bgd)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            
            # instance segmentation inference
            if self.instance_on:
                if self.task_switch['bbox']:
                    box_pred_result = bbox_postprocess(box_pred_result, input_size, image_size, height, width)
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, box_pred_result)
                processed_results[-1]["instances"] = instance_r
            if self.task_switch['caption']:
                processed_results[-1]["captions"] = caption_pred_result
                processed_results[-1]["masks"] = mask_pred_result

        return processed_results

    def evaluate_retrieval(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)
        v_emb_it = outputs['pred_captions'][:,-1]

        # compute backbone score
        if self.task_switch['retrieval'] and self.retrieval_emsemble:
            _v_emb_it = features['res5']
            bs,nc,_,_ = _v_emb_it.shape
            _v_emb_it = _v_emb_it.reshape(bs,nc,-1)
            _v_emb_it = F.adaptive_avg_pool1d(_v_emb_it, 1).reshape(bs,nc) @ self.backbone_proj

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            caption_ids = []
            t_emb_its = []
            processed_results.append({})
            for caption in batch_data['captions']:
                lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(caption)
                t_emb_it = lang_results['class_emb']
                caption_ids.append(batch_data['image_id'])
                t_emb_its.append(t_emb_it)

            t_emb_it = torch.cat(t_emb_its, dim=0)

            image_embeds = [v_emb_it[idx].unsqueeze(0)]
            if self.task_switch['retrieval'] and self.retrieval_emsemble:
                image_embeds += [_v_emb_it[idx].unsqueeze(0)]
            caption_results = {
                    'image_embeds': image_embeds,
                    'text_embeds': t_emb_it,
                    'caption_ids': caption_ids,
                    'image_ids': batch_data['image_id'],
                }
            processed_results[-1]["caption"] = caption_results            
        return processed_results

    def evaluate_captioning(self, batched_inputs, extra={}):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        if not hasattr(self, 'start_token'):
            self.start_token = torch.tensor([[49406]*77], device=self.device)
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)

        captioning_mask = None
        if 'captioning_mask' in batched_inputs[-1]:
            captioning_mask = torch.cat([x['captioning_mask'] for x in batched_inputs])

        extra.update({'start_token': self.start_token, 'captioning_mask': captioning_mask})
        outputs = self.sem_seg_head(features, target_queries=queries_grounding, task='captioning_infer', extra=extra)

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            processed_results.append({})
            processed_results[-1]["captioning_token"] = outputs['pred_captionings'][idx]
            processed_results[-1]["captioning_text"] = outputs['pred_texts'][idx].split('.')[0]
            processed_results[-1]["image_id"] = batched_inputs[idx]['image_id']
            
        return processed_results

    def evaluate_classification(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            processed_results.append({})
            processed_results[-1]["pred_class"] = outputs['pred_logits'][idx,-1]
        return processed_results

    def evaluate_grounding_baseline(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding)

        mask_pred_results = outputs["pred_masks"]
        caption_pred_results = outputs["pred_captions"] if self.task_switch['caption'] else [None for i in range(len(mask_pred_results))]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bicubic",
            align_corners=False,
            antialias=True
        )

        processed_results = []
        for mask_pred_result, caption_pred_result, input_per_image, image_size in zip(
            mask_pred_results, caption_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )[:-1]

            texts_all = input_per_image['groundings']['texts']
            grd_masks = []
            for texts in texts_all:
                if mode == 'grounding_refcoco':
                    self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(texts, name='grounding', prompt=False, is_eval=True)
                elif mode == 'grounding_phrasecut':
                    self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(texts, name='grounding', prompt=True, is_eval=False)
                t_emb = getattr(self.sem_seg_head.predictor.lang_encoder, "{}_text_embeddings".format('grounding')).t()
                v_emb = caption_pred_result[:-1]
                v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
                vt_sim = v_emb @ t_emb
                max_id = vt_sim.max(0)[1][0]
                grd_masks += [mask_pred_result[max_id]]
            processed_results[-1]['grounding_mask'] = torch.stack(grd_masks)

        return processed_results

    def evaluate_grounding(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        extra = {}
        # mask_pred_results = []
        # for idx, batch_per_image in enumerate(batched_inputs):
        #     grd_texts = batch_per_image['groundings']['texts']
        #     grd_masks = []
        #     for anno_text in grd_texts:
        #         gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings([anno_text[0]], name='grounding', token=False, norm=False)
        #         token_emb = gtext['token_emb']
        #         tokens = gtext['tokens']
            
        #         grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
        #         extra['grounding_tokens'] = grd_emb[:,None]

        #         assert len(images.tensor) == 1, "grounding evaluation only support single batch size now"
        #         features = self.backbone(images.tensor)
        #         outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')
                
        #         pred_gmasks = outputs['pred_masks'][idx,self.num_queries:2*self.num_queries-1]
        #         v_emb = outputs['pred_captions'][idx,self.num_queries:2*self.num_queries-1]
        #         t_emb = grd_emb[-1:]

        #         t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
        #         v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

        #         temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
        #         out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
                
        #         matched_id = out_prob.max(0)[1]
        #         grd_masks += [pred_gmasks[matched_id,:,:]]
        #     mask_pred_results += [torch.cat(grd_masks)]

        # comment for multi object inference.
        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            grd_texts = [x[0] for x in grd_texts]

            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            extra['grounding_tokens'] = query_emb[:,None]

            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

            pred_gmasks = outputs['pred_masks'][idx,self.num_queries:2*self.num_queries-1]
            v_emb = outputs['pred_captions'][idx,self.num_queries:2*self.num_queries-1]
            t_emb = gtext['class_emb']

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            
            matched_id = out_prob.max(0)[1]
            mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            # upsample masks
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = mask_pred_result

            # compute bbox
            # bbox = BitMasks(mask_pred_result > 0).get_bounding_boxes()
            # bbox = BoxMode.convert(bbox.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            # processed_results[-1]['grounding_box'] = bbox

        return processed_results

    def evaluate_vqa(self, batched_inputs, mode):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        img_bs = images.tensor.shape[0]

        targets_vlp = self.prepare_vqa_targets(batched_inputs, images.tensor.device)
        
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_vlp=targets_vlp, task='vqa') #, extra={'start_token': self.start_token})

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            processed_results.append({})
            pred = outputs['pred_captionings'][idx].argmax(dim=-1)
            id2answer = MetadataCatalog.get("vqa_metadata").id2answer
            vqa_preds = [id2answer[pred.item()]] 
            processed_results[-1]["preds"] = vqa_preds

        return processed_results

    def prepare_vqa_targets(self, batched_inputs, device):
        input_ids = []
        attention_mask = []
        for cnt, x in enumerate(batched_inputs):
            captions = x['questions']
            randid = random.randint(0, len(captions)-1)
            input_ids += x['tokens']['input_ids'][randid:randid+1]
            attention_mask += x['tokens']['attention_mask'][randid:randid+1]

        input_ids = torch.stack(input_ids).cuda()
        attention_mask = torch.stack(attention_mask).cuda()
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)

        target_vlp = []
        for cnt, x in enumerate(batched_inputs):
            target_dict = {}
            target_dict["caption_tokens"] = lang_results['token_emb'][cnt:cnt+1]
            target_dict["caption_proj"] = lang_results['class_emb'][cnt:cnt+1]
            target_dict["caption_tokenids"] = lang_results['tokens']['input_ids'][cnt:cnt+1]
            target_dict["caption_mask"] = lang_results['tokens']['attention_mask'][cnt:cnt+1]            
            target_vlp.append(target_dict)
        return target_vlp

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if box_pred is not None:
            box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

            if box_pred is not None:
                box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        if box_pred is not None:
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result



@register_model
def get_xdecoder_model(cfg, **kwargs):
    return GeneralizedXdecoder(cfg)