# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import time
import logging
import datetime

from mpi4py import MPI
import numpy as np
from PIL import Image
from torchvision import transforms

import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import log_every_n_seconds

from utils.arguments import load_opt_command
from utils.distributed import init_distributed, is_main_process, apply_distributed, synchronize
from utils.misc import hook_metadata, hook_switcher, hook_opt
from datasets import build_evaluator, build_eval_dataloader
from xdecoder import build_model
from xdecoder.BaseModel import BaseModel
from xdecoder.utils import get_class_names
from MaskBLIP.maskblip import MaskBLIP
from MaskBLIP.nlp import get_nouns, map_labels
from utils.constants import CITYSCAPES
from sentence_transformers import SentenceTransformer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from cityscapesscripts.helpers.labels import name2label
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


def main(args=None):
    '''
    Main execution point for xdecoder evaluation.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir
    opt = init_distributed(opt)

    # build model
    label_generator = MaskBLIP(device='cuda')
    model = BaseModel(opt, build_model(opt)).from_pretrained(opt['WEIGHT']).eval().cuda()
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

    # build dataloade
    dataloaders = build_eval_dataloader(opt)
    # evaluation dataset
    dataset_names = opt['DATASETS']['TEST']
    output_root = './output'

    # init metadata
    scores = {}
    summary = {}
    for dataloader, dataset_name in zip(dataloaders, dataset_names):
        # build evaluator
        evaluator = build_evaluator(opt, dataset_name, opt['SAVE_DIR'])
        evaluator.reset()
        with torch.no_grad():

            # setup timer
            total = len(dataloader)
            num_warmup = min(5, total - 1)
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0
            start_data_time = time.perf_counter()
            class_names = get_class_names(dataset_name)
            class_embeds = nlp_model.encode(class_names, convert_to_tensor=True)
            
            t = []
            t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
            transform = transforms.Compose(t)

            for idx, batch in enumerate(dataloader):
                batch[0]['image'] = torch.load('sample_data/image.pt')
                batch[0]['semseg'] = torch.load('sample_data/sem_seg.pt')

                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0
                start_compute_time = time.perf_counter()
                image = batch[0]['image'].float() / 255

                captions = label_generator(image.unsqueeze(0))[1][0]
                names = get_nouns(captions, label_generator.spacy_model) + ['background']
                #names = ["road", "picture", "motorcycle", "photo", "stop", "street", "bus", "background"]


                names_dict = {}
                for name_idx, name in enumerate(names):
                    names_dict[name_idx] = name
                print("names dict: ", names_dict)
                metadata = MetadataCatalog.get(dataset_name)
                model.model.metadata = metadata

                eval_type = model.model.metadata.evaluator_type
                model.model.sem_seg_head.num_classes = len(names) - 1
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
                hook_switcher(model, dataset_name)
                hook_opt(model, dataset_name)
                
                
                
                
                
                
                total_compute_time += time.perf_counter() - start_compute_time
                start_eval_time = time.perf_counter()
                
                # TODO
                # map predicted MaskBLIP classes to the dataset classes in the predicted X-Decoder output
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch, mode=eval_type)

                output = outputs[0]["sem_seg"].argmax(dim=0)#.to().numpy()
                all_predicted_idx = torch.unique(output)
                
                evaluated_names = [names_dict[pred_idx.item()] for pred_idx in all_predicted_idx]
                mapped_names = map_labels(evaluated_names, class_names, class_embeds, nlp_model)# + ['background']
                map_dict = {}
                print("evaluated names", evaluated_names, len(evaluated_names))
                print("mapped names", mapped_names, len(mapped_names))
                background_ignore_idx = []
                for cc, n in enumerate(evaluated_names):
                    mapped_name = mapped_names[cc]
                    if mapped_name == 'background':
                        background_ignore_idx.append(all_predicted_idx[cc])
                        continue
                    elif name2label[mapped_name].ignoreInEval:
                        continue
                    map_dict[all_predicted_idx[cc].item()] = name2label[mapped_name].trainId
                for i, val in enumerate(all_predicted_idx):
                    if val in background_ignore_idx:
                        continue

                    val = val.item()
                    output[output==val] = map_dict[val]
                               
                
                save_output = True
                
                if save_output:
                    image_pth = batch[0]['file_name']
                    #image_pth ='datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_003920_leftImg8bit.png'# batch[0]['file_name']

                    image_ori = Image.open(image_pth).convert("RGB")
                    visual = Visualizer(image_ori, metadata=metadata)
                    demo = visual.draw_sem_seg(output.cpu(), alpha=0.5) # rgb Image

                    if not os.path.exists(output_root):
                        os.makedirs(output_root)
                    
                    demo.save(os.path.join(output_root, str(idx) + '_sem.png'))
                
                
                evaluator.process(batch, [output.cpu()])
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

                if is_main_process()  and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )                
                start_data_time = time.perf_counter()
        # evaluate
        results = evaluator.evaluate()

        # summary
        if eval_type == 'retrieval':
            result_key = 'retrieval'
            summary_keys = ['ir1', 'tr1']
            if is_main_process():
                results[result_key] = results['recall']
        elif eval_type == 'captioning':
            result_key = 'captioning'
            summary_keys = ['Bleu_4', 'CIDEr']
            if is_main_process():
                pop_keys = list(results.keys())
                results[result_key] = {}
                for key in pop_keys:
                    results[result_key][key] = results[key]
                    results.pop(key)
        elif eval_type == 'classification':
            result_key = 'classification'
            summary_keys = ['top1', 'top5']
            if is_main_process():
                results[result_key] = results.pop('class')
        elif 'grounding' in eval_type:
            result_key = 'grounding'
            summary_keys = ['cIoU', 'mIoU', 'precision@0.5']
        else:
            summary_keys = []
            if opt['MODEL']['DECODER']['TEST']['PANOPTIC_ON']:
                result_key = 'panoptic_seg'
                summary_keys += ['PQ', 'SQ', 'RQ']
            if opt['MODEL']['DECODER']['TEST']['INSTANCE_ON']:
                result_key = 'segm'
                summary_keys += ['AP']
            if opt['MODEL']['DECODER']['TEST']['SEMANTIC_ON']:
                result_key = 'sem_seg'
                summary_keys += ['mIoU']

        if is_main_process():
            for eval_type in results.keys():
                for key in results[eval_type]:
                    scores["{}/{}/{}".format(dataset_name, eval_type, key)] = results[eval_type][key]
                    if key in summary_keys:
                        summary["{}/{}/{}".format(dataset_name, eval_type, key)] = results[eval_type][key]

    logger.info(summary)


if __name__ == "__main__":
    main()
    sys.exit(0)

