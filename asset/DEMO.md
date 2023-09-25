### Demo

OpenVocab Semantic Segmentation
```sh
CUDA_VISIBLE_DEVICES=0 python inference/xdecoder/infer_semseg.py evaluate \
            --conf_files configs/xdecoder/xdecoder_focall_lang.yaml \
            --overrides \
            RESUME_FROM /pth/to/xdecoder_focall_best_openseg.pt
```

OpenVocab Instance Segmentation
```sh
CUDA_VISIBLE_DEVICES=0 python inference/xdecoder/infer_instseg.py evaluate \
            --conf_files configs/xdecoder/xdecoder_focall_lang.yaml \
            --overrides \
            RESUME_FROM /pth/to/xdecoder_focall_best_openseg.pt
```

OpenVocab Panoptic Segmentation
```sh
CUDA_VISIBLE_DEVICES=0 python inference/xdecoder/infer_panoseg.py evaluate \
            --conf_files configs/xdecoder/xdecoder_focall_lang.yaml \
            --overrides \
            RESUME_FROM /pth/to/xdecoder_focall_best_openseg.pt
```

OpenVocab Referring Segmentation
```sh
CUDA_VISIBLE_DEVICES=0 python inference/xdecoder/infer_refseg.py evaluate \
            --conf_files configs/xdecoder/xdecoder_focall_lang.yaml \
            --overrides \
            RESUME_FROM /pth/to/xdecoder_focall_last.pt
```

Region Retrieval
```sh
CUDA_VISIBLE_DEVICES=0 python inference/xdecoder/infer_region_retrieval.py evaluate \
            --conf_files configs/xdecoder/xdecoder_focall_lang.yaml \
            --overrides \
            RESUME_FROM /pth/to/xdecoder_focall_last.pt
```

Image Captioning
```sh
CUDA_VISIBLE_DEVICES=0 python inference/xdecoder/infer_captioning.py evaluate \
            --conf_files configs/xdecoder/xdecoder_focalt_lang.yaml \
            --overrides \
            RESUME_FROM /pth/to/xdecoder_focalt_last_novg.pt
```