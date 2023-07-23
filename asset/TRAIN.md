### Installation

Single GPU
```sh
CUDA_VISIBLE_DEVICES=0 python entry.py train \
            --conf_files configs/xdecoder/segvlp_focalt_lang.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.CAPTIONING.ENABLED True \
            MODEL.DECODER.RETRIEVAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            MODEL.DECODER.CAPTIONING_WEIGHT 8 \
            MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
            MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
            MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
            MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
            COCO.TEST.BATCH_SIZE_TOTAL 1 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 1 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VLP.TEST.BATCH_SIZE_TOTAL 32 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 32 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 32 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            FP16 True \
```

Multi-GPU
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -n 8 python entry.py train \
            --conf_files configs/xdecoder/segvlp_focalt_lang.yaml \
            --overrides \
            COCO.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.CAPTIONING.ENABLED True \
            MODEL.DECODER.RETRIEVAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            MODEL.DECODER.CAPTIONING_WEIGHT 8 \
            MODEL.DECODER.RETRIEVAL_WEIGHT 8 \
            MODEL.DECODER.TOP_CAPTIONING_LAYERS 3 \
            MODEL.DECODER.TOP_RETRIEVAL_LAYERS 3 \
            MODEL.DECODER.TOP_GROUNDING_LAYERS 6 \
            COCO.TEST.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_TOTAL 8 \
            COCO.TRAIN.BATCH_SIZE_PER_GPU 1 \
            VLP.TEST.BATCH_SIZE_TOTAL 128 \
            VLP.TRAIN.BATCH_SIZE_TOTAL 256 \
            VLP.TRAIN.BATCH_SIZE_PER_GPU 32 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            FP16 True \
```
