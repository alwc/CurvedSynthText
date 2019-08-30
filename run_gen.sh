#!/bin/sh

rlaunch --cpu=20 --memory=8000  --  mdl gen.py \
    --begin_index=6000 \
    --end_index=8000 \
    --max_time=100 \
    --gen_data_path=/unsullied/sharefs/_csg_algorithm/Interns/guanyushuo/OCR/SynthText/SynthTextData/GenData/data \
    --bg_data_path=/unsullied/sharefs/_csg_algorithm/Interns/guanyushuo/OCR/SynthText/SynthTextData \
    --instance_per_image=50 \
    --output_path=/unsullied/sharefs/_csg_algorithm/Interns/guanyushuo/OCR/SynthText/SynthTextData/results_curved_bin2\
    --jobs=20
