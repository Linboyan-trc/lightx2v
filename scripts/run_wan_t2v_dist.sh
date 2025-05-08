#!/bin/bash

# 1. 项目根目录
# 1.1 之后在<项目根目录>/lightx2v/__main__.py开始运行
# 1.1.1 必须主动设置项目根目录
# 1.1.2 设置找依赖的时候，可供筛选的路径列表，并优先在项目目录下寻找
# 1.1.2 能别识别成依赖的前提是.py文件的父目录有一个__init__.py文件
lightx2v_path=
if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi
export PYTHONPATH=${lightx2v_path}:$PYTHONPATH

# 1.2 模型目录, wan在"/mtc/yongyang/models/x2v_models/wan"，包含了Wan2.1-I2V-14B-480P/, Wan2.1-T2V-1.3B/
# 1.2.1 必须主动设置模型目录
model_path=
if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

# 1.3 如果不在脚本中主动设置CUDA_VISIBLE_DEVICES，并export
# 1.3.1 则默认CUDA_VISIBLE_DEVICES为0,1,2,3，并export
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0,1,2,3
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using defalt value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

# 1.4 关闭Hugging Face Transformers库中的分词器并行处理功能
export TOKENIZERS_PARALLELISM=false

# 1.5 性能分析，默认启用
export ENABLE_PROFILING_DEBUG=true

# 1.6 启用多进程，指定项目入口
# 1.6.1 启用了多进程，指定进程数量
# 1.6.2 指定项目入口__main__.py
# 1.6.3 指定模型、模型路径、prompt、task
# 1.6.4 指定推理步数、视频长、宽、高、attention_type、seed
# 1.6.5 指定negative_prompt、sample_guide_scale、sample_shift
# 1.6.6 指定parallel_attn_type、parallel_vae
# 1.6.7 指定保存路径
torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/__main__.py \
--model_cls wan2.1 \
--model_path $model_path \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--task t2v \
--infer_steps 50 \
--target_video_length 84 \
--target_width  832 \
--target_height 480 \
--attention_type flash_attn2 \
--seed 42 \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--sample_guide_scale 6 \
--sample_shift 8 \
--parallel_attn_type ring \
--parallel_vae \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_t2v_dist_ring.mp4 \


torchrun --nproc_per_node=4 ${lightx2v_path}/lightx2v/__main__.py \
--model_cls wan2.1 \
--model_path $model_path \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--task t2v \
--infer_steps 50 \
--target_video_length 81 \
--target_width  832 \
--target_height 480 \
--attention_type flash_attn2 \
--seed 42 \
--negative_prompt 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走 \
--sample_guide_scale 6 \
--sample_shift 8 \
--parallel_attn_type ulysses \
--parallel_vae \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_t2v_dist_ulysses.mp4
