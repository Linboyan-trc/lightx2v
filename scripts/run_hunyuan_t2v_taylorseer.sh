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

# 1.2 模型目录, hunyuan在"/mtc/yongyang/models/x2v_models/hunyuan/lightx2v_format"，包含了itv/, t2v/
# 1.2.1 必须主动设置模型目录
model_path=
if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

# 1.3 如果不在脚本中主动设置CUDA_VISIBLE_DEVICES，并export
# 1.3.1 则默认CUDA_VISIBLE_DEVICES为0，并export
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using defalt value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

# 1.4 关闭Hugging Face Transformers库中的分词器并行处理功能
export TOKENIZERS_PARALLELISM=false

# 1.5 性能分析，默认启用
export ENABLE_PROFILING_DEBUG=true

# 1.6 默认运行lightx2v模块__main__.py
# 1.6.1 指定模型、模型路径、prompt、任务类型
# 1.6.2 指定配置文件，含有: 推理步数、视频长、宽、高、保存路径、attention_type、seed、mm_config，用于加速计算
# 1.6.3 指定保存路径
python -m lightx2v \
--model_cls hunyuan \
--model_path $model_path \
--prompt "A cat walks on the grass, realistic style." \
--task t2v \
--config_json ${lightx2v_path}/configs/caching/hunyuan_t2v_TaylorSeer.json \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_hy_t2v.mp4
