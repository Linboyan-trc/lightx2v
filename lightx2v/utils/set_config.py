import json
import os
from easydict import EasyDict
from loguru import logger


############################################################################################################################################
# 1. 获取默认属性
def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
        "parallel_attn_type": None,  # [None, "ulysses", "ring"]
        "parallel_vae": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
        "teacache_thresh": 0.26,
        "use_ret_steps": False,
        "use_bfloat16": True,
        "lora_configs": None,  # List of dicts with 'path' and 'strength' keys
        "mm_config": {},
        "use_prompt_enhancer": False,
    }
    return default_config


############################################################################################################################################
# 1. 所有的属性都会被设置为一级属性，通过config.xxx取出
def set_config(args):
    ##################################################################################################################################
    # 1.1 设置config的属性，一共来自: 默认属性，参数，本项目中的配置文件，模型目录下的配置文件
    # 1.1 获取默认属性
    config = get_default_config()

    # 1.2 根据args参数更新部分属性
    config.update({k: v for k, v in vars(args).items()})

    # 1.3 将config转化为字典，使得属性可以用config.xxx取出
    config = EasyDict(config)

    # 1.4 读取参数指定的配置文件，然后更新部分属性
    with open(config.config_json, "r") as f:
        config_json = json.load(f)
    config.update(config_json)

    # 1.5 如果模型目录下也有config.json文件，读取，继续更新部分属性
    if os.path.exists(os.path.join(config.model_path, "config.json")):
        with open(os.path.join(config.model_path, "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)

    # 1.6 如果本次推理使用的不是原模型，而是量化的模型
    # 1.6 量化模型的目录下也有config.json文件，那就读取，继续更新部分属性
    if config.get("dit_quantized_ckpt", None) is not None:
        config_path = os.path.join(config.dit_quantized_ckpt, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config = json.load(f)
            config.update(model_config)

    ##################################################################################################################################
    # 2.1 target_video_length要求是4n+1，不满足的话，新找的4n+1是小于原来的target_video_length中最大的那个
    # 2.1 这是由于默认配置中vae_stride是(4,8,8)，同时一般我们不在配置文件中指定vae_stride，所以要求target_video_length是4的倍数+1
    if config.task == "i2v":
        if config.target_video_length % config.vae_stride[0] != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {config.vae_stride[0]}. Rounding to the nearest number.")
            config.target_video_length = config.target_video_length // config.vae_stride[0] * config.vae_stride[0] + 1

    ##################################################################################################################################
    # 3.1 返回配置
    return config
