import json
import os

import torch.distributed as dist
from easydict import EasyDict
from loguru import logger
from torch.distributed.tensor.device_mesh import init_device_mesh


############################################################################################################################################
# 1. 获取默认属性
def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
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
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "use_image_encoder": True,
        "lat_h": None,
        "lat_w": None,
        "tgt_h": None,
        "tgt_w": None,
        "target_shape": None,
        "return_video": False,
        "audio_num": None,
        "person_num": None,
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
    elif os.path.exists(os.path.join(config.model_path, "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
        with open(os.path.join(config.model_path, "low_noise_model", "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)
    elif os.path.exists(os.path.join(config.model_path, "distill_models", "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
        with open(os.path.join(config.model_path, "distill_models", "low_noise_model", "config.json"), "r") as f:
            model_config = json.load(f)
        config.update(model_config)
    elif os.path.exists(os.path.join(config.model_path, "original", "config.json")):
        with open(os.path.join(config.model_path, "original", "config.json"), "r") as f:
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

    if config.audio_path:
        if os.path.isdir(config.audio_path):
            logger.info(f"audio_path is a directory, loading config.json from {config.audio_path}")
            audio_config_path = os.path.join(config.audio_path, "config.json")
            assert os.path.exists(audio_config_path), "config.json not found in audio_path"
            with open(audio_config_path, "r") as f:
                audio_config = json.load(f)
            for talk_object in audio_config["talk_objects"]:
                talk_object["audio"] = os.path.join(config.audio_path, talk_object["audio"])
                talk_object["mask"] = os.path.join(config.audio_path, talk_object["mask"])
            config.update(audio_config)
        else:
            logger.info(f"audio_path is a file: {config.audio_path}")

    assert not (config.save_video_path and config.return_video), "save_video_path and return_video cannot be set at the same time"

    ##################################################################################################################################
    # 3.1 返回配置
    return config


def set_parallel_config(config):
    if config.parallel:
        cfg_p_size = config.parallel.get("cfg_p_size", 1)
        seq_p_size = config.parallel.get("seq_p_size", 1)
        assert cfg_p_size * seq_p_size == dist.get_world_size(), f"cfg_p_size * seq_p_size must be equal to world_size"
        config["device_mesh"] = init_device_mesh("cuda", (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))

        if config.parallel and config.parallel.get("seq_p_size", False) and config.parallel.seq_p_size > 1:
            config["seq_parallel"] = True

        if config.get("enable_cfg", False) and config.parallel and config.parallel.get("cfg_p_size", False) and config.parallel.cfg_p_size > 1:
            config["cfg_parallel"] = True


def print_config(config):
    config_to_print = config.copy()
    config_to_print.pop("device_mesh", None)
    if config.parallel:
        if dist.get_rank() == 0:
            logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
    else:
        logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
