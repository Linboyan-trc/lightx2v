import argparse

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_animate_runner import WanAnimateRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_sf_runner import WanSFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


######################################################################################################################################################
# 1. 初始化runner
def init_runner(config):
    # 1. 关闭自动求导机制
    # 1. 因为推理时用反向传播去更新参数，所以不需要记录自动求导
    torch.set_grad_enabled(False)

    # 2.1 初始化runner，设置配置，设置设备GPU
    runner = RUNNER_REGISTER[config["model_cls"]](config)

    # 2.2 初始化model
    runner.init_modules()
    return runner


######################################################################################################################################################
# 1. 推理入口
def main():
    ##################################################
    # 1. 解析参数
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=42, help="The seed for random generator")
    # parser.add_argument(
    #     "--model_cls",
    #     type=str,
    #     required=True,
    #     choices=[
    #         "wan2.1",
    #         "wan2.1_distill",
    #         "wan2.1_vace",
    #         "wan2.1_sf",
    #         "seko_talk",
    #         "wan2.2_moe",
    #         "wan2.2",
    #         "wan2.2_moe_audio",
    #         "wan2.2_audio",
    #         "wan2.2_moe_distill",
    #         "qwen_image",
    #         "wan2.2_animate",
    #     ],
    #     default="wan2.1",
    # )
    # parser.add_argument("--task", type=str, choices=["t2v", "i2v", "t2i", "i2i", "flf2v", "vace", "animate", "s2v"], default="t2v")
    # parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--sf_model_path", type=str, required=False)
    # parser.add_argument("--config_json", type=str, required=True)
    # parser.add_argument("--use_prompt_enhancer", action="store_true")
    # parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    # parser.add_argument("--negative_prompt", type=str, default="")
    # parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    # parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    # parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file or directory for audio-to-video (s2v) task")
    # parser.add_argument(
    #     "--src_ref_images",
    #     type=str,
    #     default=None,
    #     help="The file list of the source reference images. Separated by ','. Default None.",
    # )
    # parser.add_argument(
    #     "--src_video",
    #     type=str,
    #     default=None,
    #     help="The file of the source video. Default None.",
    # )
    # parser.add_argument(
    #     "--src_mask",
    #     type=str,
    #     default=None,
    #     help="The file of the source mask. Default None.",
    # )
    # parser.add_argument("--save_result_path", type=str, default=None, help="The path to save video path/file")
    # parser.add_argument("--return_result_tensor", action="store_true", help="Whether to return result tensor. (Useful for comfyui)")
    # args = parser.parse_args()
    class Args:
        seed = 42
        model_cls = "wan2.1"
        task = "t2v"
        model_path = "/data/nvme0/models/Wan-AI/Wan2.1-T2V-1.3B"
        config_json = "/home/yangrongjin/lightx2v/configs/wan/wan_t2v.json"
        use_prompt_enhancer = False
        prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        image_path = ""
        last_frame_path = ""
        audio_path = ""
        src_ref_images = None
        src_video = None
        src_mask = None
        save_result_path = "/home/yangrongjin/lightx2v/save_results/output_lightx2v_wan_t2v.mp4"
        return_result_tensor = False

    args = Args()
    seed_all(args.seed)

    # 2. 设置配置
    config = set_config(args)

    # 3. 分布式推理
    if config["parallel"]:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        set_parallel_config(config)
    print_config(config)

    # 4. 开始推理
    with ProfilingContext4DebugL1("Total Cost"):
        # 4.1 初始化runner
        runner = init_runner(config)
        input_info = set_input_info(args)

        # 4.2 开始推理
        runner.run_pipeline(input_info)

    # 5. 分布式训练进程清理
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
