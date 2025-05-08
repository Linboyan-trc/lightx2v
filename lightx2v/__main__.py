import argparse
import torch
import torch.distributed as dist
import json

from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import seed_all

from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.graph_runner import GraphRunner

from lightx2v.common.ops import *

######################################################################################################################################################
# 1. 主代码
if __name__ == "__main__":
    ############################################################### 1.1 解析参数 ###############################################################
    # 1.1 必须: model_cls, model_path, prompt, task
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan"], default="hunyuan")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None, help="The path to input image file or path for image-to-video (i2v) task")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    
    # 1.2 config.json文件: 
    # 1.2.1 推理步数、视频长、宽、高
    # 1.2.2 注意力选择attention_type
    # 1.2.3 seed
    # 1.2.4 mm_config: mm_type, weight_auto_quant
    # 1.2.5 多卡并行算法parallel_attn_type
    parser.add_argument("--config_json", type=str, required=True)
    
    # 1.3 必须: 视频保存路径save_video_path
    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")

    # 1.4 hunyuan
    # 1.4.1 feature_caching: 在configs/....json中

    # 1.5 wan
    # 1.5.1 negtive_prompt
    parser.add_argument("--negative_prompt", type=str, default="")

    ##########################################################################################################################################
    # 1.6 存储在变量args中
    args = parser.parse_args()
    print(f"args: {args}")

    with ProfilingContext("Total Cost"):
        # 1.7 继续设置，将剩下的配置读入
        # 1.7 并且所有属性都会被设置为一级属性，通过config.xxx取出
        config = set_config(args)
        print(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")

        #################################################### 1.9 seed，多卡并行，暂未涉及 ####################################################
        seed_all(config.seed)

        if config.parallel_attn_type:
            dist.init_process_group(backend="nccl")
        ##################################################################################################################################

        # 1.9 在RUNNER_REGISTER中已经提前注册好了"hunyuan" -> HunyuanRunner, "wan2.1" -> WanRunner两个类名 -> 类的映射
        # 1.9.1 这两个类HunyuanRunner, WanRunner都继承DefalutRunner
        # 1.9.2 RUNNER_REGISTER['xxx']()就是新建一个对象，并且传入config进行初始化
        # 1.9.3 调用的是DefalutRunner的初始化方法，只需要传入config
        # 1.9.4 目前不使用图模式，所以默认runner是一个HunyuanRunner类的实例化对象或WanRunner类的实例化对象
        # 1.9.5 实例化一个HunyuanRunner，并初始化
        if CHECK_ENABLE_GRAPH_MODE():
            default_runner = RUNNER_REGISTER[config.model_cls](config)
            runner = GraphRunner(default_runner)
        else:
            runner = RUNNER_REGISTER[config.model_cls](config)

        # 1.9.4 调用HunyuanRunner类或WanRunner类的run_pipeline()方法
        # 1.9.5 调用HunyuanRunner的run_pipeline()开始推理
        runner.run_pipeline()
