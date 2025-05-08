import torch


####################################################################################################
# 1. 调度器
class BaseScheduler:
    ############################## 1. 初始化 ##############################
    # 1. 初始化
    # 1.1 属性: config, step_index, latents
    def __init__(self, config):
        self.config = config
        self.step_index = 0
        self.latents = None

    ######################################################################
    # 2. pre计算
    def step_pre(self, step_index):
        self.step_index = step_index
        self.latents = self.latents.to(dtype=torch.bfloat16)

    def clear(self):
        pass
