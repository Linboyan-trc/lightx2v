from .utils import cache_init, cal_type
from ..scheduler import HunyuanScheduler
import torch


####################################################################################################
# 1. 调度器
# 1.1 SchedulerTeaCaching, 继承父类Scheduler
class HunyuanSchedulerTeaCaching(HunyuanScheduler):
    ############################## 1. 初始化 ##############################
    # 1. 初始化
    def __init__(self, config):
        # 1.1 初始化，属性: config, step_index, latents
        # 1.2 初始化，属性: infer_steps, shift, timesteps, sigmas
        # 1.3 初始化，属性: embedded_guidance_scale, generator, noise_pred
        super().__init__(config)

        # 1.4 初始化，属性: cnt, num_steps = infer_steps, teacache_thresh = 0.26, accumulated_rel_l1_distance = 0
        # 1.4.1 teacache_thresh是阈值的意思
        self.cnt = 0
        self.num_steps = self.args.infer_steps
        self.teacache_thresh = self.args.teacache_thresh
        self.accumulated_rel_l1_distance = 0

        # 1.5 初始化，属性: previous_modulated_input, previous_residual, coefficients
        self.previous_modulated_input = None
        self.previous_residual = None
        self.coefficients = [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02]

    ######################################################################
    def clear(self):
        if self.previous_residual is not None:
            self.previous_residual = self.previous_residual.cpu()
        if self.previous_modulated_input is not None:
            self.previous_modulated_input = self.previous_modulated_input.cpu()

        self.previous_modulated_input = None
        self.previous_residual = None
        torch.cuda.empty_cache()


# 1.1 SchedulerTaylorCaching, 继承父类Scheduler
class HunyuanSchedulerTaylorCaching(HunyuanScheduler):
    ############################## 1. 初始化 ##############################
    # 1. 初始化
    def __init__(self, config):
        super().__init__(config)
        self.cache_dic, self.current = cache_init(self.infer_steps)
    ######################################################################

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.current["step"] = step_index
        cal_type(self.cache_dic, self.current)
