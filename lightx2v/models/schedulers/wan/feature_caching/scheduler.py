import torch
from ..scheduler import WanScheduler


# 1. 调度器
class WanSchedulerTeaCaching(WanScheduler):
    # 1. 初始化
    def __init__(self, config):
        # 1.1 父类初始化
        super().__init__(config)

        # 1.2 计数器，推理步数*2
        self.cnt = 0
        self.num_steps = self.config.infer_steps * 2

        # 1.3 阈值；奇偶累计L1；奇偶前步e0；奇偶前步残差；
        self.teacache_thresh = self.config.teacache_thresh
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None

        # 1.3 是否使用返回步数机制
        self.use_ret_steps = self.config.use_ret_steps

        # 1.3 根据i2v/t2v, 是否使用返回步数机制，设置coefficients, ret_steps, cutoff_steps
        # 1.4 对于i2v
        if self.config.task == "i2v":
            # 1.4.1 使用返回步数机制
            if self.use_ret_steps:
                # 1.4.1 如果是480p
                if self.config.target_width == 480 or self.config.target_height == 480:
                    self.coefficients = [
                        2.57151496e05,
                        -3.54229917e04,
                        1.40286849e03,
                        -1.35890334e01,
                        1.32517977e-01,
                    ]
                
                # 1.4.2 如果是720p
                if self.config.target_width == 720 or self.config.target_height == 720:
                    self.coefficients = [
                        8.10705460e03,
                        2.13393892e03,
                        -3.72934672e02,
                        1.66203073e01,
                        -4.17769401e-02,
                    ]
                
                # 1.4.3 设置返回步数，和截断步数
                self.ret_steps = 5 * 2
                self.cutoff_steps = self.config.infer_steps * 2
            
            # 1.4.2 不使用返回步数机制
            else:
                # 1.4.1 如果是480p
                if self.config.target_width == 480 or self.config.target_height == 480:
                    self.coefficients = [
                        -3.02331670e02,
                        2.23948934e02,
                        -5.25463970e01,
                        5.87348440e00,
                        -2.01973289e-01,
                    ]
                
                # 1.4.2 如果是720p
                if self.config.target_width == 720 or self.config.target_height == 720:
                    self.coefficients = [
                        -114.36346466,
                        65.26524496,
                        -18.82220707,
                        4.91518089,
                        -0.23412683,
                    ]
                
                # 1.4.3 设置返回步数，和截断步数
                self.ret_steps = 1 * 2
                self.cutoff_steps = self.config.infer_steps * 2 - 2

        # 1.5 对于t2v
        elif self.config.task == "t2v":
            # 1.4.1 使用返回步数机制
            if self.use_ret_steps:
                # 1.4.1 如果是1.3B
                if "1.3B" in self.config.model_path:
                    self.coefficients = [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02]
                
                # 1.4.2 如果是14B
                if "14B" in self.config.model_path:
                    self.coefficients = [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01]
                
                # 1.4.3 设置返回步数，和截断步数
                self.ret_steps = 5 * 2
                self.cutoff_steps = self.config.infer_steps * 2
            
            # 1.4.2 不使用返回步数机制
            else:
                # 1.4.1 如果是1.3B
                if "1.3B" in self.config.model_path:
                    self.coefficients = [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01]
                
                # 1.4.2 如果是14B
                if "14B" in self.config.model_path:
                    self.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
                
                # 1.4.3 设置返回步数，和截断步数
                self.ret_steps = 1 * 2
                self.cutoff_steps = self.config.infer_steps * 2 - 2

    def clear(self):
        if self.previous_e0_even is not None:
            self.previous_e0_even = self.previous_e0_even.cpu()
        if self.previous_e0_odd is not None:
            self.previous_e0_odd = self.previous_e0_odd.cpu()
        if self.previous_residual_even is not None:
            self.previous_residual_even = self.previous_residual_even.cpu()
        if self.previous_residual_odd is not None:
            self.previous_residual_odd = self.previous_residual_odd.cpu()
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        torch.cuda.empty_cache()
