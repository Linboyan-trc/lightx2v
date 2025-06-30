import numpy as np
from ..transformer_infer import WanTransformerInfer
import torch


class WanTransformerInferTeaCaching(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        # 1. 根据config选择调制后的噪声
        modulated_inp = embed0 if self.scheduler.use_ret_steps else embed

        # 2. 对于单步去噪的第一次transformer推理
        if self.scheduler.cnt % 2 == 0:  # even -> conditon
            # 2.1 标记为条件推理
            self.scheduler.is_even = True

            # 2.2 如果不在可以使用缓存的范围内，就必须计算
            if self.scheduler.cnt < self.scheduler.ret_steps or self.scheduler.cnt >= self.scheduler.cutoff_steps:
                should_calc_even = True
                self.scheduler.accumulated_rel_l1_distance_even = 0
            
            # 2.3 否则使用缓存
            else:
                rescale_func = np.poly1d(self.scheduler.coefficients)
                self.scheduler.accumulated_rel_l1_distance_even += rescale_func(
                    ((modulated_inp - self.scheduler.previous_e0_even).abs().mean() / self.scheduler.previous_e0_even.abs().mean()).cpu().item()
                )
                if self.scheduler.accumulated_rel_l1_distance_even < self.scheduler.teacache_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.scheduler.accumulated_rel_l1_distance_even = 0
            self.scheduler.previous_e0_even = modulated_inp.clone()

        # 3. 对于单步去噪的第二次transformer推理
        else:  # odd -> unconditon
            self.scheduler.is_even = False
            if self.scheduler.cnt < self.scheduler.ret_steps or self.scheduler.cnt >= self.scheduler.cutoff_steps:
                should_calc_odd = True
                self.scheduler.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.scheduler.coefficients)
                self.scheduler.accumulated_rel_l1_distance_odd += rescale_func(
                    ((modulated_inp - self.scheduler.previous_e0_odd).abs().mean() / self.scheduler.previous_e0_odd.abs().mean()).cpu().item()
                )
                if self.scheduler.accumulated_rel_l1_distance_odd < self.scheduler.teacache_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.scheduler.accumulated_rel_l1_distance_odd = 0
            self.scheduler.previous_e0_odd = modulated_inp.clone()

        # 4. 对于单步去噪的第一次transformer推理
        if self.scheduler.is_even:
            if not should_calc_even:
                x += self.scheduler.previous_residual_even
            else:
                ori_x = x.clone()
                x = super().infer(
                    weights,
                    grid_sizes,
                    embed,
                    x,
                    embed0,
                    seq_lens,
                    freqs,
                    context,
                )
                self.scheduler.previous_residual_even = x - ori_x
                if self.config["cpu_offload"]:
                    ori_x = ori_x.to("cpu")
                    del ori_x
                    torch.cuda.empty_cache()
        
        # 5. 对于单步去噪的第二次transformer推理
        else:
            if not should_calc_odd:
                x += self.scheduler.previous_residual_odd
            else:
                ori_x = x.clone()
                x = super().infer(
                    weights,
                    grid_sizes,
                    embed,
                    x,
                    embed0,
                    seq_lens,
                    freqs,
                    context,
                )
                self.scheduler.previous_residual_odd = x - ori_x
                if self.config["cpu_offload"]:
                    ori_x = ori_x.to("cpu")
                    del ori_x
                    torch.cuda.empty_cache()
        
        return x
