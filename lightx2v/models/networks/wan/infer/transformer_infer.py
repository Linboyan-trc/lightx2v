from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
import torch
import numpy as np
from .utils import compute_freqs, compute_freqs_dist, apply_rotary_emb
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.utils.envs import *


class BaseWanTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        # 1.1 config + task + attention
        self.config = config
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")

        # 1.2 blocks_num:30, head_num, in_dim, window_size
        self.blocks_num = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None

        # 1.3 switch status for cache
        self.infer_conditional = True

        if self.config["cpu_offload"]:
            self.offload_granularity = self.config.get("offload_granularity", "block")
            self.weights_stream_mgr = WeightAsyncStreamManager()

    # per block
    def infer_block_1(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, shift_msa, scale_msa):
        # 2. first part before +
        if hasattr(weights, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa) * weights.smooth_norm1_weight.tensor
            norm1_bias = shift_msa * weights.smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa
            norm1_bias = shift_msa

        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * norm1_weight + norm1_bias).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            freqs_i = compute_freqs(q.size(2) // 2, grid_sizes, freqs)
        else:
            freqs_i = compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)

        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=seq_lens)

        if not self.parallel_attention:
            attn_out = weights.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
                model_cls=self.config["model_cls"],
            )
        else:
            attn_out = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=q.shape[0],
                cu_seqlens_qkv=cu_seqlens_q,
            )

        y = weights.self_attn_o.apply(attn_out)
        return y

    def infer_block_2(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa):
        x.add_(y_out * gate_msa.squeeze(0))

        norm3_out = weights.norm3.apply(x)

        if self.task == "i2v":
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        n, d = self.num_heads, self.head_dim
        q = weights.cross_attn_norm_q.apply(weights.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = weights.cross_attn_norm_k.apply(weights.cross_attn_k.apply(context)).view(-1, n, d)
        v = weights.cross_attn_v.apply(context).view(-1, n, d)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device),
        )

        attn_out = weights.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
            model_cls=self.config["model_cls"],
        )

        if self.task == "i2v" and context_img is not None:
            k_img = weights.cross_attn_norm_k_img.apply(weights.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = weights.cross_attn_v_img.apply(context_img).view(-1, n, d)

            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )

            img_attn_out = weights.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
                model_cls=self.config["model_cls"],
            )

            attn_out = attn_out + img_attn_out

        attn_out = weights.cross_attn_o.apply(attn_out)
        return attn_out

    def infer_block_3(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if hasattr(weights, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze(0)) * weights.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze(0) * weights.smooth_norm2_bias.tensor
        else:
            norm2_weight = 1 + c_scale_msa.squeeze(0)
            norm2_bias = c_shift_msa.squeeze(0)

        norm2_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        y = weights.ffn_0.apply(norm2_out * norm2_weight + norm2_bias)
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = weights.ffn_2.apply(y)
        return y

    def infer_block_4(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa):
        x.add_(y_out * c_gate_msa.squeeze(0))
        return x

    def _calculate_q_k_len(self, q, k_lens):
        # Handle query and key lengths (use `q_lens` and `k_lens` or set them to Lq and Lk if None)
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)

        # We don't have a batch dimension anymore, so directly use the `q_lens` and `k_lens` values
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k

    def switch_status(self):
        self.infer_conditional = not self.infer_conditional

    # only in Wan2.1 TeaCaching
    def set_attributes_by_task_and_model(self):
        if self.config.task == "i2v":
            if self.use_ret_steps:
                if self.config.target_width == 480 or self.config.target_height == 480:
                    self.coefficients = [
                        2.57151496e05,
                        -3.54229917e04,
                        1.40286849e03,
                        -1.35890334e01,
                        1.32517977e-01,
                    ]
                if self.config.target_width == 720 or self.config.target_height == 720:
                    self.coefficients = [
                        8.10705460e03,
                        2.13393892e03,
                        -3.72934672e02,
                        1.66203073e01,
                        -4.17769401e-02,
                    ]
                self.ret_steps = 5 * 2
                self.cutoff_steps = self.config.infer_steps * 2
            else:
                if self.config.target_width == 480 or self.config.target_height == 480:
                    self.coefficients = [
                        -3.02331670e02,
                        2.23948934e02,
                        -5.25463970e01,
                        5.87348440e00,
                        -2.01973289e-01,
                    ]
                if self.config.target_width == 720 or self.config.target_height == 720:
                    self.coefficients = [
                        -114.36346466,
                        65.26524496,
                        -18.82220707,
                        4.91518089,
                        -0.23412683,
                    ]
                self.ret_steps = 1 * 2
                self.cutoff_steps = self.config.infer_steps * 2 - 2

        elif self.config.task == "t2v":
            if self.use_ret_steps:
                if "1.3B" in self.config.model_path:
                    self.coefficients = [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02]
                if "14B" in self.config.model_path:
                    self.coefficients = [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01]
                self.ret_steps = 5 * 2
                self.cutoff_steps = self.config.infer_steps * 2
            else:
                if "1.3B" in self.config.model_path:
                    self.coefficients = [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01]
                if "14B" in self.config.model_path:
                    self.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
                self.ret_steps = 1 * 2
                self.cutoff_steps = self.config.infer_steps * 2 - 2

    # calculate should_calc
    def calculate_should_calc(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        # 1. timestep embedding
        modulated_inp = embed0 if self.use_ret_steps else embed

        # 2. L1 calculate
        should_calc = False
        if self.infer_conditional:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp - self.previous_e0_even).abs().mean() / self.previous_e0_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else:
            if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
                should_calc = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                rescale_func = np.poly1d(self.coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp - self.previous_e0_odd).abs().mean() / self.previous_e0_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.teacache_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()

        # 3. return the judgement
        return should_calc

    # 1. get taylor step_diff when there is two caching_records in scheduler
    def get_taylor_step_diff(self):
        step_diff = 0
        if self.infer_conditional:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step
        else:
            current_step = self.scheduler.step_index
            last_calc_step = current_step - 1
            while last_calc_step >= 0 and not self.scheduler.caching_records_2[last_calc_step]:
                last_calc_step -= 1
            step_diff = current_step - last_calc_step

        return step_diff


class WanTransformerInfer(BaseWanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, embed, grid_sizes, x, embed0, seq_lens, freqs, context):
        # 1. read list
        index = self.scheduler.step_index
        caching_records = self.scheduler.caching_records

        # 2. judge to fully calculate or use cache
        if caching_records[index]:
            return self.infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        else:
            return self.infer_using_cache(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

    def infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        if not self.config.cpu_offload:
            return self._infer_calculating(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
        else:
            if self.offload_granularity == "block":
                return self._infer_calculating_block_offload(weights, grid_sizes, x, embed0, seq_lens, freqs, context)
            else:
                return self._infer_calculating_phase_offload(weights, grid_sizes, x, embed0, seq_lens, freqs, context)

    def _infer_calculating(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            if embed0.dim() == 3:
                modulation = weights.blocks[block_idx].modulation.tensor.unsqueeze(2)
                embed0 = (modulation + embed0).chunk(6, dim=1)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
            elif embed0.dim() == 2:
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.blocks[block_idx].modulation.tensor + embed0).chunk(6, dim=1)

            y_out = super().infer_block_1(weights.blocks[block_idx].compute_phases[0], grid_sizes, x, embed0, seq_lens, freqs, context, shift_msa, scale_msa)
            attn_out = super().infer_block_2(weights.blocks[block_idx].compute_phases[1], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
            y_out = super().infer_block_3(weights.blocks[block_idx].compute_phases[2], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
            x = super().infer_block_4(None, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)
        return x

    def _infer_calculating_block_offload(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = weights.blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                if embed0.dim() == 3:
                    modulation = self.weights_stream_mgr.active_weights[0].modulation.tensor.unsqueeze(2)
                    embed0 = (modulation + embed0).chunk(6, dim=1)
                    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
                elif embed0.dim() == 2:
                    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (self.weights_stream_mgr.active_weights[0].modulation.tensor + embed0).chunk(6, dim=1)

                y_out = super().infer_block_1(self.weights_stream_mgr.active_weights[0].compute_phases[0], grid_sizes, x, embed0, seq_lens, freqs, context, shift_msa, scale_msa)
                attn_out = super().infer_block_2(self.weights_stream_mgr.active_weights[0].compute_phases[1], grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
                y_out = super().infer_block_3(self.weights_stream_mgr.active_weights[0].compute_phases[2], grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
                x = super().infer_block_4(None, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)
            self.weights_stream_mgr.swap_weights()

        return x

    def _infer_calculating_phase_offload(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            weights.blocks[block_idx].modulation.to_cuda()

            if embed0.dim() == 3:
                modulation = weights.blocks[block_idx].modulation.tensor.unsqueeze(2)
                current_embed0 = (modulation + embed0).chunk(6, dim=1)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in current_embed0]
            elif embed0.dim() == 2:
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.blocks[block_idx].modulation.tensor + embed0).chunk(6, dim=1)

            for phase_idx in range(3):
                if block_idx == 0 and phase_idx == 0:
                    phase = weights.blocks[block_idx].compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    cur_phase_idx, cur_phase = self.weights_stream_mgr.active_weights[0]
                    if cur_phase_idx == 0:
                        y_out = super().infer_block_1(cur_phase, grid_sizes, x, embed0, seq_lens, freqs, context, shift_msa, scale_msa)
                    elif cur_phase_idx == 1:
                        attn_out = super().infer_block_2(cur_phase, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, gate_msa)
                    elif cur_phase_idx == 2:
                        y_out = super().infer_block_3(cur_phase, grid_sizes, x, embed0, seq_lens, freqs, context, attn_out, c_shift_msa, c_scale_msa)
                        x = super().infer_block_4(cur_phase, grid_sizes, x, embed0, seq_lens, freqs, context, y_out, c_gate_msa)

                is_last_phase = block_idx == weights.blocks_num - 1 and phase_idx == 2
                if not is_last_phase:
                    next_block_idx = block_idx + 1 if cur_phase_idx == 2 else block_idx
                    next_phase_idx = (cur_phase_idx + 1) % 3
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            weights.blocks[block_idx].modulation.to_cpu()

        torch.cuda.empty_cache()

        return x

    def infer_using_cache(self, weights, grid_sizes, x, embed0, seq_lens, freqs, context):
        pass
