import torch
import math
from einops import rearrange
from lightx2v.attentions import attention


####################################################################################################
class HunyuanPreInfer:
    ############################## 1. 初始化 ##############################
    def __init__(self, config):
        self.config = config
        self.heads_num = 24

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    ############################## 2. 推理 ##############################
    # 2.1 推理
    # 2.1.1 weights是一个PreWeight权重实例，具有weight_list属性，是一系列算子，每个算子有weight属性和bias属性
    # 2.1.2 inputs是从顶层HunyuanRunner的run_pipeline(), run()传入，是一个字典
    # 2.1.2 inputs["image_encoder_output"]["img"], ["img_latents"]
    # 2.1.2 inputs["text_encoder_output"]["text_encoder_1_text_states"], ["text_encoder_1_attention_mask"], ["text_encoder_2_text_states"], ["text_encoder_2_attention_mask"]
    def infer(self, weights, inputs):
        # 2.1 获取scheduler的多维张量latents
        x = self.scheduler.latents

        # 2.2 根据推理步数下标获取scheduler的timesteps属性中的元素，是一个1*1的张量
        # 2.2 timesteps是sigmas去掉一个后放大1000倍，元素个数和推理步数一致
        t = self.scheduler.timesteps[self.scheduler.step_index]

        # 2.3 获取scheduler的freqs_cos, freqs_sin, 均为多维张量
        # 2.3 获取scheduler的guidance，为1*1张量
        freqs_cos = self.scheduler.freqs_cos
        freqs_sin = self.scheduler.freqs_sin
        guidance = self.scheduler.guidance

        # 2.4 获取文本的states_1, mask_1, states_2
        text_states = inputs["text_encoder_output"]["text_encoder_1_text_states"]
        text_mask = inputs["text_encoder_output"]["text_encoder_1_attention_mask"]
        text_states_2 = inputs["text_encoder_output"]["text_encoder_2_text_states"]

        # 2.5 图片任务
        if self.config["task"] == "i2v":
            # 2.5.1 得到一个1*1的0张量
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.infer_time_in(weights, token_replace_t)
            th = x.shape[-2] // 2
            tw = x.shape[-1] // 2
            frist_frame_token_num = th * tw

        # 2.6 一开始用随机的latents转化成第一次的视频输出
        # 2.6.1 具体就是对latents做一个3D卷积，得到一个五维张量，并且展开成三维
        img_out = self.infer_img_in(weights, x)

        time_out = self.infer_time_in(weights, t)

        infer_vector_out = self.infer_vector_in(weights, text_states_2)

        guidance_out = self.infer_guidance_in(weights, guidance)

        vec = time_out + infer_vector_out + guidance_out

        infer_text_out = self.infer_text_in(weights, text_states, text_mask, t)

        if self.config["task"] == "i2v":
            token_replace_vec = token_replace_vec + infer_vector_out

        txt_seq_len = infer_text_out.shape[0]
        img_seq_len = img_out.shape[1]
        batch_size = text_mask.shape[0]
        text_len = text_mask.sum(dim=1)
        max_len = text_mask.shape[1] + img_seq_len

        cu_seqlens_qkv = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            s = text_len[i] + img_seq_len
            s1 = i * max_len + s
            s2 = (i + 1) * max_len
            cu_seqlens_qkv[2 * i + 1] = s1
            cu_seqlens_qkv[2 * i + 2] = s2

        max_seqlen_qkv = img_seq_len + txt_seq_len
        if self.config["task"] == "i2v":
            return img_out[0], infer_text_out, vec, cu_seqlens_qkv, max_seqlen_qkv, (freqs_cos, freqs_sin), token_replace_vec, frist_frame_token_num
        return img_out[0], infer_text_out, vec, cu_seqlens_qkv, max_seqlen_qkv, (freqs_cos, freqs_sin)

    def infer_time_in(self, weights, t):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128).to(device=t.device)
        args = t.unsqueeze(0).unsqueeze(0).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype=torch.bfloat16)
        out = weights.time_in_mlp_0.apply(embedding)
        out = torch.nn.functional.silu(out)
        out = weights.time_in_mlp_2.apply(out)
        return out

    def infer_img_in(self, weights, x):
        out = weights.img_in_proj.apply(x)
        out = out.flatten(2).transpose(1, 2)
        return out

    def infer_text_in(self, weights, text_states, text_mask, t):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128).to(device=t.device)
        args = t.unsqueeze(0).unsqueeze(0).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype=torch.bfloat16)
        out = weights.txt_in_t_embedder_mlp_0.apply(embedding)
        out = torch.nn.functional.silu(out)
        timestep_aware_representations = weights.txt_in_t_embedder_mlp_2.apply(out)

        mask_float = text_mask.float().unsqueeze(-1).to(torch.bfloat16)  # [b, s1, 1]
        context_aware_representations = (text_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
        context_aware_representations = context_aware_representations

        out = weights.txt_in_c_embedder_linear_1.apply(context_aware_representations)
        out = torch.nn.functional.silu(out)
        context_aware_representations = weights.txt_in_c_embedder_linear_2.apply(out)
        c = timestep_aware_representations + context_aware_representations

        txt_in_input_embed = weights.txt_in_input_embedder.apply(text_states[0])

        batch_size = text_mask.shape[0]
        seq_len = text_mask.shape[1]
        self_attn_mask_1 = text_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
        self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
        self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
        self_attn_mask[:, :, :, 0] = True

        cx = torch.nn.functional.silu(c)
        cx = weights.txt_in_individual_token_refiner_blocks_0_adaLN_modulation_1.apply(cx)
        gate_msa, gate_mlp = cx.chunk(2, dim=1)
        normx = weights.txt_in_individual_token_refiner_blocks_0_norm1.apply(txt_in_input_embed)
        qkv = weights.txt_in_individual_token_refiner_blocks_0_self_attn_qkv.apply(normx)
        q, k, v = rearrange(qkv.unsqueeze(0), "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        attn = attention(attention_type="torch_sdpa", q=q, k=k, v=v, attn_mask=self_attn_mask)[0]
        out = weights.txt_in_individual_token_refiner_blocks_0_self_attn_proj.apply(attn)
        out_1 = txt_in_input_embed + out * gate_msa
        out = weights.txt_in_individual_token_refiner_blocks_0_norm2.apply(out_1)
        # mlp
        out = weights.txt_in_individual_token_refiner_blocks_0_mlp_fc1.apply(out)
        out = torch.nn.functional.silu(out)
        out = weights.txt_in_individual_token_refiner_blocks_0_mlp_fc2.apply(out)
        txt_in_input_embed = out_1 + out * gate_mlp

        cx = torch.nn.functional.silu(c)
        cx = weights.txt_in_individual_token_refiner_blocks_1_adaLN_modulation_1.apply(cx)
        gate_msa, gate_mlp = cx.chunk(2, dim=1)

        normx = weights.txt_in_individual_token_refiner_blocks_1_norm1.apply(txt_in_input_embed)
        qkv = weights.txt_in_individual_token_refiner_blocks_1_self_attn_qkv.apply(normx)

        q, k, v = rearrange(qkv.unsqueeze(0), "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        attn = attention(attention_type="torch_sdpa", q=q, k=k, v=v, attn_mask=self_attn_mask)[0]
        out = weights.txt_in_individual_token_refiner_blocks_1_self_attn_proj.apply(attn)
        out_1 = txt_in_input_embed + out * gate_msa

        out = weights.txt_in_individual_token_refiner_blocks_1_norm2.apply(out_1)
        # mlp
        out = weights.txt_in_individual_token_refiner_blocks_1_mlp_fc1.apply(out)
        out = torch.nn.functional.silu(out)
        out = weights.txt_in_individual_token_refiner_blocks_1_mlp_fc2.apply(out)

        out = out_1 + out * gate_mlp
        return out

    def infer_vector_in(self, weights, text_states_2):
        out = weights.vector_in_in_layer.apply(text_states_2)
        out = torch.nn.functional.silu(out)
        out = weights.vector_in_out_layer.apply(out)
        return out

    def infer_guidance_in(self, weights, guidance):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128).to(device=guidance.device)
        args = guidance.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype=torch.bfloat16)
        out = weights.guidance_in_mlp_0.apply(embedding)
        out = torch.nn.functional.silu(out)
        out = weights.guidance_in_mlp_2.apply(out)
        return out
