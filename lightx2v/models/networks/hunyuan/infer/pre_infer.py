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
        # 2.2 一开始用随机的latents转化成第一次的视频输出
        # 2.2.1 具体就是对latents做一个3D卷积，得到一个五维张量，并且展开成三维
        # 2.3 img_seq_len是视频卷积展开后的第二维，为32400
        x = self.scheduler.latents
        img_out = self.infer_img_in(weights, x)
        img_seq_len = img_out.shape[1]

        # 2.4 根据推理步数下标获取scheduler的timesteps属性中的元素，是一个1*1的张量
        # 2.4 timesteps是sigmas去掉一个后放大1000倍，元素个数和推理步数一致
        # 2.4.1 将timesteps[0]转化为时间步嵌入，得到一个[1,3072]的二维张量
        t = self.scheduler.timesteps[self.scheduler.step_index]
        time_out = self.infer_time_in(weights, t)        

        # 2.5 图片任务
        if self.config["task"] == "i2v":
            # 2.5.1 得到一个1*1的0张量
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.infer_time_in(weights, token_replace_t)
            th = x.shape[-2] // 2
            tw = x.shape[-1] // 2
            frist_frame_token_num = th * tw

        # 2.6.2 infer_vector_out，来自clip对prompt的编码，形状[1, 768]
        # 2.6.2 得到infer_vector_out形状为[1, 3072]
        text_states_2 = inputs["text_encoder_output"]["text_encoder_2_text_states"]
        infer_vector_out = self.infer_vector_in(weights, text_states_2)

        # 2.7.3 guidance为[1,1]的二维张量
        # 2.7.3 其实和时间步嵌入的计算一样，时间步计算用timesteps[0]去对freqs扩倍再求正弦、余弦
        # 2.7.3 而这里用scheduler.guidance去对freqs扩倍再求正弦、余弦
        # 2.7.3 timesteps随着推理的进行减小，而guidance保持不变
        guidance = self.scheduler.guidance
        guidance_out = self.infer_guidance_in(weights, guidance)

        # 2.8.4 vec是一个[1, 3072]的二维张量
        vec = time_out + infer_vector_out + guidance_out

        if self.config["task"] == "i2v":
            token_replace_vec = token_replace_vec + infer_vector_out

        # 2.9 来自llama对于文本的编码[1,256,4096], 掩码[1,256]
        # 2.9.1 经过一系列权重计算之后infer_text_out形状为[256, 3072]
        # 2.10 txt_seq_len其实就是prompt的token长度，为256
        text_states = inputs["text_encoder_output"]["text_encoder_1_text_states"]
        text_mask = inputs["text_encoder_output"]["text_encoder_1_attention_mask"]
        infer_text_out = self.infer_text_in(weights, text_states, text_mask, t)
        txt_seq_len = infer_text_out.shape[0]

        # 2.11 batch_size为llama掩码的第一维，llama掩码形状为[1, 256]，因为只处理一句话，只有一句prompt
        # 2.11 所以batch_size = 1
        batch_size = text_mask.shape[0]

        # 2.12 统计llama掩码第二维所有元素的和，也就是统计1的数量
        # 2.12 text_len等于256个token中有效token的数量
        text_len = text_mask.sum(dim=1)

        # 2.13 最大长度就是256 + 视频卷积展开后的第二维32400
        max_len = text_mask.shape[1] + img_seq_len

        # 2.14 所以cu_seqlens_qkv形状为[3]
        cu_seqlens_qkv = torch.zeros(
            [2 * batch_size + 1], 
            dtype=torch.int32, 
            device="cuda"
        )

        # 2.15 s  = 有效token的数量 + 视频卷积展开后的第二维32400
        # 2.15 s1 = 有效token的数量 + 视频卷积展开后的第二维32400
        # 2.15 s2 = 256 + 视频卷积展开后的第二维32400
        # 2.15 cu_seqlens_qkv[1] = 有效 + 32400
        # 2.15 cu_seqlens_qkv[2] = 256 + 32400
        for i in range(batch_size):
            s = text_len[i] + img_seq_len
            s1 = i * max_len + s
            s2 = (i + 1) * max_len
            cu_seqlens_qkv[2 * i + 1] = s1
            cu_seqlens_qkv[2 * i + 2] = s2

        # 2.16 cu_seqlens_qkv = 256 + 32400
        max_seqlen_qkv = txt_seq_len + img_seq_len

        # 2.6 获取scheduler的freqs_cos, freqs_sin, 均为多维张量
        # 2.6 获取scheduler的guidance，为1*1张量
        freqs_cos = self.scheduler.freqs_cos
        freqs_sin = self.scheduler.freqs_sin

        if self.config["task"] == "i2v":
            return img_out[0], infer_text_out, vec, cu_seqlens_qkv, max_seqlen_qkv, (freqs_cos, freqs_sin), token_replace_vec, frist_frame_token_num
        return img_out[0], infer_text_out, vec, cu_seqlens_qkv, max_seqlen_qkv, (freqs_cos, freqs_sin)

    def infer_img_in(self, weights, x):
        out = weights.img_in_proj.apply(x)
        out = out.flatten(2).transpose(1, 2)
        return out

    # 2.7 来自llama对于文本的编码[1,256,4096], 掩码[1,256]，clip的编码[1,768]
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

    # 2.8.1 将timesteps[0]转化为时间步嵌入
    def infer_time_in(self, weights, t):
        # 1. 得到一个[128]的一维张量，元素大小从1到0，曲线先快速下降，再慢速下降
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128
        ).to(device=t.device)

        # 2. t是一个0维标量，tensor(1000)
        # 2.1 t.unsqueeze(0)可以把0维标量扩展为一维张量[1]，tensor([1000])
        # 2.2 继续.unsqueeze(0)可以把一维张量扩展为二维张量[1,1]，tensor([[1000]])
        # 2.3 freqs[None]就是把一维张量[128]扩展为二维张量[1，128]
        # 2.4 所以t.unsqueeze(0).unsqueeze(0) * freqs[None]最终就是一个[1,1] * [1,128]的矩阵乘法，具体就是1000和freqs中每个元素相乘
        # 2.4 freqs中每个元素放大1000倍，最后得到一个[1,128]的二维张量args
        # 2.5 args是一个[1,128]的二维张量，直接由freqs扩倍而来
        args = t.unsqueeze(0).unsqueeze(0).float() * freqs[None]

        # 3. 对[1,128]args中每个元素求余弦，正弦，然后拼接成[1,256]的embedding
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)], 
            dim=-1
        ).to(dtype=torch.bfloat16)

        # 4. 对embedding计算一下，得到out形状为[1, 3072], 3072取决于time_in_mlp.0.weight第二维的长度
        out = weights.time_in_mlp_0.apply(embedding)

        # 5. 对每个元素激活一下
        # 5.1 silu的作用是使大于0的元素基本保持原值，使小于0的元素变成一个e-20~30的接近0的负数
        # 5.2 而relu作用是使大于0的元素为原值，使小于0的元素变成0，比silu更粗暴，导致死神经元风险高，因为梯度消失了，而silu还是保持了部分梯度
        out = torch.nn.functional.silu(out)

        # 6. 再计算一下，得到out形状为[1, 3072], 3072取决于time_in_mlp.2.weight第二维的长度
        out = weights.time_in_mlp_2.apply(out)

        # 7. 返回一个[1, 3072]的二维张量
        return out

    # 2.8.2 infer_vector_out，来自clip对prompt的编码，形状[1, 768]
    def infer_vector_in(self, weights, text_states_2):
        # 1. addmm或mm后，out形状为[1, 3072]
        out = weights.vector_in_in_layer.apply(text_states_2)

        # 2. 负值趋于0一下
        out = torch.nn.functional.silu(out)

        # 3. addmm或mm后，out形状为[1, 3072]
        out = weights.vector_in_out_layer.apply(out)

        # 4. 返回二维张量out，形状为[1, 3072]
        return out

    # 2.8.3 guidance为[1,1]的二维张量
    def infer_guidance_in(self, weights, guidance):
        # 1. 制作一个[128]的一维张量，元素大小从1到0，曲线先快速下降，再慢速下降
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=128, dtype=torch.float32) / 128
        ).to(device=guidance.device)

        # 2. guidance是一个[1,1]的二维张量，直接为freqs[None]形状为[1,128]矩阵相乘
        # 2. 得到args形状为[1,128]
        args = guidance.float() * freqs[None]

        # 3. 嵌入一下，得到一个[1,256]的二维张量
        embedding = torch.cat(
            [torch.cos(args), torch.sin(args)], 
            dim=-1
        ).to(dtype=torch.bfloat16)

        # 4. 返回一个out, 形状为[1, 3072]
        out = weights.guidance_in_mlp_0.apply(embedding)
        out = torch.nn.functional.silu(out)
        out = weights.guidance_in_mlp_2.apply(out)
        return out
