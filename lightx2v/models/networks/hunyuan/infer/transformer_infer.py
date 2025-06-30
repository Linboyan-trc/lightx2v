import torch
from einops import rearrange
from lightx2v.attentions import attention
from .utils_bf16 import apply_rotary_emb
from lightx2v.common.offload.manager import WeightStreamManager
from lightx2v.utils.envs import *


############################################################################################################################################
# 1. TransformerInfer
# 1.1 不使用feature_caching
class HunyuanTransformerInfer:
    ################################################## 1. 初始化 ##################################################
    # 1.1 初始化
    def __init__(self, config):
        # 1.1 属性: config, attention_type
        self.config = config
        self.attention_type = config.get("attention_type", "flash_attn2")

        # 1.2 属性: double_blocks_num, single_blocks_num, heads_num
        # 1.2.1 属性: hidden_size, mlp_hidden_dim
        self.double_blocks_num = 20
        self.single_blocks_num = 40
        self.heads_num = 24
        self.hidden_size = 3072
        self.mlp_hidden_dim = 12288

        # 1.3 多卡并行
        self.parallel_attention = None

        # 1.4 使用CPU推理
        if self.config["cpu_offload"]:
            self.double_weights_stream_mgr = WeightStreamManager()
            self.single_weights_stream_mgr = WeightStreamManager()
            self.infer_func = self._infer_with_offload
        
        # 1.5 不使用CPU推理
        else:
            self.infer_func = self._infer_without_offload

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    #############################################################################################################
    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        return self.infer_func(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

    def _infer_with_offload(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        txt_seq_len = txt.shape[0]
        img_seq_len = img.shape[0]

        for double_block_idx in range(self.double_blocks_num):
            # 1. 先到第一个block权重，移动到cuda上
            if double_block_idx == 0:
                self.double_weights_stream_mgr.active_weights[0] = weights.double_blocks_weights[0]
                self.double_weights_stream_mgr.active_weights[0].to_cuda()

            # 2. 写入高优先队列，异步推理
            with torch.cuda.stream(self.double_weights_stream_mgr.compute_stream):
                img, txt = self.infer_double_block(self.double_weights_stream_mgr.active_weights[0], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

            # 3. 不是最后一个，写入低优先队列
            # 3.1 对于第0个块，拿到第1个块，移动到cuda，然后给[1]；叫做预先拿
            # 3.2 对于第1个块，老块0在[1]移回cpu，然后第2个块拿到cuda，给[1]
            if double_block_idx < self.double_blocks_num - 1:
                self.double_weights_stream_mgr.prefetch_weights(double_block_idx + 1, weights.double_blocks_weights)
            
            # 4. 让新拿的[1]给[0]，下一轮用；[1]用于存老的
            self.double_weights_stream_mgr.swap_weights()

        x = torch.cat((img, txt), 0)

        # 5. 移回cpu然后删掉，清除cuda缓存
        img = img.cpu()
        txt = txt.cpu()
        del img, txt
        torch.cuda.empty_cache()

        # 6. 一样的流程
        for single_block_idx in range(self.single_blocks_num):
            if single_block_idx == 0:
                self.single_weights_stream_mgr.active_weights[0] = weights.single_blocks_weights[0]
                self.single_weights_stream_mgr.active_weights[0].to_cuda()
            with torch.cuda.stream(self.single_weights_stream_mgr.compute_stream):
                x = self.infer_single_block(self.single_weights_stream_mgr.active_weights[0], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            if single_block_idx < self.single_blocks_num - 1:
                self.single_weights_stream_mgr.prefetch_weights(single_block_idx + 1, weights.single_blocks_weights)
            self.single_weights_stream_mgr.swap_weights()
            torch.cuda.empty_cache()

        img = x[:img_seq_len, ...]
        return img, vec

    def _infer_without_offload(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        # 1. latents: 得到32400
        img_seq_len = img.shape[0]

        # 2. prompt: 得到256
        txt_seq_len = txt.shape[0]

        # 20个
        # 分别处理latents和prompt
        for i in range(self.double_blocks_num):
            img, txt = self.infer_double_block(weights.double_blocks_weights[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
        
        # 3. 噪声卷积和文本特征拼接，形状为[32656, 3072]
        x = torch.cat((img, txt), 0)

        # 40个
        # 合并
        for i in range(self.single_blocks_num):
            x = self.infer_single_block(weights.single_blocks_weights[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

        img = x[:img_seq_len, ...]

        return img, vec

    def infer_double_block(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num):
        # 1. vec修改，让正值几乎不变，让负值接近0
        # 1.1 vec形状为[1, 3072]，来自timesteps[0]的时间步嵌入，prompt经过clip的编码，guidance的嵌入
        # 1.2 silu对每个元素更新
        vec_silu = torch.nn.functional.silu(vec)

        # 2. 传入的weights是一个double_blocks.0.xxx.weight, .bias之类的
        # 2. 每个double_blocks有7个图片属性，img_mod, img_attn_qkv, img_attn_q_norm, img_attn_k_norm, img_mlp_fc1, img_mlp_fc2, img_attn_proj
        # 2. 每个double_blocks有7个文本属性，mod, attn_qkv, attn_q_norm, attn_k_norm, mlp_fc1, mlp_fc2, attn_proj

        # 3. 这里是根据开源的Hunyuan的模型结构来决定的Transformer运算流程，并不是一个规范的标程

        # 4. 先进行img: mod调制，就是对融合后的特征向量做线性变换，变换的结果是形状从[1, 3072]变为[1, 18432]
        # 4.1 然后切成6部分，分别是1_shift, scale, gate, 2_shift, scale, gate，每个形状是[1, 3072]
        img_mod_out = weights.img_mod.apply(vec_silu)
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod_out.chunk(6, dim=-1)

        if token_replace_vec is not None:
            token_replace_vec_silu = torch.nn.functional.silu(token_replace_vec)
            token_replace_vec_img_mod_out = weights.img_mod.apply(token_replace_vec_silu)
            (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = token_replace_vec_img_mod_out.chunk(6, dim=-1)
        else:
            (tr_img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate) = None, None, None, None, None, None

        # 4. 再进行txt: mod调制，就是对融合后的特征向量做线性变换，变换的结果是形状从[1, 3072]变为[1, 18432]
        # 4.1 然后切成6部分，分别是1_shift, scale, gate, 2_shift, scale, gate，每个形状是[1, 3072]
        # 4.2 两者都是对融合后的特征向量进行调制，只不过使用的权重矩阵不一样
        txt_mod_out = weights.txt_mod.apply(vec_silu)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod_out.chunk(6, dim=-1)

        # 5. 再进行img: img_attn_qkv, img_attn_q_norm, img_attn_k_norm, 自注意力计算
        # 5. 再进行txt: attn_qkv, attn_q_norm, attn_k_norm, 自注意力计算
        img_q, img_k, img_v = self.infer_double_block_img_pre_atten(weights, img, img_mod1_scale, img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_shift, frist_frame_token_num, freqs_cis)
        txt_q, txt_k, txt_v = self.infer_double_block_txt_pre_atten(weights, txt, txt_mod1_scale, txt_mod1_shift)

        # 6. 把img和txt的自注意力输出按照第一维拼接
        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)

        if not self.parallel_attention:
            attn = attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=max_seqlen_qkv,
                max_seqlen_kv=max_seqlen_qkv,
            )
        else:
            # world_size = dist.get_world_size()
            attn = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=img_q.shape[0],
                cu_seqlens_qkv=cu_seqlens_qkv,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )

        # 7. 注意力输出attn形状为[32656, 3072]
        # 7.1 attn[:32400]就是取前32400行，attn[32400:]就是取前32400之后的行
        img_attn, txt_attn = attn[: img.shape[0]], attn[img.shape[0] :]

        img = self.infer_double_block_img_post_atten(
            weights, img, img_attn, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, frist_frame_token_num
        )
        txt = self.infer_double_block_txt_post_atten(weights, txt, txt_attn, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate)
        return img, txt

    def infer_double_block_img_pre_atten(self, weights, img, img_mod1_scale, img_mod1_shift, tr_img_mod1_scale, tr_img_mod1_shift, frist_frame_token_num, freqs_cis):
        # 1. 对噪声卷积的每一行归一化，得到[32400, 3072]
        img_modulated = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        if tr_img_mod1_scale is not None:
            x_zero = img_modulated[:frist_frame_token_num] * (1 + tr_img_mod1_scale) + tr_img_mod1_shift
            x_orig = img_modulated[frist_frame_token_num:] * (1 + img_mod1_scale) + img_mod1_shift
            img_modulated = torch.concat((x_zero, x_orig), dim=0)
        
        # 2. 对归一化后的噪声卷积做一个矩阵点积
        # 2.1 就是对[1, 3072]的img_mod1_scale复制扩展成[32400, 3072]，然后每个元素 + 1， 再和[32400, 3072]的噪声卷积点乘， 最后加上img_mod1_shift
        # 2.2 整体来说就是用img_mod1_scale, shift对噪声输入做一个线性变换
        else:
            img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift
        
        # 3. 对调制后的噪声卷积进行线性变换，得到注意力img_qkv，形状为[32400, 9216]
        img_qkv = weights.img_attn_qkv.apply(img_modulated)

        # 4. 将img_qkv切成三份, [32400, 3072], [32400, 3072], [32400, 3072]
        # 4.1 每一份再切成[3072/24, 3072/24, ..., ]，也就是[32400, "128,128,128,..."]，也就是[32400, 24, 128], 把每一行的3072拆成24份，每份128个元素
        # 4.2 比如[32400, 9216]的第一行0,1,2, ..., 36
        # 4.3 被拆成[0~11], [12~23], [24~35]
        # 4.3 然后各自再用head_nums分一下
        # 2.1 原来的形状是[L, K * H * D]
        # 2.2 现在对最后一维划分成K份
        # 2.3 然后每一份重新组成形状[L, H * D]，进一步组成形状[L, H, D]，也就是token数量，head数量，每一个head含有的元素数量
        img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        # 2.4 均方根一下，形状不变，还是[32400, 24, 128]
        img_q = weights.img_attn_q_norm.apply(img_q)
        img_k = weights.img_attn_k_norm.apply(img_k)

        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)
        return img_q, img_k, img_v

    def infer_double_block_txt_pre_atten(self, weights, txt, txt_mod1_scale, txt_mod1_shift):
        txt_modulated = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift
        txt_qkv = weights.txt_attn_qkv.apply(txt_modulated)

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        txt_q = weights.txt_attn_q_norm.apply(txt_q)
        txt_k = weights.txt_attn_k_norm.apply(txt_k)
        return txt_q, txt_k, txt_v

    def infer_double_block_img_post_atten(
        self, weights, img, img_attn, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate, tr_img_mod1_gate, tr_img_mod2_shift, tr_img_mod2_scale, tr_img_mod2_gate, frist_frame_token_num
    ):
        out = weights.img_attn_proj.apply(img_attn)
        if tr_img_mod1_gate is not None:
            x_zero = out[:frist_frame_token_num] * tr_img_mod1_gate
            x_orig = out[frist_frame_token_num:] * img_mod1_gate
            out = torch.concat((x_zero, x_orig), dim=0)
        else:
            out = out * img_mod1_gate
        img = img + out

        out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        if tr_img_mod1_gate is not None:
            x_zero = out[:frist_frame_token_num] * (1 + tr_img_mod2_scale) + tr_img_mod2_shift
            x_orig = out[frist_frame_token_num:] * (1 + img_mod2_scale) + img_mod2_shift
            out = torch.concat((x_zero, x_orig), dim=0)
        else:
            out = out * (1 + img_mod2_scale) + img_mod2_shift
        out = weights.img_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.img_mlp_fc2.apply(out)
        out = out * img_mod2_gate
        img = img + out
        return img

    def infer_double_block_txt_post_atten(
        self,
        weights,
        txt,
        txt_attn,
        txt_mod1_gate,
        txt_mod2_shift,
        txt_mod2_scale,
        txt_mod2_gate,
    ):
        out = weights.txt_attn_proj.apply(txt_attn)
        out = out * txt_mod1_gate
        txt = txt + out

        out = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        out = out * (1 + txt_mod2_scale) + txt_mod2_shift
        out = weights.txt_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.txt_mlp_fc2.apply(out)
        out = out * txt_mod2_gate
        txt = txt + out
        return txt

    def infer_single_block(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        if token_replace_vec is not None:
            token_replace_vec_out = torch.nn.functional.silu(token_replace_vec)
            token_replace_vec_out = weights.modulation.apply(token_replace_vec_out)
            tr_mod_shift, tr_mod_scale, tr_mod_gate = token_replace_vec_out.chunk(3, dim=-1)

        out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        if token_replace_vec is not None:
            x_zero = out[:frist_frame_token_num] * (1 + tr_mod_scale) + tr_mod_shift
            x_orig = out[frist_frame_token_num:] * (1 + mod_scale) + mod_shift
            x_mod = torch.concat((x_zero, x_orig), dim=0)
        else:
            x_mod = out * (1 + mod_scale) + mod_shift

        x_mod = weights.linear1.apply(x_mod)

        qkv, mlp = torch.split(x_mod, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "L (K H D) -> K L H D", K=3, H=self.heads_num)

        q = weights.q_norm.apply(q)
        k = weights.k_norm.apply(k)

        img_q, txt_q = q[:-txt_seq_len, :, :], q[-txt_seq_len:, :, :]
        img_k, txt_k = k[:-txt_seq_len, :, :], k[-txt_seq_len:, :, :]
        img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis)

        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)

        if not self.parallel_attention:
            attn = attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=max_seqlen_qkv,
                max_seqlen_kv=max_seqlen_qkv,
            )
        else:
            attn = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=img_q.shape[0],
                cu_seqlens_qkv=cu_seqlens_qkv,
                # cu_seqlens_qkv=cu_seqlens_qkv,
                # max_seqlen_qkv=max_seqlen_qkv,
            )

        out = torch.nn.functional.gelu(mlp, approximate="tanh")
        out = torch.cat((attn, out), 1)
        out = weights.linear2.apply(out)

        if token_replace_vec is not None:
            x_zero = out[:frist_frame_token_num] * tr_mod_gate
            x_orig = out[frist_frame_token_num:] * mod_gate
            out = torch.concat((x_zero, x_orig), dim=0)
        else:
            out = out * mod_gate
        x = x + out
        return x
