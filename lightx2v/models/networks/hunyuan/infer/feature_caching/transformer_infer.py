import torch
import numpy as np
from einops import rearrange
from lightx2v.attentions import attention
from .utils import taylor_cache_init, derivative_approximation, taylor_formula
from ..utils_bf16 import apply_rotary_emb
from ..transformer_infer import HunyuanTransformerInfer


############################################################################################################################################
# 1. TransformerInferTeaCaching
# 1.1 使用feature_caching: tea
class HunyuanTransformerInferTeaCaching(HunyuanTransformerInfer):
    ################################################## 1. 初始化 ##################################################
    # 1.1 初始化
    def __init__(self, config):
        super().__init__(config)
    #############################################################################################################

    def infer(
        self,
        weights,
        img,
        txt,
        vec,
        cu_seqlens_qkv,
        max_seqlen_qkv,
        freqs_cis,
        token_replace_vec=None,
        frist_frame_token_num=None,
    ):
        # 1. 复制img[32400, 3072], 复制vec[1, 3072]
        inp = img.clone()
        vec_ = vec.clone()

        # 2. 对句子进行线性偏移的偏移量，缩放因子
        # 2. 这里使用20个double_blocks_weights中的[0]，而不是[1]~[19]
        # 2. 因为在transformer的20层double_blocks_weights计算中，第一个double_blocks_weights[0]具有决定性作用，[1]~[19]是在[0]的基础上对输入噪声进行逐步调整
        weights.double_blocks_weights[0].to_cuda()
        img_mod1_shift, img_mod1_scale, _, _, _, _ = weights.double_blocks_weights[0].img_mod.apply(vec_).chunk(6, dim=-1)
        weights.double_blocks_weights[0].to_cpu_sync()

        # 3. 对img归一化一下，均值为0，方差为1
        normed_inp = torch.nn.functional.layer_norm(inp, (inp.shape[1],), None, None, 1e-6)

        # 4. 对归一化后的img线性变换一下
        modulated_inp = normed_inp * (1 + img_mod1_scale) + img_mod1_shift

        # 5. 只留下线性变换后的img，初始的img和归一化后的img删除
        del normed_inp, inp, vec_

        # 6. 第一次计算和最后一次计算
        # 6. should_calc参数用于决定是否进行普通推理
        if self.scheduler.cnt == 0 or self.scheduler.cnt == self.scheduler.num_steps - 1:
            should_calc = True
            self.scheduler.accumulated_rel_l1_distance = 0
        
        # 7. 后几次计算
        else:
            # 7.1 创建多项式函数，系数由SchedulerTea的coefficients属性给定
            # 7.1 通过rescale_func(3)就可以计算x = 3的时候多项式的值
            rescale_func = np.poly1d(self.scheduler.coefficients)

            # 7.2 计算rL1并且累计
            self.scheduler.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.scheduler.previous_modulated_input).abs().mean() / self.scheduler.previous_modulated_input.abs().mean()).cpu().item()
            )

            # 7.3 小于阈值就不重新计算
            if self.scheduler.accumulated_rel_l1_distance < self.scheduler.teacache_thresh:
                should_calc = False
            
            # 7.4 大于等于阈值就重新计算，并且重置累计的rL1
            else:
                should_calc = True
                self.scheduler.accumulated_rel_l1_distance = 0
        
        # 8. 缓存一下当前latents的计算结果
        self.scheduler.previous_modulated_input = modulated_inp
        del modulated_inp

        # 9. 小于阈值，直接加上推理后的latents和初始的latents的差值
        if not should_calc:
            img += self.scheduler.previous_residual

        # 10. 大于等于阈值，重新计算
        else:
            # 10.1 拷贝图片
            ori_img = img.clone()

            # 10.2 用普通的HunyuanTransformers推理
            img, vec = super().infer(weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)
            
            # 10.3 计算推理后的latents和初始的latents的差值，并记录
            self.scheduler.previous_residual = img - ori_img
            del ori_img

            torch.cuda.empty_cache()

        return img, vec


############################################################################################################################################
# 1. TransformerInferTaylorCaching
# 1.1 使用feature_caching: TaylorSeer
class HunyuanTransformerInferTaylorCaching(HunyuanTransformerInfer):
    ################################################## 1. 初始化 ##################################################
    # 1.1 初始化
    # 1.1.1 由于TaylorSeer算法核心是进行展开成近似的多项式，因此依赖GPU的并行计算来对多项式计算，不支持使用CPU推理
    def __init__(self, config):
        super().__init__(config)
        assert not self.config["cpu_offload"], "Not support cpu-offload for TaylorCaching"
    #############################################################################################################

    def infer(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec=None, frist_frame_token_num=None):
        # 1. latents: 得到32400
        txt_seq_len = txt.shape[0]
        
        # 2. prompt: 得到256
        img_seq_len = img.shape[0]

        # 20个
        # 分别处理latents和prompt
        self.scheduler.current["stream"] = "double_stream"
        for i in range(self.double_blocks_num):
            self.scheduler.current["layer"] = i
            img, txt = self.infer_double_block(weights.double_blocks_weights[i], img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

        # 3. 噪声卷积和文本特征拼接，形状为[32656, 3072]
        x = torch.cat((img, txt), 0)

        # 40个
        # 合并
        self.scheduler.current["stream"] = "single_stream"
        for i in range(self.single_blocks_num):
            self.scheduler.current["layer"] = i
            x = self.infer_single_block(weights.single_blocks_weights[i], x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis, token_replace_vec, frist_frame_token_num)

        img = x[:img_seq_len, ...]
        return img, vec

    def infer_double_block(self, weights, img, txt, vec, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis):
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

        # 4. 再进行txt: mod调制，就是对融合后的特征向量做线性变换，变换的结果是形状从[1, 3072]变为[1, 18432]
        # 4.1 然后切成6部分，分别是1_shift, scale, gate, 2_shift, scale, gate，每个形状是[1, 3072]
        # 4.2 两者都是对融合后的特征向量进行调制，只不过使用的权重矩阵不一样
        txt_mod_out = weights.txt_mod.apply(vec_silu)
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod_out.chunk(6, dim=-1)

        # 5. 开始选择是否进行注意力计算
        # 5.1 正常计算
        if self.scheduler.current["type"] == "full":
            # 5. 再进行img: img_attn_qkv, img_attn_q_norm, img_attn_k_norm, 自注意力计算
            # 5. 再进行txt: attn_qkv, attn_q_norm, attn_k_norm, 自注意力计算
            img_q, img_k, img_v = self.infer_double_block_img_pre_atten(weights, img, img_mod1_scale, img_mod1_shift, freqs_cis)
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
                weights,
                img,
                img_attn,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            )
            txt = self.infer_double_block_txt_post_atten(
                weights,
                txt,
                txt_attn,
                txt_mod1_gate,
                txt_mod2_shift,
                txt_mod2_scale,
                txt_mod2_gate,
            )
            return img, txt

        # 5.2 近似计算
        elif self.scheduler.current["type"] == "taylor_cache":
            # 5.2 直接跳过噪声卷积的层归一化，到注意力输出的线性变换，用泰勒计算替代第一次残差
            # 5.2 直接开始第一次残差连接前的gate
            self.scheduler.current["module"] = "img_attn"
            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)
            out = out * img_mod1_gate
            img = img + out

            # 5.2 用泰勒计算替代第二次残差
            # 5.2 直接开始第二次残差连接前的gate
            self.scheduler.current["module"] = "img_mlp"
            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)
            out = out * img_mod2_gate
            img = img + out

            self.scheduler.current["module"] = "txt_attn"
            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)
            out = out * txt_mod1_gate
            txt = txt + out

            self.scheduler.current["module"] = "txt_mlp"
            out = out * txt_mod2_gate
            txt = txt + out

            return img, txt

    def infer_double_block_img_post_atten(
        self,
        weights,
        img,
        img_attn,
        img_mod1_gate,
        img_mod2_shift,
        img_mod2_scale,
        img_mod2_gate,
    ):
        # 1. 在cache_dic中设置cache[-1]["double_stream"][0]["img_attn"] = {}
        self.scheduler.current["module"] = "img_attn"                                   
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)             

        out = weights.img_attn_proj.apply(img_attn)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out) 

        out = out * img_mod1_gate
        img = img + out

        self.scheduler.current["module"] = "img_mlp"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = torch.nn.functional.layer_norm(img, (img.shape[1],), None, None, 1e-6)
        out = out * (1 + img_mod2_scale) + img_mod2_shift
        out = weights.img_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.img_mlp_fc2.apply(out)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

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
        self.scheduler.current["module"] = "txt_attn"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = weights.txt_attn_proj.apply(txt_attn)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

        out = out * txt_mod1_gate
        txt = txt + out

        self.scheduler.current["module"] = "txt_mlp"
        taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

        out = torch.nn.functional.layer_norm(txt, (txt.shape[1],), None, None, 1e-6)
        out = out * (1 + txt_mod2_scale) + txt_mod2_shift
        out = weights.txt_mlp_fc1.apply(out)
        out = torch.nn.functional.gelu(out, approximate="tanh")
        out = weights.txt_mlp_fc2.apply(out)
        derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

        out = out * txt_mod2_gate
        txt = txt + out
        return txt

    def infer_single_block(self, weights, x, vec, txt_seq_len, cu_seqlens_qkv, max_seqlen_qkv, freqs_cis):
        out = torch.nn.functional.silu(vec)
        out = weights.modulation.apply(out)
        mod_shift, mod_scale, mod_gate = out.chunk(3, dim=-1)

        # 1. 开始选择是否进行注意力计算
        # 1.1 正常计算
        if self.scheduler.current["type"] == "full":
            out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
            x_mod = out * (1 + mod_scale) + mod_shift
            x_mod = weights.linear1.apply(x_mod)
            qkv, mlp = torch.split(x_mod, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

            self.scheduler.current["module"] = "attn"
            taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

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

            derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, attn)
            self.scheduler.current["module"] = "total"
            taylor_cache_init(self.scheduler.cache_dic, self.scheduler.current)

            out = torch.nn.functional.gelu(mlp, approximate="tanh")
            out = torch.cat((attn, out), 1)
            out = weights.linear2.apply(out)
            derivative_approximation(self.scheduler.cache_dic, self.scheduler.current, out)

            out = out * mod_gate
            x = x + out
            return x

        # 1.2 近似计算
        elif self.scheduler.current["type"] == "taylor_cache":
            self.scheduler.current["module"] = "total"
            out = taylor_formula(self.scheduler.cache_dic, self.scheduler.current)
            out = out * mod_gate
            x = x + out
            return x
