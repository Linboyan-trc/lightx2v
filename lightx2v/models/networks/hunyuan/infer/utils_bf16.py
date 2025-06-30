import torch
from typing import Any, List, Tuple, Optional, Union, Dict


def rms_norm(x, weight, eps):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * weight
    return x


def rotate_half(x, shape_0, shape_1):
    # 3.x, 多头注意力噪声q[32400, 24, 128]；
    # 3. shape_0, 噪声序列长度, shape_1, 头数量
    x_real, x_imag = x.reshape(shape_0, shape_1, -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(2)


def rotary_emb(x, shape_0, shape_1, cos, sin):
    # 3.x, 多头注意力噪声q[32400, 24, 128]；
    # 3. shape_0, 噪声序列长度, shape_1, 头数量
    # 3. cos，sin，(32400, 1, 128)
    x_out = x * cos + rotate_half(x, shape_0, shape_1) * sin
    return x_out


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. xq是img_q, [32400, 24, 128]
    # 1. xk是img_k, [32400, 24, 128]
    # 1. 所以shape_0, sahpe_1, shape_2 = 32400, 24, 128
    shape_0, shape_1, shape_2 = xq.shape

    # 2. freqs_cis[0]就是freqs_cos, [32400, 3072]；这里.view(32400, 1, 128)就是只取得[32400, 3072]的前面128列，后面两千多列被丢弃
    # 2. freqs_cis[1]就是freqs_sin, [32400, 3072]
    cos = freqs_cis[0].view(shape_0, 1, shape_2)
    sin = freqs_cis[1].view(shape_0, 1, shape_2)

    # 3. img_q, 多头注意力噪声q[32400, 24, 128]；
    # 3. shape_0, 噪声序列长度, shape_1, 头数量
    # 3. cos，sin，(32400, 1, 128)
    xq_out = rotary_emb(xq, shape_0, shape_1, cos, sin)
    xk_out = rotary_emb(xk, shape_0, shape_1, cos, sin)
    return xq_out, xk_out
