import spas_sage_attn
import torch
import matplotlib.pyplot as plt
import os

# 1. 构造输入
q = torch.randn(32760, 12, 128, dtype=torch.bfloat16).cuda()
k = torch.randn(32760, 12, 128, dtype=torch.bfloat16).cuda()
v = torch.randn(32760, 12, 128, dtype=torch.bfloat16).cuda()

# 2. 直接用PyTorch计算注意力
q_ = q.float()
k_ = k.float()
v_ = v.float()
attn_weights = torch.matmul(q_, k_.transpose(-2, -1)) / (128 ** 0.5)
attn_weights = torch.softmax(attn_weights, dim=-1)
output_pt = torch.matmul(attn_weights, v_)

# 3. 用spas_sage2_attn_meansim_cuda计算注意力
q = q.unsqueeze(0)  # shape: (1, 32760, 12, 128)
k = k.unsqueeze(0)
v = v.unsqueeze(0)
output_cuda = spas_sage_attn.core.spas_sage2_attn_meansim_cuda(q, k, v, tensor_layout="NHD")
output_cuda = output_cuda.float()

# 4. 取左上角[3000, 3000]，只取第一个head
output_pt_crop = output_pt[0, 0, :3000, :3000].cpu().detach().numpy()
output_cuda_crop = output_cuda[0, 0, :3000, :3000].cpu().detach().numpy()

# 5. 保存图片
save_dir = os.path.expanduser("~/Log/10-22/")
os.makedirs(save_dir, exist_ok=True)

plt.imshow(output_pt_crop, aspect='auto')
plt.title('PyTorch Attention (left-top 3000x3000)')
plt.savefig(os.path.join(save_dir, "attn.png"))
plt.close()

plt.imshow(output_cuda_crop, aspect='auto')
plt.title('spas_sage2_attn_meansim_cuda (left-top 3000x3000)')
plt.savefig(os.path.join(save_dir, "spas_attn.png"))
plt.close()