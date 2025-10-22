import spas_sage_attn
for module_info in pkgutil.iter_modules(spas_sage_attn.__path__):
    print(module_info.name)

# import spas_sage_attn
# import torch

# q = torch.randn(32760, 12, 128, dtype=torch.bfloat16).cuda()
# k = torch.randn(32760, 12, 128, dtype=torch.bfloat16).cuda()
# v = torch.randn(32760, 12, 128, dtype=torch.bfloat16).cuda()
