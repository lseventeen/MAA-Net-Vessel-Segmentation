import torch
a = torch.randn(3, 4)

d = torch.max(a, -1, keepdim=True)
d1 = torch.max(a, -1,)
print(a)