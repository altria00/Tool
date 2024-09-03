import torch

tensor = torch.arange(0, 9, 2)

print(tensor)

tensor = torch.eye(4)  # 对角为1

print(tensor)

tensor = torch.randn(3, 3)  # 随机生成

print(tensor)

tensor = torch.tensor([1, 2, 3], dtype=torch.float32)

print(tensor)

tensor = torch.full([3, 3], 2)  # 填充

print(tensor)

a = torch.randn(2, 2, dtype=torch.float32)
b = torch.randn(2, 3, dtype=torch.float)
print(a, b)

tensor = torch.cat([a, b], dim=1)  # 拼接

print(tensor)




