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

b = torch.randn(3, 2)
tensor = torch.chunk(b, chunks=2, dim=1)  # 分割
print("chunk")
print(b)
print(tensor)

tensor = torch.arange(3, 19).view(4, 4)
index = torch.tensor([[3, 1, 2, 0]])
output = tensor.gather(0, index)  # 索引选取
print("gather")
print(tensor)
print(output)

tensor = torch.randn(3, 2, dtype=torch.float)
print("reshape")
print(tensor)
tensor = torch.reshape(tensor, [2, 3])  # 改变形状
print(tensor)
tensor = torch.reshape(tensor, [-1])
print(tensor)

tensor = torch.arange(10).reshape(5, 2)
print("split")
print(tensor)
tensor = torch.split(tensor, 2)  # 切割张量
print(tensor)

tensor = torch.arange(6).reshape(3, 2)
print("squeeze")
print(tensor)
tensor = torch.reshape(tensor, [3, 1, 2])
print(tensor)
tensor = torch.squeeze(tensor, dim=1)  # 减少维度
print(tensor)

tensor2 = torch.randn(3, 2)
tensor = torch.stack([tensor2, tensor])
print("stack")
print(tensor, tensor.shape)

a = torch.tile(tensor, [2, 1])  # 复制  行(2, 1) 列(1, 2)
print("tile")
print(a)

tensor = torch.arange(6).reshape(3, 2)
print("transpose")
print(tensor)
tensor = torch.transpose(tensor, 0, 1)
print(tensor)

tensor = torch.unbind(tensor, dim=1)  # 对于某个维度，按这个维度拆成多个小张量，小张量比原来降了一个维度
print("unbind")
print(tensor)

tensor = torch.arange(6).reshape(3, 2)
print(tensor)
tensor = torch.unsqueeze(tensor, 0)  # 增加维度
print("unsqueeze")
print(tensor, tensor.shape)

x = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
y = torch.tensor([[5, 6, 7], [7, 8, 9], [9, 10, 11]])
z = torch.where(x > 5, x, y)  # 根据条件，也就是condition，返回从x或y中选择的元素的张量根据条件，也就是condition，返回从x或y中选择的元素的张量
print("where")
print(z)

torch.manual_seed(seed=23)  # 设置随机种子

tensor = torch.rand(3, 3)
print(tensor)
tensor = torch.bernoulli(tensor)
print("bernoulli")
print(tensor)



def block_transpose(matrix, block_size):
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            for k in range(i, min(i + block_size, rows)):
                for l in range(j, min(j + block_size, cols)):
                    transposed[l][k] = matrix[k][l]

    return transposed

# matrix = torch.randn(4, 4)
# t_matrix = block_transpose(matrix, 4)
# t_matrix = torch.tensor(t_matrix)
# print(matrix)
# print(t_matrix)
