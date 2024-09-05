import torchvision

from torch import nn
from torch.utils.data import DataLoader

# Load the dataset

train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

print("训练长度为:{}".format(train_data_size))
print("测试长度为:{}".format(test_data_size))

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

