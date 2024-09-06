import time

import torch.optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
from torch import nn
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the dataset

train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

print("训练长度为:{}".format(train_data_size))
print("测试长度为:{}".format(test_data_size))

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

network = Network().to(device)

loss_function = nn.CrossEntropyLoss().to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10


writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("-----------第{}轮训练开始---------".format(i+1))

    network.train()
    for data in train_dataloader:
        imgs, target = data
        imgs, target = imgs.to(device), target.to(device)
        outputs = network(imgs)
        loss = loss_function(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("花费时间:{}".format(end_time - start_time))
            print("训练次数:{}, 损失为:{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss, total_train_step)

    network.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, target = data
            imgs, target = imgs.to(device), target.to(device)
            outputs = network(imgs)
            loss = loss_function(outputs, target)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy

    print("测试集的loss:{}".format(total_test_loss / test_data_size))
    print("测试集的acc:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(network, "network_{}".format(i))
    print("模型已保存")

writer.close()
