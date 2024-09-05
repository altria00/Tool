import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
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

network = Network().cuda()

loss_function = nn.CrossEntropyLoss().cuda()

learning_rate = 1e-2
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10


writer = SummaryWriter("/logs_train")
for i in range(epoch):
    print("-----------第{}轮训练开始---------".format(i+1))

    for data in train_dataloader:
        imgs, target = data
        imgs, target = imgs.cuda(), target.cuda()
        outputs = network(imgs)
        loss = loss_function(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, 损失为:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, target = data
            imgs, target = imgs.cuda(), target.cuda()
            outputs = network(imgs)
            loss = loss_function(outputs, target)
            total_test_loss = total_test_loss + loss.item()
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
