import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data.distributed import DistributedSampler


import logging
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def load_data_fashion_mnist(mnist_train, mnist_test, batch_size,train_sampler):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,sampler=train_sampler)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output




def evaluate_accuracy(data_iter, net, device=None):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(local_rank)).argmax(dim=1) == y.to(local_rank)).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = nn.parallel.DistributedDataParallel(net.cuda(local_rank), device_ids=[local_rank])
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # 为了让每张卡在每个周期中得到的数据是随机的
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

if __name__ == "__main__":

    if torch.cuda.is_available():
        logging.warning("Cuda is available!")
        if torch.cuda.device_count() > 1:
            logging.warning(f"Find {torch.cuda.device_count()} GPUs!")
        else:
            logging.warning("Too few GPU!")
            sys.exit()
    else:
        logging.warning("Cuda is not available! Exit!")
        sys.exit()
    # gpu数
    n_gpus = 2
    # 以nccl模式多卡通信，多线程多卡
    # params:
    #  world_size : gpu数
    #  rank : 当前主线程的gpu编号
    local_rank = 0
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=local_rank)
    torch.cuda.set_device(local_rank)
    net = LeNet()
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    train_sampler = DistributedSampler(mnist_train)
    batch_size = 256
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_iter, test_iter = load_data_fashion_mnist(mnist_train, mnist_test, batch_size,train_sampler)

    train(net, train_iter, test_iter, batch_size, optimizer, local_rank, num_epochs)