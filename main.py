import os
import time
import torch
from PIL import Image
import numpy as np
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
# import torchvision
import matplotlib.pyplot as plt


import sys
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()
def loadtraindata(path):
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose(
                                                    [transforms.Resize((224, 224)),
                                                     transforms.Grayscale(num_output_channels=1),
                                                     # transforms.CenterCrop(224),
                                                     transforms.ToTensor()])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
    return trainloader


def get_test_result(net):
    fo = open("C:\\Users\\13302\\Desktop\\test\\pred2.txt", "w")
    for i in range(1800):
        m = i + 1
        tpath = os.path.join('C:\\Users\\13302\\Desktop\\test\\test\\' + str(m) + '.bmp')  # 路径(/home/ouc/river/test)+图片名（img_m）
        fopen = Image.open(tpath)
        transform = transforms.Compose([transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
        data = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
        data = data.unsqueeze(0)
        y = net(data).argmax(dim=1)[0]
        if y == 0:
            fo.write("1\n")
        if y == 4:
            fo.write("2\n")
        if y == 5:
            fo.write("3\n")
        if y == 6:
            fo.write("4\n")
        if y == 7:
            fo.write("5\n")
        if y == 8:
            fo.write("6\n")
        if y == 9:
            fo.write("7\n")
        if y == 10:
            fo.write("8\n")
        if y == 11:
            fo.write("9\n")
        if y == 1:
            fo.write("10\n")
        if y == 2:
            fo.write("11\n")
        if y == 3:
            fo.write("12\n")
    fo.close()



def test(data_iter, net):
    for X, Y in data_iter:
        print(net(X).argmax(dim=1))


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                # print(net(X.to(device)).argmax(dim=1))
                # print(y.to(device))
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n
def startTrain(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print("epoch", epoch)
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            print(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        train_acc = train_acc_sum / n
        duration = time.time() - start
        loss = train_l_sum / batch_count
        # 训练结束，保存网络
        torch.save(net, 'CNN')
        # torch.save(net.state_dict(), 'net_params.pkl')
        print('epoch %d, spentTime %.2f sec, loss %.5f, train accuracy %.2f, test accuracy %.2f'
              % (epoch + 1, duration, loss, train_acc, test_acc))


# 构建AlexNet进行训练
# AlexNet包含8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层，识别12个汉字因此是12
            nn.Linear(4096, 12),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

if __name__ == '__main__':
    # print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.is_available())

    # net = AlexNet()
    net = torch.load('CNN')
    # net.load_state_dict(torch.load("t_params.pkl"))
    # lr, num_epochs = 0.000001, 1
    # trainloader = loadtraindata("C:\\Users\\13302\\Desktop\\train\\train") #加载训练数据
    # testloader = loadtraindata("C:\\Users\\13302\\Desktop\\train\\test2")
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # startTrain(net, trainloader, trainloader, optimizer, device, num_epochs)

    get_test_result(net)

    # testloader = loadtraindata("C:\\Users\\13302\\Desktop\\train\\test2")
    # for X, Y in testloader:
        # print(X)
        # y_p = net(X.to(device)).argmax(dim=1)
        # print(y_p)



    # （博:0），（学：4），（笃：5），（志：6），（切：7），（问：8）
    # （近：9），（思：10），（自：11），（由：1），（无：2），（用：3）
    # 查看精确度
    # testloader = loadtraindata("C:\\Users\\13302\\Desktop\\train\\test")
    # testloader = loadtraindata("C:\\Users\\13302\\Desktop\\train\\test2")
    # print(evaluate_accuracy(trainloader, net))

    # for X, Y in testloader:
    #     print(X.shape)
        # print(X)
        # y = net(X)
        # print(y)
    # test(testloader, net)
    # print(evaluate_accuracy(testloader, net))

    # print(net)
    # trainloader = loadtraindata()
    # for image, label in trainloader:
    #     print(image[0])
    #     print(image[0].shape)
    #     print(label[0])

    # data = load_test_data(net)
    # y = net(data)
    # print(y.argmax(dim=1)[0])
    # print(predict)

    # 图片的高H为460，宽W为346，颜色通道C为3
    # print(img.shape)
    # print(img.dtype)
    # print(type(img))
    # plt.imshow(img)
    # plt.show()

