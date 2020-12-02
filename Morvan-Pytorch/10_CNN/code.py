"""
CNN 卷积神经网络
    手写数字识别
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,  # False  return test data
    transform=torchvision.transforms.ToTensor(),  # transform to tensor  ( 0, 1)
    download=DOWNLOAD_MNIST
)

# plot one example
# print(train_data.train_data.size())  # (60000, 28, 28)
# print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]

# build CNN network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(   # (1, 28, 28)
                in_channels=1,  # 通道数
                out_channels=16,  # filter的个数（卷积核的个数，累计得到深度）
                kernel_size=5,  # filter的宽和高是5x5
                stride=1,  # 步长
                padding=2  # if stride = 1, padding = (kernel_size - 1) / 2
            ),  # -> (16, 28, 28)  使用了padding，所以长宽没有变
            nn.ReLU(),  # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # -> (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(kernel_size=2)  # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每50步测试训练效果
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| Step ', step, ' | train loss: %.4f' % loss.data[0], ' | test accuracy: %.4f' % accuracy)

# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, ' prediction number')
print(test_y[:10].numpy(), ' real number')