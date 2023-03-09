
'''
这是关于OAM-Lenet网络的训练代码。
该网络是基于Google-Lenet构建的；代码完成时间：2023.3.9；作者：吴清源
这里只展示了一种采样率下的OAM贝尔态识别，其他采样率的操作是相同的。
需要说明的是，模型在重新训练后，结果会略有差异。

This is the training and testing code on the OAM-Lenet network.
The network is built on Google-Lenet; code completed: 2023.3.9; by Qing-Yuan Wu
Only one sample rate is shown here for OAM Bell state recognition, the operation is the same for the other sample rates.
It should be noted that the results of the model will vary slightly after retraining.
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from my_dataset import MyMnistDataset
from scipy.io import savemat

# 仅用CPU进行运算
# CPU-only computing
device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

# 设置mini-batch，增大batch_size可以减少训练时间,并且对loss.backward()有帮助
# Setting the mini-batch and increasing the batch_size can reduce the training time and help with loss.backward()
batch_size = 64

# 使用基于激光光源的干涉图像数据集进行训练
# Training with interference image datasets based on laser light sources
training_dataset = MyMnistDataset(root='training_set', transform=transform)
train_load = DataLoader(dataset=training_dataset, shuffle=True, batch_size=batch_size)

# 用基于单光子源的干涉图像数据集进行测试
# Testing with interference image datasets based on single photon sources
testing_dataset = MyMnistDataset(root='testing_set', transform=transform)
test_load = DataLoader(dataset=testing_dataset, shuffle=True, batch_size=batch_size)

class inception(nn.Module):
    def __init__(self, in_channels):
        super(inception, self).__init__()
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_pool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)




class google_lenet(nn.Module):
    def __init__(self):
        super(google_lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = inception(in_channels=10)
        self.incep2 = inception(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1408, 4)

    def forward(self, x):
        in_size = x.size(0)

        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)

        return x

model = google_lenet()
model.to(device)

critizer = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    loss_draw = 0.0
    for batch_index, data in enumerate(train_load, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)


        img_pred = model(inputs)

        loss = critizer(img_pred, labels.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_draw += loss.item()
        print('loss: %f' % (np.double(loss.item())))
        return loss_draw

def test():
    corrct = 0.0
    total = 0.0
    for batch_index, data in enumerate(test_load, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        img_pred = model(inputs)
        _, img_pred = torch.max(img_pred.data, dim=1)

        corrct += (img_pred==labels).sum().item()
        total += labels.size(0)
    print('corrct: %d %% [%d/%d]' % (100*corrct/total, corrct, total))
    return 100*corrct/total


if __name__ == '__main__':
    epoch = 500
    loss = []
    accuracy = []
    for k in np.arange(epoch):
        loss.append(train(k))
        print(k)
        accuracy.append(test())
        if k % 2000 == 0:
            torch.save(model.state_dict(), "./model/m=3_nornal_google_lenet_params%d.pkl" % (k))

    file_name = 'data.mat'
    savemat(file_name, {'loss': loss, 'accuracy': accuracy})
    plt.figure()
    plt.plot(loss)

    plt.figure()
    plt.plot(accuracy)
    plt.show()