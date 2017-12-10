import torch
import torch.nn as nn
import torch.nn.functional as ftn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import random

# Make Convolutional Neural Network and Training repeatedly
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = ftn.relu(ftn.max_pool2d(self.conv1(x), 2))
        x = ftn.relu(ftn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = ftn.relu(self.fc1(x))
        x = ftn.dropout(x, training=self.training)
        x = self.fc2(x)
        return ftn.log_softmax(x)


def inp(epoch):
    random.shuffle(data)
    Num_channels = 1
    Batch_size = 32
    Height, Width = data[0][0].shape
    n = len(data)

    for i in range(0, n, Batch_size):
        input = []
        target = []
        for k in range(0, Batch_size):
            input.append(data[k][0])
            target.append(data[k][1])
        input = torch.Tensor(input)

        input.unsqueeze_(0)
        input = input.view(Batch_size, Num_channels, Height, Width)
        target = Variable(torch.LongTensor(target))
        input = Variable(input)
        train(i, input, target)

    test(epoch)


def test(epoch):
    model.eval()
    Num_channels = 1
    Height, Width = test_data[0][0].shape
    n = len(test_data)
    input = []
    target = []
    for k in range(0, n):
        input.append(test_data[k][0])
        target.append(test_data[k][1])

    input = torch.Tensor(input)
    input.unsqueeze_(0)
    input = input.view(-1, Num_channels, Height, Width)
    target = Variable(torch.LongTensor(target))
    input = Variable(input)

    out = model(input)
    _, pred = out.max(1)
    d = 0
    for i in range(0, n):
        if pred[i].data[0] == target[i].data[0]:
            d += 1
    accuracy = float(1.0 * d / n) * 100
    print("Epoch number", epoch, ": accuracy = ", accuracy, "%")

    if (ans[-1] <= accuracy):
        ans.append(accuracy)
        torch.save(model.state_dict(), save_file)
        print("Saved model")


def train(batch, input, target):
    model.train()
    for i in range(0, 10):
        out = model(input)
        loss = ftn.nll_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()


model = Net()
opt = optim.Adam(model.parameters(), lr=0.0001)

file_name = "mnist_data.dat"
with open(file_name, "rb") as file:
    data = pickle.load(file)

file_name = "mnist_test.dat"
with open(file_name, "rb") as file:
    test_data = pickle.load(file)


save_file = "mnist_parameters.inp"
ans = [0]
try:
    model.load_state_dict(torch.load(save_file))
    print("load saved data")
except IOError:
    torch.save(model.state_dict(), save_file)
    print("init")
    model.load_state_dict(torch.load(save_file))

for epoch in range(40):
    inp(epoch)