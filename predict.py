import torch
import torch.nn as nn
import torch.nn.functional as ftn
from torch.autograd import Variable

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

# Predict Image by using pretrained data (mnist_parameters.inp)
def predict(input):
    model = Net()
    model.eval()
    save_file = "mnist_parameters.inp"
    try:
        model.load_state_dict(torch.load(save_file))
    except IOError:
        print("No parameters.")

    input = input / 255
    input = torch.from_numpy(input)
    input.unsqueeze_(0)
    input.unsqueeze_(0)
    input = input.float()
    input = Variable(input)

    out = model(input)
    _, pred = out.max(1)

#    print("Predicted value: " + str(pred.data[0]))
    return pred.data[0]



