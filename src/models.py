import numpy as np

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V
#################################
# Models for federated learning #
#################################

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600


class TwoNN(nn.Module):
    def __init__(self, name, in_features, num_hiddens, num_classes):
        super(TwoNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens,
                             out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens,
                             out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

# McMahan et al., 2016; 1,663,370 parameters


class CNN(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(
            5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels
                               * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2)
                             * (7 * 7), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens,
                             out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# for CIFAR10


class CNN2(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CNN2, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(
            5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels
                               * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(hidden_channels * 2)
                             * (8 * 8), out_features=num_hiddens, bias=False)
        self.fc2 = nn.Linear(in_features=num_hiddens,
                             out_features=num_classes, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorNetwork(nn.Module):
    def __init__(self, name, state_size):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.steering.weight, 0, 1e-4)
        self.acceleration = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.acceleration.weight, 0, 1e-4)
        self.brake = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.brake.weight, 0, 1e-4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out1 = t.tanh(self.steering(x))
        out2 = t.sigmoid(self.acceleration(x))
        out3 = t.sigmoid(self.brake(x))
        out = t.cat((out1, out2, out3), 1)
        return out


class CriticNetwork(nn.Module):
    def __init__(self, name, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.w1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.a1 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h3 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.V = nn.Linear(HIDDEN2_UNITS, action_size)

    def forward(self, s, a):
        w1 = F.relu(self.w1(s))
        a1 = self.a1(a)
        h1 = self.h1(w1)
        h2 = h1 + a1
        h3 = F.relu(self.h3(h2))
        out = self.V(h3)
        return out
