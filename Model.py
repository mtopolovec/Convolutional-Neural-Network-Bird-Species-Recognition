import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv_mreza(nn.Module):


    def __init__(self):
        super(Conv_mreza, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        input_dims = self.izracun_dimenzije()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(input_dims, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(512, 275)

    def izracun_dimenzije(self, size=70):
        batch_data = torch.zeros((1, 3, size, size))
        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.product(batch_data.size()))

    def forward(self, batch_data):
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size(0), -1)
        batch_data = self.dropout1(batch_data)
        batch_data = self.fc1(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.dropout2(batch_data)
        batch_data = self.fc2(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.dropout3(batch_data)
        batch_data = self.fc3(batch_data)

        return batch_data
