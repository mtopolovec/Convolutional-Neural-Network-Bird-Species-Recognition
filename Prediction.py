import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.autograd import Variable

import model

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')

print("Is GPU available: ", torch.cuda.is_available())
print("Device in use: ", device)

transforms = torchvision.transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 128
dataset = "/home/mtopolovec/PycharmProjects/zavrsniRad/NeuronskaMreza/Projekt/Pickle/serialized_torch_weights.npy"
picture_validation = "/home/mtopolovec/PycharmProjects/zavrsniRad/NeuronskaMreza/Projekt/bird_data/birds/valid"

valid_set = torchvision.datasets.ImageFolder(picture_validation, transform=transforms)

valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)

model = model.ResNet50().to(device)
model.load_state_dict(torch.load(dataset))
model.eval()

num_correct = 0
num_samples = 0

with torch.no_grad():
    for x, y in valid_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(f'Number of correct ones {num_correct} out of {num_samples} that makes accuracy of {float(num_correct) / float(num_samples) * 100:.2f}%')
