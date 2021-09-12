import pathlib
import pickle

import torch
import torchvision
import model

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from time import time

cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')

print("Is GPU available: ", torch.cuda.is_available())
print("Device in use: ", device)

nif = "Pickle/dataset_serialized.npy"
saved_dataModelWithWeights = "Pickle/serialized_torch_weights.npy"
batch_size = 128
writer = SummaryWriter('runs/CNN-finalProject')

picture_train = "/home/mtopolovec/PycharmProjects/zavrsniRad/NeuronskaMreza/Projekt/bird_data/birds/train"
picture_test = "/home/mtopolovec/PycharmProjects/zavrsniRad/NeuronskaMreza/Projekt/bird_data/birds/test"

transforms = torchvision.transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#For checking how much there is to test!
#classes = sorted([i.name.split("/")[-1] for i in pathlib.Path(picture_train).iterdir()])

train_set = torchvision.datasets.ImageFolder(picture_train, transform=transforms)
test_set = torchvision.datasets.ImageFolder(picture_test, transform=transforms)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True)


##### BIRANJE RESNET ARHITEKTURE IZ MODELA #######
model = model.ResNet50().to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#### TRENIRANJE ####
time0 = time()
epochs = 1
train_per_epoch = int(len(train_set) / batch_size)

maxAccuracy = 0
minLoss = 100

for e in range(epochs):
    running_loss = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for idx, (images, labels) in loop:

        images = images.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels.to(device))
        loss.backward()
        optimizer.step()

        ### TQDM BAR ###
        predictions = output.argmax(dim=1, keepdim=True).squeeze().to(device)
        labels = labels.to(device)
        correct = (predictions == labels).sum().item()
        accuracy = 100. * (correct / len(predictions))
        loop.set_description(f"Epoch [{(e+1)}/{epochs}")
        loop.set_postfix(loss=loss.item(), acc=accuracy)

        ### TENSORBOARD ###
        writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
        writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)

        running_loss += loss.item()
        currentLoss = running_loss / len(train_loader)

        if accuracy > maxAccuracy and currentLoss < minLoss:
            maxAccuracy = accuracy
            minLoss = currentLoss
            print("New accuracy is: " + str(maxAccuracy) + " with new minimal loss: " + str(minLoss))
            torch.save(model.state_dict(), saved_dataModelWithWeights)
    else:
        #torch.save(model.state_dict(), saved_dataModelWithWeights)
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))
print("\nWe trained (in minutes):", (time() - time0) / 60)


num_correct = 0
num_samples = 0
model.eval()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    #print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')
    print(
        f'Number of correct ones {num_correct} out of {num_samples} that makes accuracy of {float(num_correct) / float(num_samples) * 100:.2f}%')