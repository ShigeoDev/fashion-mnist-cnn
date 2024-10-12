import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, img_batch):
        a = F.relu(self.conv1(img_batch))
        a = self.pool(a)
        a = F.relu(self.conv2(a))
        a = self.pool(a)
        a = torch.flatten(a, 1)            # flatten all dimensions except batch
        a = self.fc1(a)
        a = self.fc2(a)
        a = self.fc3(a)
        return a

def train(dataset, loader, model, crit, optimizer, epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
    model.to(device)

    model.train()                     # sets the model into training mode

    for epoch in range(epochs):
        full_loss = 0.0
        n_correct = 0

        for img_batch, labels in tqdm(loader):      # tqdm displays a progress bar as we iterate through the loop
            img_batch = img_batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(img_batch)
            predicted = outputs.argmax(dim=1)
            loss = crit(outputs, labels)
            loss.backward()
            optimizer.step()

            # we want the loss for every single training example so that we can calculate the average loss over the entire dataset
            full_loss += loss.item() * len(labels)
            n_correct += (predicted == labels.data).sum()     # keep track of the number of correct predictions to calculate accuracy

        average_loss = full_loss / len(dataset)
        accuracy = n_correct / len(dataset) * 100

        print(f"Epoch {epoch+1} training average loss: {average_loss:.3f}",
              f"with {accuracy:.2f}% accuracy")


def test(dataset, loader, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    model.eval()        # set the model to testing mode

    n_correct = 0
    with torch.no_grad():                       #torch.no_grad() disables computing gradients
        for img_batch, labels in tqdm(loader):
            img_batch = img_batch.to(device)
            labels = labels.to(device)
            outputs = model(img_batch)
            predicted = outputs.argmax(dim=1)
            n_correct += (predicted == labels.data).sum()

    accuracy = n_correct / len(dataset) * 100
    print(f"Test accuracy: {accuracy:.2f}%")

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5), (0.5))])

    batch_size = 64

    dataset = torchvision.datasets.FashionMNIST(
        "data", 
        download=True,
        transform=transform)
    
    trainset, testset = torch.utils.data.random_split(dataset, [.7, .3])

    trainloader = DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testloader = DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

    train(trainset, trainloader, net, criterion, optimizer, 40)
    test(testset, testloader, net)

if __name__ == "__main__":
    main()

