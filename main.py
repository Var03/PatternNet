import sys
import dill as pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from patternNet import PatternNetSignalEstimator

mnist_data_path = 'data/mnist_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, init_depth=16):
        super(CNN, self).__init__()

        d = {}
        for i in range(5):
            d[i] = init_depth * 2**i
        self.conv1 = nn.Conv2d( 1, d[0], kernel_size=(3,3) )
        self.conv2 = nn.Conv2d( d[0], d[1], kernel_size=(3,3) )
        self.conv3 = nn.Conv2d( d[1], d[2], kernel_size=(3,3) )
        self.conv4 = nn.Conv2d( d[2], d[3], kernel_size=(3,3) )
        self.dense1 = nn.Linear( d[3], d[4] )
        self.dense2 = nn.Linear( d[4], 10 )

        self.a = nn.ReLU()
        self.mp = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.a(self.conv1(x))
        x = self.mp(x)
        x = self.a(self.conv2(x))
        x = self.mp(x)
        x = self.a(self.conv3(x))
        x = self.a(self.conv4(x))
        x = self.a(self.dense1(x.view(batch_size, -1)))
        x = self.dense2(x)
        return x


def imshow(img):
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def gaussian_noise(item, eps=.2):
    return torch.randn_like(item)*eps + item

def map_domain(x):
    return x*2 - 1

transform = transforms.Compose([ transforms.ToTensor(), transforms.Lambda(map_domain) ])

def get_test_accuracy(model):
    test_dataset = datasets.MNIST(root=mnist_data_path, train=False, transform=transform, download=True)
    batch_size = 200
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    with torch.no_grad():
        for batch, labels in test_dataloader:
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

    test_dat_len = len(test_dataloader.dataset)
    accuracy = 100. * correct.item() / test_dat_len
    return accuracy

# train for mnist
def train_classify_mnist(model, save_name):
    print("training a model to classify mnist...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_dataset = datasets.MNIST(root=mnist_data_path, train=True, transform=transform, download=True)
    batch_size = 200
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = 10

    model.train()
    for epoch in range(n_epochs):
        losses = []
        for batch, labels in train_dataloader:
            model.zero_grad()
            batch, labels = batch.to(device), labels.to(device)

            y = model(batch)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print('Epoch %d | Train Loss: %.4f' % (epoch, np.mean(losses)) )
    print("Test Accuracy: ", get_test_accuracy(model))
    torch.save(model.state_dict(), "model_weights/"+save_name+'_mnist_classifier.pt')

def train_explain_mnist(model_arch, save_name):
    print("training the patterns of the patternNet signal estimator...")
    model_arch.load_state_dict(torch.load("model_weights/"+save_name+'_mnist_classifier.pt'))
    signal_estimator = PatternNetSignalEstimator(model_arch)

    train_dataset = datasets.MNIST(root=mnist_data_path, train=True, transform=transform, download=True)
    batch_size = 10
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    # only one epoch of data is needed to find the closed-form solution
    # labels aren't necessary
    # it is a RAM-hungry process and it might need lower batchsize to train the patterns
    # lowering batch size does not increase stochasticity, because it is not SGD
    with torch.no_grad():
        for batch, _ in tqdm(train_dataloader):
            signal_estimator.update_E(batch.to(device))

    signal_estimator.get_patterns()

    print("Saving to pickle file...")
    filehandler = open("model_weights/"+save_name+'_mnist_explainer.pkl', 'wb')
    pickle.dump(signal_estimator, filehandler)
    torch.save(signal_estimator.net.state_dict(), "model_weights/"+save_name+'_mnist_explainer_net.pt')

def show_explain_mnist(save_name):
    print("Loading from pickle file...")
    filehandler = open("model_weights/"+save_name+'_mnist_explainer.pkl', 'rb')
    signal_estimator = pickle.load(filehandler)
    signal_estimator.net.load_state_dict(torch.load("model_weights/"+save_name+'_mnist_explainer_net.pt'))

    test_dataset = datasets.MNIST(root=mnist_data_path, train=True, transform=transform, download=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=25, shuffle=False)

    for batch, _ in test_dataloader:
        plt.figure()
        imshow(make_grid(batch, nrow=5))

        plt.figure()
        signal = signal_estimator.get_signal(batch.to(device))
        # signal = signal_estimator.get_attribution(batch.to(device)) # patternAttribution
        # signal = signal_estimator.get_signal(batch.to(device), c=5) # class activation mapping for specific class

        # map between 0 and 1 such that 0 becomes .5 (grey)
        signal = signal/2 + .5

        # # map between 0 and 1, hard
        # for i in range(len(signal)):
        #     r = torch.max(signal[i, ...]) - torch.min(signal[i, ...])
        #     signal[i, ...] = signal[i, ...] / r
        #     signal[i, ...] = signal[i, ...] + abs(torch.min(signal[i, ...]))

        imshow(make_grid(signal, nrow=5))

        plt.show()
        sys.exit()

if __name__ == "__main__":

    net_arch = CNN().to(device)
    save_name = 'cnn'

    train_classify_mnist(net_arch, save_name)
    train_explain_mnist(net_arch, save_name)
    show_explain_mnist(save_name)