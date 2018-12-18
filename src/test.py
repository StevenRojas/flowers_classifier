# 1 Import libraries
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import numpy as np

from network import Network
from trainer import Trainer
from arch import Arch


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


#2 Load data

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#3 Define different architectures
arch = Arch()

#4 Define hyper parameters
epochs = 2
show_every = 128
criterion = nn.NLLLoss()
flatten = lambda x: x.view(x.shape[0], -1)

#5 Train network
for architecture in arch.next_arch():
    network = Network(architecture)
    print(network.get_description())
    for lr in arch.learning_rates:
        network.set_lr(lr)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        trainer = Trainer(network, criterion, optimizer)\
            .set_train_loader(trainloader)\
            .set_test_loader(testloader)

        trainer.train(epochs, flatten, show_every)
        filename = os.getcwd() + "/network/checkpoints/checkpoint_{}_{}.pth".format(network.get_id(), network.get_lr())
        trainer.save_checkpoint(filename)

filename = os.getcwd() + "/network/checkpoints/checkpoint_net_2_0.0001.pth"
checkpoint = torch.load(filename)
network = Network(checkpoint['architecture'])
network.load_state_dict(checkpoint['model']['state_dict'])
print(network)
images, labels = next(iter(testloader))
img = images[1]
ps = torch.exp(network(img, flatten))
print(ps.data.numpy().squeeze(), network(img, flatten))
view_classify(img, ps, version='Fashion')
# accuracy = checkpoint['values']['accuracy_values']
# plt.plot(accuracy)
# plt.show()

# model = fc_model.Network(checkpoint['input_size'],
#                          checkpoint['output_size'],
#                          checkpoint['hidden_layers'])
# model.load_state_dict(checkpoint['state_dict'])
model = {}

checkpoint = {
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epochs": epochs,
    "class_idx_training": trainloader.dataset.class_to_idx
}
torch.save(checkpoint, "checkpoint.pth")

model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                      nn.ReLU(),
                      nn.Linear(4096, 2048),
                      nn.ReLU(),
                      nn.Linear(2048, 102))
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict = checkpoint["state_dict"]
model.eval()
