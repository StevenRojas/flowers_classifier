# !/usr/bin/env python3
from utils import parse_arguments
from arch_handler import ArchHandler
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from trainer import Trainer
import os
from time import time


def main():
    start_time = time()
    args = parse_arguments()  # TODO: Check arguments
    print(args)
    arch_handler = ArchHandler()
    print("Getting model...")
    model = arch_handler.create_model(args)
    show_time(start_time, time(), "Model downloaded")
    print("Training model...")
    start_time = time()
    trainer = get_trainer(model, args)
    trainer.train(args.epochs)
    show_time(start_time, time(), "Model trained")

    checkpoint_path = os.getcwd() + "/" + args.save_dir
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path += "/checkpoint_t.pth"
    print("Saving model to {}".format(checkpoint_path))
    trainer.save_checkpoint(checkpoint_path)
    print("Done")


def get_trainer(model, args):
    device = "cpu"
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("Warning: CUDA is not available, using CPU")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # TODO: allow setup optimizer and it's parameters
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    # TODO: allow define the use of scheduler or not
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    trainloader, validloader, testloader = get_loaders(args.data_dir)
    trainer = Trainer(model, criterion, optimizer, device, scheduler) \
        .set_train_loader(trainloader) \
        .set_valid_loader(validloader) \
        .set_test_loader(testloader)

    return trainer


def get_loaders(data_dir):
    # TODO: Check that folders exist
    data_dir = os.getcwd() + "/" + data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/valid'

    train_transforms, val_test_transforms = get_transformations()
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)
    # Get dataloaders
    batch = 64 # TODO: Allow define batch size by command line
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return trainloader, validloader, testloader


def get_transformations():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, val_test_transforms


def show_time(start, end, label):
    diff = end - start
    time_str = str(int((diff / 3600))) + ":" + str(int((diff % 3600) / 60)) + ":" + str(int((diff % 3600) % 60))
    print("{}, elapsed time: {}".format(label, time_str))


if __name__ == "__main__":
    main()