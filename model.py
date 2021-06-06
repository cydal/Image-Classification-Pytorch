#
# Author: cydal
#
#
import copy
import os
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



import copy
import time
from typing import Dict
from matplotlib import colors
from sklearn import metrics
import numpy as np
from tqdm import tqdm

import wandb
import torch


def set_parameter_requires_grad(model, feature_extracting: bool, num_ft_layers: int):
    """
    Freeze the weights of the model is feature_extracting=True
    Fine tune layers >= num_ft_layers
    Batch Normalization: https://keras.io/guides/transfer_learning/

    Args:
        model: PyTorch model
        feature_extracting (bool): A bool to set all parameters to be trainable or not
        num_ft_layers (int): Number of layers to freeze and unfreezing the rest
    """
    if feature_extracting:
        if num_ft_layers != -1:
            for i, module in enumerate(model.modules()):
                if i >= num_ft_layers:
                    if not isinstance(module, nn.BatchNorm2d):
                        module.requires_grad_(True)
                else:
                    module.requires_grad_(False)
        else:
            for param in model.parameters():
                param.requires_grad = False



def build_model(model_name: str,
                num_class: int,
                in_channels: int,
                embedding_size: int,
                feature_extract: bool = True,
                use_pretrained: bool,
                bst_model_weights=None):

    """
    Various architectures to train from scratch, finetune or as feature extractor.

    Args:
        model_name (str) : Name of model from [enet-b3, enet-b7, resnext101, resnet18, resnet50, densenet121, mobilenetv2]
        num_classes (int) : Number of output classes added as final layer
        in_channels (int) : Number of input channels
        embedding_size (int): Size of intermediate features
        feature_extract (bool): Flag for feature extracting.
                               False = finetune the whole model,
                               True = only update the new added layers params
        use_pretrained (bool): Pretraining parameter to pass to the model or if base_model_path is given use that to
                                initialize the model weights
        num_ft_layers (int) : Number of layers to finetune
                             Default = -1 (do not finetune any layers)
        bst_model_weights : Best weights obtained after training pretrained model
                            which will be used for further finetuning.

    Returns:
        model : A pytorch model
    """
    model_ft = None

    if model_name == "enet-b3":
        model_ft = EfficientNet.from_pretrained("efficientnet-b3")
        if in_channels == 1:
            model_ft._conv_stem = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft._fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft._fc = head

    elif model_name == "enet-b7":
        model_ft = EfficientNet.from_pretrained("efficientnet-b7")
        if in_channels == 1:
            model_ft._conv_stem = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft._fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft._fc = head

    elif model_name == "resnext101":
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        if in_channels == 1:
            model_ft.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.fc = head

    elif model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        if in_channels == 1:
            model_ft.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.fc = head

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        if in_channels == 1:
            model_ft.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.fc.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.fc = head

    elif model_name == "densenet121":
        model_ft = models.densenet121(pretrained=use_pretrained)
        if in_channels == 1:
            model_ft.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        num_ftrs = model_ft.classifier.in_features
        head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.classifier = head

    elif model_name == "mobilenetv2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        if in_channels == 1:
            model_ft.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
        set_parameter_requires_grad(model_ft, feature_extract, num_ft_layers)
        head = nn.Sequential(
            nn.Linear(1280, embedding_size),
            nn.Linear(embedding_size, num_classes)
        )
        model_ft.classifier = head

    else:
        print("Invalid model name, exiting...")
        exit()

    # load best model dict for further finetuning
    if bst_model_weights is not None:
        pretrain_model = torch.load(bst_model_weights)
        best_model_wts = copy.deepcopy(pretrain_model.state_dict())
        if feature_extract and num_ft_layers != -1:
            model_ft.load_state_dict(best_model_wts)
    return model_ft


def train(model, device, train_loader, optimizer, epoch, criterion):
    """
    Train functionality for passed in model

    Args:
        model : Pytorch model
        device (str) : GPU / CPU
        train_loader : PyTorch data loader
        optimizer : torch optimizer
        epoch (int) : Current epoch
        criterion : Torch loss function
    """
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()

    # Predict
    y_pred = model(data)

    # Calculate loss
    # F.nll_loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader, criterion):
    """
    Test data evaluation

    Args:
        model (torch model) : Pytorch model for training
        device (str) : GPU / CPU
        test_loader (Trainloader) : Number of input channels
        criterion : Torch loss function
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

def train_model(

    model,
    dataloaders: Dict,
    num_classes: int,
    input_channels: int,
    batch_size: int,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int,
    device,
    use_wandb: bool
):
    """
    Train a pytorch classification model

    Args:
        model : PyTorch model
        dataloader: Dict containing train and val dataloaders
        num_classes (int) : Number of classes to one hot targets
        input_channels (int) : Number of channels of input
        criterion : pytorch loss function
        optimizer : pytorch optimizer function
        scheduler : pytorch scheduler function
        num_epochs (int) : Number of epochs to train the model
        device : torch.device indicating whether device is cpu or gpu
        use_wandb: a bool indicating whether to use wandb.ai for logging experiments

    Returns:
        best_model : Best pytorch model on validation dataset
        history : PyTorch history
    """
    #model =  Net().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        print("EPOCH:", epoch)
        train(model, device, dataloaders["train_loader"], optimizer, epoch)
        scheduler.step()
        test(model, device, dataloaders["val_loader"])

    history = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_acc": train_acc,
        "test_acc": test_acc,
    }
        # load best model weights
    model.load_state_dict(best_model_wts)
    return(model, history)
