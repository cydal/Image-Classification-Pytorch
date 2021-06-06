#
# Author: cydal
#
#
%matplotlib inline
import json
import os
import cv2
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import torch
import torch.nn as nn
from torchsummary import summary
import wandb

from utils import datasets_to_df, get_train_transforms, get_val_transforms
from utils import set_global_seeds, Params, plot_hist

from model import train_model, build_model
from dataset import ImagesDataset


#wandb.init(project="img_classifier") ## take from params


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")

if __name__ == "__main__":

    # Todo: Set set_global_seeds


    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    #ToDo: Set WandB


    train_root_dir = params.train_path
    train_df = datasets_to_df(params.train_path)

    print(train_df)
    print(f"[INFO] Found Image Data of shape: {train_df.shape} ")

    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(train_df["label"])
    print("[INFO] Label Encoding:", labelEncoder.classes_)

    # split data
    X_train, X_val, y_train, y_val = train_test_split(train_df["fname"],
                                                        y,
                                                        test_size=params.SPLIT_RATIO,
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=params.RANDOM_SEED)
    print(
        "[INFO] Training shape:",
        X_train.shape,
        y_train.shape,
        np.unique(y_train, return_counts=True)
    )

    print(
        "[INFO] Validation shape:",
        X_val.shape,
        y_val.shape,
        np.unique(y_val, return_counts=True)
    )

    cws = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
    print("[INFO] Class weights:", cws)

    class_mapping = {k: v for k, v in enumerate(labelEncoder.classes_)}
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    ##ToDo: Apply transformations
    train_transforms = get_train_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEANS,
                                      params.STDS)

    val_transforms = get_val_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEANS,
                                      params.STDS)

    train_dataset = ImagesDataset(X_train, y_train, None, train_transforms)
    val_dataset = ImagesDataset(X_val, y_val, None, val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params.BATCH_SIZE,
                                               pin_memory=True,
                                               shuffle=True,
                                               num_workers=params.NUM_WORKERS)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=params.BATCH_SIZE,
                                             pin_memory=True,
                                             shuffle=False,
                                             num_workers=params.NUM_WORKERS)

    print(next(iter(train_loader)))
    im, lbl = next(iter(train_loader))
    print(im.shape, lbl.shape)

    print("[INFO] Training length:", len(train_loader.dataset))
    print("[INFO] Validation length:", len(val_loader.dataset))

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    ## Feature Extraction
    if params.FEATURE_EXTRACT:
        net = build_model(
                params.MODEL_NAME,
                params.NUM_CLASSES,
                params.INPUT_CHANNELS,
                params.EMBED_SIZE,
                params.FEATURE_EXTRACT,
                params.USE_PRETRAIN,
                bst_model_weights=params.TRAINED_MODEL_PATH
        )

        # Set optimzer & lr_scheduler
        if params.NUM_CLASSES < 2:
            criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        #optimizer = torch.optim.AdamW(t_parameters, lr=params.LR, amsgrad=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.LR * 100,
            steps_per_epoch=len(train_loader),
            epochs=params.EPOCHS
        )

        summary(net, (params.INPUT_CHANNELS, params.WIDTH, params.HEIGHT))

        best_model, history = train_model(
            model=net,
            dataloaders=dataloaders_dict,
            num_classes=params.NUM_CLASSES,
            input_channels=params.INPUT_CHANNELS,
            batch_size=params.BATCH_SIZE,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=params.EPOCHS,
            device=device,
            use_wandb=params.USE_WANDB
        )

        plot_hist(history)
        # save best model
        torch.save(best_model, params.SAVE_MODEL_PATH)


    # Finetuning
    if params.TRAINED_MODEL_PATH is None and params.FINETUNE_LAYER != -1:
        params.TRAINED_MODEL_PATH = params.SAVE_MODEL_PATH

        print("Freezing upto => ", params.FINETUNE_LAYER, "layers")

        net = build_model(
                params.MODEL_NAME,
                params.NUM_CLASSES,
                params.INPUT_CHANNELS,
                params.EMBED_SIZE,
                params.FEATURE_EXTRACT,
                params.USE_PRETRAIN,
                params.FINETUNE_LAYER,
                bst_model_weights=params.TRAINED_MODEL_PATH
        )

        # Set optimzer & lr_scheduler
        if params.NUM_CLASSES < 2:
            criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(cws)).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        #optimizer = torch.optim.AdamW(t_parameters, lr=params.LR, amsgrad=True)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        # one cycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.LR * 100,
            steps_per_epoch=len(train_loader),
            epochs=params.EPOCHS
        )

        summary(net, (params.INPUT_CHANNELS, params.WIDTH, params.HEIGHT))

        best_model, history = train_model(
            model=net,
            dataloaders=dataloaders_dict,
            num_classes=params.NUM_CLASSES,
            input_channels=params.INPUT_CHANNELS,
            batch_size=params.BATCH_SIZE,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=params.EPOCHS,
            device=device,
            use_wandb=params.USE_WANDB
        )

        plot_hist(history)
        # save best model
        torch.save(best_model, params.SAVE_MODEL_PATH)
