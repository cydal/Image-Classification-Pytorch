#
# Author: cydal
#
#

""" General Utility """
import os
import json
import glob2
import logging
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from PIL import Image
import cv2

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensor


class Params:
    """Load hyperparameters from a json file

    Example:

    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5
    ```

    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self) -> str:
        return str(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def set_global_seeds():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

def datasets_to_df(ds_path: str):
    """
    Convert folder path to DataFrame

    Args:
        ds_path (string): Path to dataset

    Returns:
        pd.DataFrame : A pandas dataframe containing paths to dataset and labels.
    """

    if not os.path.exists:
        raise FileNotFoundError(f"Directory Dataset not found: {ds_path}")

    filenames = glob2.glob(os.path.join(ds_path, "*/**.jpg"))

    labels = []
    img_filenames = []

    for f in filenames:
      labels.append(f.split("/")[-2])
      img_filenames.append(f)

    df = pd.DataFrame()
    df["fname"] = img_filenames
    df["label"] = labels


    return(df)

def pred_images_to_df(ds_path: str):
    """
    Convert images path to DataFrame

    Args:
        ds_path (string): Path to dataset (prediction)

    Returns:
        pd.DataFrame : A pandas dataframe containing paths to dataset and labels.
    """
    if not os.path.exists:
        raise FileNotFoundError(f"Directory Dataset not found: {ds_path}")

    filenames = glob2.glob(os.path.join(ds_path, "*/**.jpg"))

    img_filenames = []

    for f in filenames:
      labels.append(None)
      img_filenames.append(f)

    df = pd.DataFrame({
        "fname":img_filenames,
        "label":[None] * len(img_filenames)
    })

    return(df)

def get_train_transforms(h, w, mu, std):
    """
    Transformations using albumentation library
    """

    train_transforms = A.Compose([
    A.Resize(h, w, cv2.INTER_NEAREST),
    A.CenterCrop(h, w),
    A.Normalize(mean=mu, std=std),
    ToTensor()
    ])

    return(train_transforms)

def get_val_transforms(h, w, mu, std):
    """
    Transformations using albumentation library
    """

    val_transforms = A.Compose([
    A.Resize(h, w, cv2.INTER_NEAREST),
    A.Normalize(mean=mu, std=std),
    ToTensor()
    ])

    return(val_transforms)

def plot_hist(history):
    train_losses, train_acc = history["train_losses"], history["train_acc"]
    test_losses, test_acc = history["test_losses"], history["test_acc"]

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
