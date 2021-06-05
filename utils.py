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

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


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

def get_train_transforms(h, w, mu, std):
    """
    Transformations using albumentation library
    """

    train_transforms = A.Compose([
    A.Resize(h, w, cv2.INTER_NEAREST),
    A.CenterCrop(h, w),
    A.Normalize(mean=means, std=stds),
    ToTensor()
    ])

    return(train_transforms)

def get_val_transforms(h, w, mu, std):
    """
    Transformations using albumentation library
    """

    val_transforms = A.Compose([
    A.Resize(h, w, cv2.INTER_NEAREST),
    A.Normalize(mean=means, std=stds),
    ToTensor()
    ])

    return(val_transforms)
