#
# Author: cydal
#
#

""" General Utility """
import os
import glob2
import logging
import random
import numpy as np
import pandas as import pd
from sklearn import metrics
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F



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
