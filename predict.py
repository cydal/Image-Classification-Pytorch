#
# Author: cydal
#
#

import json
import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import glob2


from sklearn.preprocessing import LabelEncoder


import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import torch
import torch.nn as nn
import wandb


from utils import get_val_transforms, Params, pred_images_to_df


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")

parser.add_argument("--img_dir",
                    default=".",
                    help="Directory containing images")

if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    imgs_df = pred_images_to_df(args.img_dir)

    test_transforms = get_val_transforms(params.HEIGHT,
                                      params.WIDTH,
                                      params.MEANS,
                                      params.STDS)

    test_dataset = ImagesDataset(imgs_df["fname"], None, None, test_transforms)

    
