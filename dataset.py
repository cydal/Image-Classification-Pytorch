#
# Author: cydal
#
#
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    """Image Classification Dataset"""
    def __init__(self, img_paths, labels=None, root_dir: str = None, transform=None):
        """
        Args:
            img_paths (string): Path to the images.
            labels (np.ndarray) : list or ndarray containing labels corresponding to images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_paths = img_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.root_dir, self.img_filepath.iloc[idx, 0])
        img = np.array(Image.open(img_filename).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.lbls is not None:
            return img, self.labels[idx]
        else:
            return img
