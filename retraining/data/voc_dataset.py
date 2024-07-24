__all__ = ['VOCSubset']

import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[2])))

import matplotlib.pyplot as plt
from functools import partial
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import random

from retraining.models.torch.mobilenetv3 import ImageSegmenationMobilenetV3, INPUT_SHAPE

class VOCSubset(Dataset):
    def __init__(self, root=None, year='2012', image_set='train', download=False, transform=None, target_transform=None, subset_size=5):
        if root is None:
            root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

        self.input_shape = INPUT_SHAPE
        self.transform = ImageSegmenationMobilenetV3.get_transform(self.input_shape)
        self.target_transform = partial(mask_transform, input_shape=self.input_shape)
        full_voc_dataset = VOCSegmentation(root=root, year=year, image_set=image_set, download=download, transform=self.transform, target_transform=self.target_transform)
        full_dataset_size = len(full_voc_dataset)
        
        
        indices = list(range(full_dataset_size))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        
        self.voc_subset = Subset(full_voc_dataset, subset_indices)
    
    def __len__(self):
        return len(self.voc_subset)
    
    def __getitem__(self, idx):
        return self.voc_subset[idx]
    
def mask_transform(mask, input_shape):
    mask = transforms.Resize(input_shape)(mask)
    mask = transforms.ToTensor()(mask).squeeze(0)
    return mask.long()


if __name__ == "__main__":
    custom_dataset = VOCSubset()

    idx = 3
    image, mask = custom_dataset[idx]
    print(f"len data: {len(custom_dataset)}")
    print(f"idx: {idx}")
    print(f"image: {image}")
    print(f"mask: {mask}")
    print(f"image shape: {image.shape}")
    print(f"mask shape: {mask.shape}")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Label Mask')
    plt.show()
