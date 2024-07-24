__all__ = ['CustomDataset']

import os, sys
from pathlib import Path
sys.path.append(str((Path(os.path.abspath(__file__)).parents[2])))

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retraining.models.torch.mobilenetv3 import ImageSegmenationMobilenetV3, INPUT_SHAPE

ROOT_DIR = str((Path(os.path.abspath(__file__)).parents[2]))
LOG_PATH = os.path.join(ROOT_DIR, "logs", "realtime_monitoring", "old_model_log.csv")

class CustomDataset(Dataset):
    def __init__(self, log_path=None, transform=None, target_transform=None):
        self.log_path = log_path
        if self.log_path is None:
            self.log_path = log_path

        df = pd.read_csv(LOG_PATH)
        self.log_df = df[df['fail'] == 1]

        self.image_paths = self.log_df['original_image_path'].tolist()
        self.mask_paths = self.log_df['true_mask_image_path'].tolist()

        self.transform = transform
        if self.transform is None:
            self.transform = ImageSegmenationMobilenetV3.get_transform(INPUT_SHAPE)
        
        self.target_transform = target_transform
        if self.target_transform is None:
            self.target_transform = ImageSegmenationMobilenetV3.get_mask_transform(INPUT_SHAPE)
        
        self.to_tensor = transforms.ToTensor()
        self.color_map = ImageSegmenationMobilenetV3.get_colormap()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        label = self.color_to_label(mask, self.color_map)
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        
        if self.target_transform is not None:
            label = Image.fromarray(label)
            label = self.target_transform(label)
            label = np.array(label)

        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
    def color_to_label(self, mask_img, colormap):
        mask_array = np.array(mask_img)
        label_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)
        for idx, color in enumerate(colormap):
            label_mask[np.all(mask_array == color, axis=-1)] = idx
       
        return label_mask

# Custom 데이터셋 

if __name__ == "__main__":
    custom_dataset = CustomDataset()

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
