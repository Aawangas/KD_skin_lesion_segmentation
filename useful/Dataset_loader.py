import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import numpy as np
import random

class SkinDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform):
        self.transform = transform
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 添加你需要的图像格式

        self.image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith(valid_extensions)]
        self.gt_paths = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                         if f.lower().endswith(valid_extensions)]
        self.image_paths.sort()
        self.gt_paths.sort()
        # They are already sorted in that two folders

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        gt = Image.open(self.gt_paths[idx]).convert("L")
        # Which is binary, notice that afterward
        image, gt = self.transform(image, gt)
        return image, gt

    def __len__(self):
        return len(self.image_paths)

    def check_align(self):
        print(self.image_paths[:10],
              self.gt_paths[:10])




class Partial_dataset(SkinDataset):
    def __init__(self, img_dir, gt_dir, transform, size):
        self.transform = transform

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 添加你需要的图像格式

        self.image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.lower().endswith(valid_extensions)]
        self.gt_paths = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                         if f.lower().endswith(valid_extensions)]

        self.image_paths.sort()
        self.gt_paths.sort()

        self.image_paths = self.image_paths[:size]
        self.gt_paths = self.gt_paths[:size]

        assert len(self.image_paths) == len(self.gt_paths)


def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

