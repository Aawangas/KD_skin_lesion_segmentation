import numpy as np
import random
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from typing import Tuple

class DataAugmentationTransform:
    """
    Applies the same geometric transformations to both image and ground truth,
    but different pixel-level transformations.
    """
    def __init__(self, geometric_transform, image_transform, gt_transform):
        self.geometric_transform = geometric_transform
        self.image_transform = image_transform
        self.gt_transform = gt_transform

    def __call__(self, image: Image.Image, gt: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = np.random.randint(2147483647)

        # Apply geometric transform to image
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.geometric_transform(image)
        
        # Apply geometric transform to GT using the same seed
        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.geometric_transform(gt)

        # Apply specific transforms
        image = self.image_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

class MirrorPadding:
    """
    Custom padding using mirror reflection.
    """
    def __init__(self, padding: int):
        self.padding = padding

    def __call__(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)
        padded_image = cv2.copyMakeBorder(
            image_np, self.padding, self.padding, self.padding, self.padding,
            cv2.BORDER_REFLECT
        )
        return Image.fromarray(padded_image)

def get_train_transform(size: int = 224):
    geometric = transforms.Compose([
        transforms.Resize(size=(size, size)),
        MirrorPadding(60),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
        ),
        transforms.CenterCrop(size),
    ])

    image_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    gt_transform = transforms.ToTensor()

    return DataAugmentationTransform(geometric, image_transform, gt_transform)

def get_val_transform(size: int = 224):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gt_transform = transforms.ToTensor()
    
    return DataAugmentationTransform(transforms.Resize((size, size)), image_transform, gt_transform)

# Default transforms for backward compatibility
train_transform = get_train_transform()
vanilla_transform = get_val_transform()
