import numpy as np
import random
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

# This file is implemented to deal with the data augmentation. One thing we should notice is
# we have to apply the same geometry transform to original image and gt. We therefore implement
# DataAugmetationtransform

"""
API for dataset: 
    self.transform = transform
    ...
    image, gt = transform(image, gt)
"""


class DataAugmentationTransform:
    def __init__(self, geometric_transform, image_transform, gt_transform):
        # We should have the same transformation for image and gt. However, things like changing
        # the color only applies to images. In fact, the geometry changing can be applied first, followed
        # by the operations for images or gt only.
        self.geometric_transform = geometric_transform
        self.image_transform = image_transform
        self.gt_transform = gt_transform

    def __call__(self, image, gt):

        seed = np.random.randint(2147483647)

        random.seed(seed)  # 对transform进行随机变换
        torch.manual_seed(seed)  # 如果使用torch的随机数生成

        image = self.geometric_transform(image)
        image = self.image_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)

        gt = self.geometric_transform(gt)
        gt = self.gt_transform(gt)

        return image, gt


class MirrorPadding:
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, image):
        # convert it to np.array
        image = np.array(image)
        padded_image = cv2.copyMakeBorder(image, self.padding, self.padding, self.padding, self.padding,
                                          cv2.BORDER_REFLECT)
        # convert it back to PIL version
        padded_image = Image.fromarray(padded_image)
        return padded_image


geometric_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    MirrorPadding(60),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
    ),
    transforms.CenterCrop(224),

])

image_train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_vate_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gt_transform = transforms.ToTensor()

train_transform = DataAugmentationTransform(geometric_transform, image_train_transform, gt_transform)

vanilla_transform = DataAugmentationTransform(transforms.Resize((224,224)), image_vate_transform, gt_transform)
