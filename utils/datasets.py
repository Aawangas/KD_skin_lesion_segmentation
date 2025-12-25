import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from typing import Tuple, List, Optional, Callable

class SkinDataset(Dataset):
    """
    Standard dataset for skin lesion segmentation.
    """
    def __init__(self, img_dir: str, gt_dir: str, transform: Callable):
        self.transform = transform
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
            print(f"Warning: Directory not found: {img_dir} or {gt_dir}")
            self.image_paths = []
            self.gt_paths = []
        else:
            self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                     if f.lower().endswith(valid_extensions)])
            self.gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                                  if f.lower().endswith(valid_extensions)])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        gt = Image.open(self.gt_paths[idx]).convert("L")
        image, gt = self.transform(image, gt)
        return image, gt

    def __len__(self) -> int:
        return len(self.image_paths)

class TeacherDataset(Dataset):
    """
    Dataset that includes teacher model outputs for distillation.
    """
    def __init__(self, img_dir: str, gt_dir: str, transform: Callable, teacher_model: torch.nn.Module, device: str = 'cuda'):
        self.transform = transform
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                 if f.lower().endswith(valid_extensions)])
        self.gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)
                              if f.lower().endswith(valid_extensions)])
        self.teacher_model = teacher_model
        self.device = device

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        image_pil = Image.open(self.image_paths[idx]).convert("RGB")
        gt_pil = Image.open(self.gt_paths[idx]).convert("L")
        
        image, gt = self.transform(image_pil, gt_pil)
        
        self.teacher_model.eval()
        with torch.no_grad():
            # Teacher prediction
            teacher_output = self.teacher_model(image.unsqueeze(0).to(self.device))
            
            # Process outputs to CPU
            res = {
                'main_output': teacher_output['main_output'].squeeze(0).cpu(),
                'auxi_dict': {k: v.squeeze(0).cpu() for k, v in teacher_output['auxi_dict'].items()},
                'gt': gt.cpu()
            }
            
        return image, res

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
