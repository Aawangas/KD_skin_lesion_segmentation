import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from models import modeling
from models.ts_models import Teacher
from utils import datasets, augmentations, losses, trainer, config

def get_args():
    parser = argparse.ArgumentParser(description="Train Teacher Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone in config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    return parser.parse_args()

def load_model(backbone_name, num_classes=1):
    if backbone_name == "resnet":
        model = modeling.deeplabv3plus_resnet101(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
    elif backbone_name == "xception":
        model = modeling.deeplabv3plus_xception(num_classes=num_classes, output_stride=8, pretrained_backbone=True)
    elif backbone_name == "teacher_auxi":
        model = Teacher(flag='eval')
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    return model

def main():
    args = get_args()
    cfg = config.load_config(args.config)
    
    # Override config with args if provided
    backbone = args.backbone or cfg['teacher']['backbone']
    batch_size = args.batch_size or cfg['data']['batch_size']
    epochs = args.epochs or cfg['train']['epochs']
    device = cfg['train']['device']
    
    # Set up data loaders
    train_transform = augmentations.get_train_transform()
    val_transform = augmentations.get_val_transform()
    
    train_dataset = datasets.SkinDataset(cfg['data']['train_img_dir'], cfg['data']['train_gt_dir'], train_transform)
    val_dataset = datasets.SkinDataset(cfg['data']['val_img_dir'], cfg['data']['val_gt_dir'], val_transform)
    
    train_loader = datasets.get_dataloader(train_dataset, batch_size)
    val_loader = datasets.get_dataloader(val_dataset, batch_size, shuffle=False)
    
    # Load model
    model = load_model(backbone, num_classes=cfg['teacher']['num_classes'])
    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cfg['train']['patience'], verbose=True)
    
    # Loss functions
    if backbone == "teacher_auxi":
        train_loss_fn = losses.TeacherAuxiliaryLoss()
        val_loss_fn = losses.TeacherAuxiliaryLoss()
    else:
        train_loss_fn = nn.BCEWithLogitsLoss()
        val_loss_fn = nn.BCEWithLogitsLoss()
    
    # Training
    os.makedirs(cfg['train']['save_path'], exist_ok=True)
    model_path = os.path.join(cfg['train']['save_path'], f"teacher_{backbone}.pth")
    loss_path = os.path.join(cfg['train']['save_path'], f"teacher_{backbone}.json")
    
    trainer.training_loop(
        epochs, train_loader, val_loader, model, 
        train_loss_fn, val_loss_fn, optimizer, scheduler, 
        device, model_path, loss_path
    )

if __name__ == "__main__":
    main()
