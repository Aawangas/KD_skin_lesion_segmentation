import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from models.ts_models import Teacher, Student
from utils import datasets, augmentations, losses, trainer, config

def get_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation for Skin Lesion Segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=['distill', 'raw', 'test'], default='distill', help="Training mode")
    parser.add_argument("--teacher_weights", type=str, help="Path to teacher weights")
    parser.add_argument("--student_weights", type=str, help="Path to student weights (for testing)")
    return parser.parse_args()

def load_teacher(weight_path, device):
    teacher = Teacher('eval')
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        teacher.load_state_dict(state_dict)
        print(f"Loaded teacher weights from {weight_path}")
    else:
        print(f"Warning: Teacher weight path {weight_path} not found.")
    return teacher.to(device)

def run_distill(cfg, args):
    device = cfg['train']['device']
    batch_size = cfg['data']['batch_size']
    
    # Load teacher for distillation
    teacher_path = args.teacher_weights or cfg['teacher']['weight_path']
    teacher = load_teacher(teacher_path, device)
    
    # Transforms
    train_transform = augmentations.get_train_transform()
    val_transform = augmentations.get_val_transform()
    
    # Datasets
    train_dataset = datasets.TeacherDataset(
        cfg['data']['train_img_dir'], cfg['data']['train_gt_dir'], 
        train_transform, teacher, device
    )
    val_dataset = datasets.SkinDataset(
        cfg['data']['val_img_dir'], cfg['data']['val_gt_dir'], 
        val_transform
    )
    
    train_loader = datasets.get_dataloader(train_dataset, batch_size)
    val_loader = datasets.get_dataloader(val_dataset, batch_size, shuffle=False)
    
    # Student model
    model = Student().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cfg['train']['patience'])
    
    # Distillation Loss
    loss_fn = losses.StudentDistillLoss(flag='full')
    val_loss_fn = losses.DiceLoss(threshold=0.5)
    
    # Paths
    os.makedirs(cfg['train']['save_path'], exist_ok=True)
    model_path = os.path.join(cfg['train']['save_path'], "student_distill.pth")
    loss_path = os.path.join(cfg['train']['save_path'], "student_distill.json")
    
    trainer.training_loop(
        cfg['train']['epochs'], train_loader, val_loader, model,
        loss_fn, val_loss_fn, optimizer, scheduler,
        device, model_path, loss_path
    )

def run_raw_student(cfg):
    device = cfg['train']['device']
    batch_size = cfg['data']['batch_size']
    
    train_transform = augmentations.get_train_transform()
    val_transform = augmentations.get_val_transform()
    
    train_dataset = datasets.SkinDataset(cfg['data']['train_img_dir'], cfg['data']['train_gt_dir'], train_transform)
    val_dataset = datasets.SkinDataset(cfg['data']['val_img_dir'], cfg['data']['val_gt_dir'], val_transform)
    
    train_loader = datasets.get_dataloader(train_dataset, batch_size)
    val_loader = datasets.get_dataloader(val_dataset, batch_size, shuffle=False)
    
    model = Student(flag='raw').to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=cfg['train']['patience'])
    
    loss_fn = losses.DiceLoss(threshold=None) # Raw training usually uses BCE or soft Dice
    val_loss_fn = losses.DiceLoss(threshold=0.5)
    
    os.makedirs(cfg['train']['save_path'], exist_ok=True)
    model_path = os.path.join(cfg['train']['save_path'], "student_raw.pth")
    loss_path = os.path.join(cfg['train']['save_path'], "student_raw.json")
    
    trainer.training_loop(
        cfg['train']['epochs'], train_loader, val_loader, model,
        loss_fn, val_loss_fn, optimizer, scheduler,
        device, model_path, loss_path
    )

def run_test(cfg, args):
    device = cfg['train']['device']
    val_transform = augmentations.get_val_transform()
    test_dataset = datasets.SkinDataset(cfg['data']['test_img_dir'], cfg['data']['test_gt_dir'], val_transform)
    test_loader = datasets.get_dataloader(test_dataset, cfg['data']['batch_size'], shuffle=False)
    
    model = Student(flag='eval').to(device)
    model_path = args.student_weights or os.path.join(cfg['train']['save_path'], "student_distill.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded student weights from {model_path}")
        trainer.test(test_loader, model, device)
    else:
        print(f"Error: Student weights not found at {model_path}")

def main():
    args = get_args()
    cfg = config.load_config(args.config)
    
    if args.mode == 'distill':
        run_distill(cfg, args)
    elif args.mode == 'raw':
        run_raw_student(cfg)
    elif args.mode == 'test':
        run_test(cfg, args)

if __name__ == "__main__":
    main()
