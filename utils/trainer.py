import torch
import numpy as np
import time
import json
import logging
from tqdm import tqdm
from typing import Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch, (img, gt) in enumerate(progress_bar):
        img = img.to(device)
        
        if isinstance(gt, torch.Tensor):
            gt = gt.to(device)
        elif isinstance(gt, dict):
            gt['main_output'] = gt['main_output'].to(device)
            gt['gt'] = gt['gt'].to(device)
            for key in gt['auxi_dict']:
                gt['auxi_dict'][key] = gt['auxi_dict'][key].to(device)
        
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, gt)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
    return total_loss / len(dataloader)

def validate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, gt in tqdm(dataloader, desc="Validating"):
            img = img.to(device)
            
            if isinstance(gt, torch.Tensor):
                gt = gt.to(device)
            elif isinstance(gt, dict):
                gt['main_output'] = gt['main_output'].to(device)
                gt['gt'] = gt['gt'].to(device)
                for key in gt['auxi_dict']:
                    gt['auxi_dict'][key] = gt['auxi_dict'][key].to(device)
            
            pred = model(img)
            loss = loss_fn(pred, gt)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def training_loop(
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    train_loss_fn: Callable,
    val_loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    model_path: str,
    loss_log_path: str
):
    best_loss = float('inf')
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(train_loader, model, train_loss_fn, optimizer, device)
        val_loss = validate(val_loader, model, val_loss_fn, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model to {model_path}")
            
        with open(loss_log_path, 'w') as f:
            json.dump(history, f)
            
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.2f} minutes")

def compute_metrics(pred, gt, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Ensure gt is float and on same device
    gt = gt.float().to(pred.device)
    
    tp = (gt * pred).sum().item()
    fp = ((1 - gt) * pred).sum().item()
    fn = (gt * (1 - pred)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}

def test(dataloader, model, device):
    model.eval()
    all_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}
    
    with torch.no_grad():
        for image, gt in tqdm(dataloader, desc="Testing"):
            image = image.to(device)
            gt = gt.to(device)
            pred = model(image)
            
            metrics = compute_metrics(pred, gt)
            for k, v in metrics.items():
                all_metrics[k].append(v)
                
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    logger.info(f"Test Metrics: {avg_metrics}")
    return avg_metrics
