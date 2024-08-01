import torch
import numpy as np
import time
import json


# This part is designed to implement the training process, validation and
# testing. More specifically, include learning rate optimization and early stopping
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = 0
    # At this moment, img is the tensor of the size (batch_size,3,size,size)
    for batch, (img, gt) in enumerate(dataloader):
        img = img.to("cuda")
        if isinstance(gt, torch.Tensor):
            gt = gt.to('cuda')
        elif isinstance(gt, dict):
            gt['main_output'] = gt['main_output'].to('cuda')
            gt['gt'] = gt['gt'].to('cuda')
            for key in gt['auxi_dict']:
                gt['auxi_dict'][key] = gt['auxi_dict'][key].to('cuda')
        pred = model(img)
        loss = loss_fn(pred, gt)
        train_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0 and batch != 0:
            loss, current = loss.item(), batch * dataloader.batch_size
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")
    train_loss /= len(dataloader)
    return train_loss


def validate(dataloader, model, loss_fn):
    model.eval()
    validate_loss = 0
    with torch.no_grad():
        for batch, (img, gt) in enumerate(dataloader):
            img = img.to("cuda")
            if isinstance(gt, torch.Tensor):
                gt = gt.to('cuda')
            elif isinstance(gt, dict):
                gt['main_output'] = gt['main_output'].to('cuda')
                gt['gt'] = gt['gt'].to('cuda')
                for key in gt['auxi_dict']:
                    gt['auxi_dict'][key] = gt['auxi_dict'][key].to('cuda')
            pred = model(img)
            loss = loss_fn(pred, gt)
            validate_loss += loss

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * dataloader.batch_size
            print(f"validate_loss: {loss:>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

    validate_loss /= len(dataloader)
    return validate_loss


def training_process_with_validation(num_epoches, train_dataloader, validate_dataloader, model, loss_fn_1, loss_fn_2,
                                     optimizer,scheduler, model_path, loss_path):
    # best_lost is the smallest validation loss so far
    best_loss = 0
    # gl is the generalization: relative increase of error
    gl = 0

    actual_start = time.time()

    loss_dict = {
        "train_loss": [],
        "validate_loss": [],
        "train_time": [],
        "validate_time": []
    }

    for epoch in range(0, num_epoches):
        start_time = time.time()
        train_loss = train(train_dataloader, model, loss_fn_1, optimizer)
        torch.cuda.empty_cache()
        print("In epoch", epoch + 1, "\n:training loss is",
              train_loss, "\nTime used is: ", time.time() - start_time)
        loss_dict["train_loss"].append(train_loss.item())
        loss_dict["train_time"].append(time.time() - start_time)
        start_time = time.time()
        validate_loss = validate(validate_dataloader, model, loss_fn_2)
        torch.cuda.empty_cache()
        print("In epoch", epoch + 1, ":\nvalidate loss is",
              validate_loss, "\nTime used is: ", time.time() - start_time)
        loss_dict["validate_loss"].append(validate_loss.item())
        loss_dict["validate_time"].append(time.time() - start_time)

        print("Total time so far is: ", time.time() - actual_start)

        with open(loss_path, 'w') as f:
            json.dump(loss_dict, f)

        scheduler.step(validate_loss.item())

        if epoch == 0:
            best_loss = validate_loss
        if validate_loss < best_loss:
            best_loss = validate_loss
            torch.save(model.state_dict(), model_path)


def metric_compute(pred, gt, threshold):
    pred = (pred > threshold).float()
    TP = (gt * pred).sum().to(torch.float32)
    FP = ((1 - gt) * pred).sum().to(torch.float32)
    FN = (gt * (1 - pred)).sum().to(torch.float32)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)

    return {
        "iou": iou.item(),
        "dice": dice.item()
    }


def test(dataloader, model):
    # The final is to return the mean metrics of different type
    all_metrics = {
        "iou": [],
        "dice": []
    }
    for batch, (image, gt) in enumerate(dataloader):
        with torch.no_grad():
            image = image.to("cuda")
            gt = gt.to("cuda")
            pred = model(image)
            metric = metric_compute(pred, gt, 0.5)
            for key in all_metrics:
                all_metrics[key].append(metric[key])

    average_metrics = {key: np.mean(all_metrics[key]) for key in all_metrics}
    print(average_metrics)
