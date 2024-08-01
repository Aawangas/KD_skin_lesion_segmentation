from models import modeling
from PIL import Image
import torch
import matplotlib.pyplot as plt
from useful import Dataset_loader, Training_testing, DataAugmentation
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.TS_models import Teacher, Student
import useful.Loss_function as Loss_function
def load_model(model_backnone):
    # Set num_class to be 1, to obtain the binary segmentation
    if model_backnone == "resnet":
        model = modeling.deeplabv3plus_resnet101(num_classes=1, output_stride=8, pretrained_backbone=True)

    elif model_backnone == "xception":
        model = modeling.deeplabv3plus_xception(num_classes=1, output_stride=8, pretrained_backbone=True)
    return model


def load_dataloader():
    train_transform = DataAugmentation.train_transform
    validate_test_transform = DataAugmentation.vanilla_transform
    # We are supposed to have 3 different datasets, but still, we load all data at the same time
    train_dataset = Dataset_loader.SkinDataset("",
                                               "",
                                               train_transform)
    validate_dataset = Dataset_loader.SkinDataset("",
                                                  "",
                                                  validate_test_transform)
    test_dataset = Dataset_loader.SkinDataset("",
                                              "",
                                              validate_dataset)
    train_dataloader = Dataset_loader.get_dataloader(train_dataset, 20)
    validate_dataset = Dataset_loader.get_dataloader(validate_dataset, 20)
    test_dataloader = Dataset_loader.get_dataloader(test_dataset, 20)

    return train_dataloader, validate_dataset, test_dataloader


def train():
    model = load_model("resnet")
    model = model.to("cuda")

    train_dataloader, validate_dataloader, test_dataloader = load_dataloader()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()
    model_path = "model.pth"
    loss_path = "model.json"
    Training_testing.training_process_with_validation(200, train_dataloader, validate_dataloader,
                                                      model, loss_fn, loss_fn, optimizer, scheduler, model_path,
                                                      loss_path)


def test(model_path):
    _, _, test_dataloader = load_dataloader()
    model = load_model("resnet")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to("cuda")
    Training_testing.test(test_dataloader, model)


def visualize_batch(dataloader, output_path='DataAugmentation_vanilla_visualization.png', num_images=6):
    # 获取一批数据
    images, labels = next(iter(dataloader))

    # 确保我们至少有 num_images 张图片
    num_images = min(num_images, images.shape[0])

    # 创建一个 5x2 的子图网格
    fig, axes = plt.subplots(3, 4, figsize=(20, 25))

    for i in range(num_images):
        # 获取图像和标签
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        label = labels[i].cpu().numpy()
        # 标准化逆操作
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        print("Images min value:", images.min())
        print("Images max value:", images.max())
        print("Ground truths min value:", label.min())
        print("Ground truths max value:", label.max())

        row = i // 2
        col = (i % 2) * 2

        axes[row, col].imshow(image)
        axes[row, col].set_title(f'Image {i + 1}')
        axes[row, col].axis('off')

        if len(label.shape) == 3:
            label = label.transpose((1, 2, 0))
        axes[row, col + 1].imshow(label, cmap='gray')
        axes[row, col + 1].set_title(f'Label {i + 1}')
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Batch visualization saved to {output_path}")


def LearningRateFind():
    dataloader, _, _ = load_dataloader()
    images, labels = next(iter(dataloader))
    images, labels = images.to("cuda"), labels.to("cuda")
    model = load_model("resnet")
    model = model.to("cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-7)

    lr = 1e-7
    end_lr = 10
    num_iter = 100
    beta = 0.98

    losses = []
    lrs = []
    avg_loss = 0.0
    best_loss = float('inf')

    for i in range(num_iter):
        optimizer.param_groups[0]['lr'] = lr
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # 平滑损失
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        # 记录学习率和损失
        lrs.append(lr)
        losses.append(smoothed_loss)

        # 动态调整学习率
        lr *= (end_lr / 1e-7) ** (1 / num_iter)

        # 停止条件
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if smoothed_loss > 4 * best_loss:
            break

    n = len(losses)
    change = np.zeros(n)
    change[0] = 0
    for i in range(1, n):
        change[i] = losses[i] - losses[i-1]
    change[n-1] = 0
    index = change.argmin()
    print(index)

def load_teacher(flag):
    teacher = Teacher(flag)
    weight_path = "/Users/wanganbang/Documents/GitHub/DLSkinLesionSeg/KnowledgeDistillation/models/model_files/teacher1.pth"
    state_dict = torch.load(weight_path, map_location='cpu')
    teacher.load_state_dict(state_dict)
    return teacher

def train_teacher_auxi():
    model = load_teacher()
    model = model.to("cuda")

    train_dataloader, validate_dataloader, test_dataloader = load_dataloader()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_fn = Loss_function.TeacherAuxiliaryLoss()
    model_path = "/home/awangas/Deeplabv3+/model_files/dlv3p1.pth"
    loss_path = "/home/awangas/Deeplabv3+/model_files/dlv3p1.json"
    Training_testing.training_process_with_validation(200, train_dataloader, validate_dataloader,
                                                      model, loss_fn, loss_fn, optimizer, scheduler, model_path,
                                                      loss_path)

