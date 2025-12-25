import torch
import torch.nn as nn
from typing import Dict, Union


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for binary segmentation.
    Note: If threshold is applied, the loss becomes non-differentiable.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)

        # Optional: thresholding makes it non-differentiable, usually used for evaluation
        if self.threshold is not None:
            inputs = (inputs > self.threshold).float()

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        dice_coef = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice_coef


class TeacherAuxiliaryLoss(nn.Module):
    """
    Loss for training the teacher model with multiple auxiliary heads.
    """

    def __init__(self):
        super(TeacherAuxiliaryLoss, self).__init__()
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.BCEWithLogitsLoss()
        self.loss3 = nn.BCEWithLogitsLoss()

    def forward(
        self, output: Dict[str, torch.Tensor], gt: torch.Tensor
    ) -> torch.Tensor:
        loss1 = self.loss1(output["resnet1"], gt)
        loss2 = self.loss2(output["resnet2"], gt)
        loss3 = self.loss3(output["aspp"], gt)
        return loss1 + loss2 + loss3


class FeatureDistillLoss(nn.Module):
    """
    Loss for distilling intermediate features from teacher to student.
    """

    def __init__(self, alpha: float = 0.05):
        super(FeatureDistillLoss, self).__init__()
        self.alpha = alpha
        self.loss_resnet1_1 = nn.BCEWithLogitsLoss()
        self.loss_resnet1_2 = nn.BCEWithLogitsLoss()
        self.loss_resnet2_1 = nn.BCEWithLogitsLoss()
        self.loss_resnet2_2 = nn.BCEWithLogitsLoss()
        self.loss_aspp_1 = nn.BCEWithLogitsLoss()
        self.loss_aspp_2 = nn.BCEWithLogitsLoss()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        auxi_image: Dict[str, torch.Tensor],
        gt: torch.Tensor,
    ) -> torch.Tensor:
        loss1 = self.loss_resnet1_1(
            output["resnet1"], torch.sigmoid(auxi_image["resnet1"])
        )
        loss2 = self.loss_resnet1_2(output["resnet1"], gt.unsqueeze(1))
        loss3 = self.loss_resnet2_1(
            output["resnet2"], torch.sigmoid(auxi_image["resnet2"])
        )
        loss4 = self.loss_resnet2_2(output["resnet2"], gt.unsqueeze(1))
        loss5 = self.loss_aspp_1(output["aspp"], torch.sigmoid(auxi_image["aspp"]))
        loss6 = self.loss_aspp_2(output["aspp"], gt.unsqueeze(1))
        return self.alpha * (loss1 + loss3 + loss5) + (loss2 + loss4 + loss6)


class StudentDistillLoss(nn.Module):
    """
    Combined distillation loss for the student model.
    """

    def __init__(self, flag: str = "full"):
        super(StudentDistillLoss, self).__init__()
        self.feature_loss_fn = FeatureDistillLoss(0.05)
        self.logits_loss_fn = nn.BCEWithLogitsLoss()
        self.label_loss_fn = nn.BCEWithLogitsLoss()
        self.flag = flag

    def forward(
        self,
        output: Dict[str, Union[torch.Tensor, Dict]],
        gt: Dict[str, Union[torch.Tensor, Dict]],
    ) -> torch.Tensor:
        if self.flag == "logits":
            logits_loss = self.logits_loss_fn(
                output["main_output"], torch.sigmoid(gt["main_output"])
            )
            label_loss = self.label_loss_fn(
                output["main_output"], gt["gt"].unsqueeze(1)
            )
            return logits_loss + label_loss

        if self.flag == "full":
            feature_loss = self.feature_loss_fn(
                output["auxi_dict"], gt["auxi_dict"], gt["gt"]
            )
            logits_loss = self.logits_loss_fn(
                output["main_output"], torch.sigmoid(gt["main_output"])
            )
            label_loss = self.label_loss_fn(
                output["main_output"], gt["gt"].unsqueeze(1)
            )
            return feature_loss + logits_loss + label_loss

        return self.label_loss_fn(output["main_output"], gt["gt"].unsqueeze(1))
