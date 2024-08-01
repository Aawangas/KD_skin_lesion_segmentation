import torch
import torchvision
import torch.nn as nn


# This file is the summary of some commonly used loss functions.
# The content will be modified along the work.

class DiceLoss(nn.Module):
    def __init__(self, threshold, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, inputs, targets):
        # Apply sigmoid to the inputs
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()

        # Flatten the inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()

        # Compute Dice coefficient
        dice_coef = (2. * intersection + self.smooth) / (total + self.smooth)

        # Compute Dice loss
        dice_loss = 1 - dice_coef

        return dice_loss


class TeacherAuxiliaryLoss(nn.Module):
    def __init__(self):
        super(TeacherAuxiliaryLoss, self).__init__()
        # We have 3 detection head to be trained, we should make use of the forwad function
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.BCEWithLogitsLoss()
        self.loss3 = nn.BCEWithLogitsLoss()

    def forward(self, output: dict, gt):
        loss1 = self.loss1(output['resnet1'], gt)
        loss2 = self.loss2(output['resnet2'], gt)
        loss3 = self.loss3(output['aspp'], gt)

        loss = loss1 + loss2 + loss3

        return loss


class FeatureDistillLoss(nn.Module):
    def __init__(self, alpha):
        super(FeatureDistillLoss, self).__init__()
        self.alpha = alpha
        self.loss_resnet1_1 = nn.BCEWithLogitsLoss()
        self.loss_resnet1_2 = nn.BCEWithLogitsLoss()
        self.loss_resnet2_1 = nn.BCEWithLogitsLoss()
        self.loss_resnet2_2 = nn.BCEWithLogitsLoss()
        self.loss_aspp_1 = nn.BCEWithLogitsLoss()
        self.loss_aspp_2 = nn.BCEWithLogitsLoss()

    def forward(self, output, auxi_image, gt):
        # output: tensor of size (1,1,224,224), if we want to convert that into
        # probability, we should use a sigmoid function
        # auxi_image: similar to output image
        # gt: binary image
        # note that output and auxi_image are both dictionary
        loss1 = self.loss_resnet1_1(output['resnet1'], torch.sigmoid(auxi_image['resnet1']))
        loss2 = self.loss_resnet1_2(output['resnet1'], gt.unsqueeze(1))
        loss3 = self.loss_resnet2_1(output['resnet2'], torch.sigmoid(auxi_image['resnet2']))
        loss4 = self.loss_resnet2_2(output['resnet2'], gt.unsqueeze(1))
        loss5 = self.loss_aspp_1(output['aspp'], torch.sigmoid(auxi_image['aspp']))
        loss6 = self.loss_aspp_2(output['aspp'], gt.unsqueeze(1))
        return self.alpha * (loss1 + loss3 + loss5) + (loss2 + loss4 + loss6)


class StudentDistillLoss(nn.Module):
    def __init__(self, flag):
        super(StudentDistillLoss, self).__init__()
        self.feature_loss = FeatureDistillLoss(0.05)
        self.logits_loss = nn.BCEWithLogitsLoss()
        self.label_loss = nn.BCEWithLogitsLoss()
        self.flag = flag
        pass

    def forward(self, output, gt):
        # output should be a dict: {'main_output':_, 'auxi_dict':_}
        # The distill process should be governed by:
        # feature auxiliary image, and the ground_truth
        # The final distill should be governed by final output of teacher model and gt
        if self.flag == 'logits':
            logits_loss = self.logits_loss(output['main_output'], torch.sigmoid(gt['main_output']))
            label_loss = self.label_loss(output['main_output'], gt['gt'].unsqueeze(1))
            return logits_loss + label_loss
        if self.flag == 'full':
            feature_loss = self.feature_loss(output['auxi_dict'], gt['auxi_dict'],gt['gt'])
            logits_loss = self.logits_loss(output['main_output'], torch.sigmoid(gt['main_output']))
            label_loss = self.label_loss(output['main_output'], gt['gt'].unsqueeze(1))

            return feature_loss + logits_loss + label_loss

