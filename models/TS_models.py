import torch
import torch.nn as nn
from typing import Dict, Union, Optional
from torchvision.models.feature_extraction import create_feature_extractor

from models import _deeplab
from models import resnet
from models import utils

class AuxiHead(nn.Module):
    """
    Auxiliary head for intermediate feature prediction.
    """
    def __init__(self, in_channel: int):
        super(AuxiHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(in_channel // 2, in_channel // 4, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(size=(224, 224))
        self.conv3 = nn.Conv2d(in_channel // 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.conv3(x)
        return x

class Teacher(nn.Module):
    """
    Teacher model based on DeepLabV3+ with ResNet-101 backbone.
    """
    def __init__(self, flag: str = 'eval', num_classes: int = 1):
        super(Teacher, self).__init__()
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
        
        backbone = resnet.__dict__["resnet101"](
            pretrained=True,
            replace_stride_with_dilation=replace_stride_with_dilation)
        
        inplanes = 2048
        low_level_planes = 256
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        
        classifier = _deeplab.DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        
        # For intermediate features
        return_layers_for_resnet = {'layer2': 'layer2', 'layer3': 'layer3'}
        self.auxi_resnet = create_feature_extractor(backbone, return_nodes=return_layers_for_resnet)
        
        backbone_deeplab = utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.model = _deeplab.DeepLabV3(backbone_deeplab, classifier)
        
        return_layers_for_aspp = {'classifier.aspp': 'aspp'}
        self.auxi_aspp = create_feature_extractor(self.model, return_nodes=return_layers_for_aspp)
        
        self.auxihead1 = AuxiHead(in_channel=512)
        self.auxihead2 = AuxiHead(in_channel=1024)
        self.auxihead3 = AuxiHead(in_channel=256)
        
        # Freeze teacher weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.flag = flag

    def forward(self, x: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict]]:
        main_output = self.model(x)
        
        # Get auxiliary features
        resnet_feats = self.auxi_resnet(x)
        auxi_resnet1 = self.auxihead1(resnet_feats['layer2'])
        auxi_resnet2 = self.auxihead2(resnet_feats['layer3'])
        
        aspp_feats = self.auxi_aspp(x)
        auxi_aspp = self.auxihead3(aspp_feats['aspp'])
        
        auxi_dict = {'resnet1': auxi_resnet1, 'resnet2': auxi_resnet2, 'aspp': auxi_aspp}
        
        if self.flag == "train":
            return auxi_dict
        
        return {'main_output': main_output, 'auxi_dict': auxi_dict}

def load_resnet18_dilated():
    """
    Load ResNet-18 with specific dilation settings for segmentation.
    """
    model = resnet.__dict__["resnet18"](pretrained=True)
    # Modify layer3 and layer4 for segmentation
    for name, module in model.named_children():
        if name == 'layer4':
            module[0].conv1.stride = (1, 1)
            module[0].conv2.dilation = (2, 2)
            module[0].conv2.padding = (2, 2)
            if module[0].downsample:
                module[0].downsample[0].stride = (1, 1)
        elif name == 'layer3':
            module[0].conv1.stride = (1, 1)
            module[0].conv2.dilation = (2, 2)
            module[0].conv2.padding = (2, 2)
            if module[0].downsample:
                module[0].downsample[0].stride = (1, 1)
    return model

class Student(nn.Module):
    """
    Student model based on DeepLabV3+ with ResNet-18 backbone.
    """
    def __init__(self, flag: str = 'full', num_classes: int = 1):
        super(Student, self).__init__()
        aspp_dilate = [12, 24, 36]
        backbone = load_resnet18_dilated()
        
        inplanes = 512
        low_level_planes = 64
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        
        classifier = _deeplab.DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        
        return_layers_for_resnet = {'layer2': 'layer2', 'layer3': 'layer3'}
        self.auxi_resnet = create_feature_extractor(backbone, return_nodes=return_layers_for_resnet)
        
        backbone_deeplab = utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.model = _deeplab.DeepLabV3(backbone_deeplab, classifier)
        
        return_layers_for_aspp = {'classifier.aspp': 'aspp'}
        self.auxi_aspp = create_feature_extractor(self.model, return_nodes=return_layers_for_aspp)
        
        self.auxihead1 = AuxiHead(in_channel=128)
        self.auxihead2 = AuxiHead(in_channel=256)
        self.auxihead3 = AuxiHead(in_channel=256)
        self.flag = flag

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict]:
        main_output = self.model(x)
        
        if not self.training or self.flag == 'raw' or self.flag == 'eval':
            return main_output
        
        # During training in distillation mode
        resnet_feats = self.auxi_resnet(x)
        auxi_resnet1 = self.auxihead1(resnet_feats['layer2'])
        auxi_resnet2 = self.auxihead2(resnet_feats['layer3'])
        
        aspp_feats = self.auxi_aspp(x)
        auxi_aspp = self.auxihead3(aspp_feats['aspp'])
        
        auxi_dict = {'resnet1': auxi_resnet1, 'resnet2': auxi_resnet2, 'aspp': auxi_aspp}
        return {'main_output': main_output, 'auxi_dict': auxi_dict}
