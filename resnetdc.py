# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:10:34 2024

@author: CVIS
"""

"""
# ResNet-D backbone with deep-stem
# Code Adapted from:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from einops import rearrange, repeat

#import collections

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict



__all__ = ['ResNet', 'resnet', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Basic Block for Resnet
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck Layer for Resnet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class layer_d(nn.Module):
    def __init__(self):
        super(layer_d, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNet(nn.Module):
    """
    Resnet
    """
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 128
        super(ResNet, self).__init__()
        # self.conv1 = nn.Sequential(
        #     conv3x3(3, 64, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     conv3x3(64, 64),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     conv3x3(64, 128))
        # self.bn1 = nn.BatchNorm2d(128)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        res = layer_d()
        resnet.layer0 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
        
        self.layer0 = resnet.layer0
        # nn.Sequential(
        #     conv3x3(3, 64, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     conv3x3(64, 64),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     conv3x3(64, 128),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        #self.layer0 = resnet.layer0
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.con_layer0 = nn.Sequential(
             conv3x3(256, 128),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace=True))
        self.con_layer1 = nn.Sequential(
             conv3x3(128, 64),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True))
        self.con_layer2 = nn.Sequential(
             conv3x3(256, 128),
             nn.BatchNorm2d(128),
             nn.ReLU(inplace=True))
        self.con_layer3 = nn.Sequential(
             conv3x3(512, 256),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def con_sum(self, x, b, n):
        x = rearrange(x, '(b n) ... -> b n ...', b=b, n=n)
        _, _, c, h, w = x.size()
        temp = torch.zeros_like(x)
        for img_id in range(b):
            for cam_id in range(n):
                if cam_id == 0:
                    temp[img_id, cam_id, :, :, :(w//2)] = x[img_id, 3, :, :, (w//2):]
                    temp[img_id, cam_id, :, :, (w//2):] = x[img_id, cam_id+1, :, :, :(w//2)]
                elif cam_id == 2:
                    temp[img_id, cam_id, :, :, :(w//2)] = x[img_id, 1, :, :, (w//2):]
                    temp[img_id, cam_id, :, :, (w//2):] = x[img_id, 5, :, :, :(w//2)]
                elif cam_id == 3:
                    temp[img_id, cam_id, :, :, :(w//2)] = x[img_id, 4, :, :, (w//2):]
                    temp[img_id, cam_id, :, :, (w//2):] = x[img_id, 0, :, :, :(w//2)]
                elif cam_id == 5:
                    temp[img_id, cam_id, :, :, :(w//2)] = x[img_id, 2, :, :, (w//2):]
                    temp[img_id, cam_id, :, :, (w//2):] = x[img_id, 4, :, :, :(w//2)]
                else:
                    temp[img_id, cam_id, :, :, :(w//2)] = x[img_id, cam_id-1, :, :, (w//2):]
                    temp[img_id, cam_id, :, :, (w//2):] = x[img_id, cam_id+1, :, :, :(w//2)]
        x_out = torch.cat((x, temp), 2)
        x_out = rearrange(x_out, 'b n ... -> (b n) ...', b=b, n=n)
        return x_out

    def forward(self, x, b, n):
        # x = self.conv1(x)##############camera change in cross_view_transformer/data/nuscenes_dataset.py
        # x = self.bn1(x)################input change in encoder.py and res_extract.py
        # x = self.relu(x)
        # x = self.maxpool(x)
        
        x0 = self.layer0(x)
        x0_cat =self.con_sum(x0, b, n)
        x0_con = self.con_layer0(x0_cat)

        x1 = self.layer1(x0_con)
        x1_cat =self.con_sum(x1, b, n)
        x1_con = self.con_layer1(x1_cat)
        
        x2 = self.layer2(x1_con)
        x2_cat =self.con_sum(x2, b, n)
        x2_con = self.con_layer2(x2_cat)

        
        x3 = self.layer3(x2_con)
        x3_cat =self.con_sum(x3, b, n)
        x3_con = self.con_layer3(x3_cat)

        
        x4 = self.layer4(x3_con)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return [x1, x2, x3, x4]


def resnet(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        checkpoint = torch.load("/data/song.wang/data/cross_view_transformers/res18_sfnet.pth")
        checkpoint_dict = checkpoint["state_dict"]
        pretrained_dict = OrderedDict()
        name_layer = "layer"
        for k, v in checkpoint_dict.items():
            if name_layer in k:
                name = k[7:]
                pretrained_dict[name]=v
        model.load_state_dict(pretrained_dict, strict=False)##################################strict=True
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./pretrained_models/resnet50-deep.pth", map_location='cpu'))
    return model



def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("./pretrained_models/resnet101-deep.pth",map_location='cpu'))
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
