# -*- coding: utf-8 -*-
# @Time    : 2018/9/19 17:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : decoder.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from .Resnet101 import resnet101
from .efficientnet import EfficientNet, get_model_params, round_filters
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import sys

sys.path.append(os.path.abspath('..'))

from .encoder import Encoder, EfficientEncoder


class Decoder(nn.Module):
    def __init__(self, class_num, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(48, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)

        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        # x_4 = F.interpolate(x, size=low_level_feature.size()[2:3], mode='bilinear', align_corners=True)
        x_4 = F.upsample(x, low_level_feature.size()[2:], mode='bilinear')
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self, class_num, output_stride=16, pretrained=True, bn_momentum=0.1, freeze_bn=False):
        super(DeepLab, self).__init__()
        self.bn_momentum = bn_momentum
        self.Resnet101 = resnet101(bn_momentum, pretrained)
        self.encoder = Encoder(bn_momentum, output_stride)
        self.decoder = Decoder(class_num, bn_momentum)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low_level_features = self.Resnet101(input)

        x = self.encoder(x)
        predict = self.decoder(x, low_level_features)
        # output = F.interpolate(predict, size=input.size()[2:3], mode='bilinear', align_corners=True)
        output = F.upsample(predict, input.size()[2:], mode='bilinear')
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()

    def change_o_stride(self, output_stride):
        new_encoder = Encoder(self.bn_momentum, output_stride)
        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder
        print("output stride changed to " + str(output_stride) + " successfully!")


class EfficientDecoder(nn.Module):
    def __init__(self, comp_coeff, class_num, bn_momentum=0.1):
        super(EfficientDecoder, self).__init__()
        blocks_args, global_params = get_model_params('efficientnet-b' + str(comp_coeff), False)
        in_channels = round_filters(blocks_args[0].output_filters, global_params)
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(8, momentum=bn_momentum)
        self.relu = nn.ReLU()
        # self.conv2 = SeparableConv2d(304, 256, kernel_size=3)
        # self.conv3 = SeparableConv2d(256, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(264, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(256, momentum=bn_momentum)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, class_num, kernel_size=1)

        self._init_weight()

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        # x_4 = F.interpolate(x, size=low_level_feature.size()[2:3], mode='bilinear', align_corners=True)
        x_4 = F.upsample(x, low_level_feature.size()[2:], mode='bilinear')
        x_4_cat = torch.cat((x_4, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)

        return x_4_cat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EfficientDeepLab(nn.Module):
    def __init__(self, comp_coeff, class_num, output_stride=16, pretrained=True, bn_momentum=0.1, freeze_bn=False):
        super(EfficientDeepLab, self).__init__()
        self.bn_momentum = bn_momentum
        self.comp_coeff = comp_coeff
        # self.Resnet101 = resnet101(bn_momentum, pretrained)
        if pretrained:
            self.effnet = EfficientNet.from_pretrained('efficientnet-b' + str(comp_coeff), num_classes=class_num)
        else:
            self.effnet = EfficientNet.from_name('efficientnet-b' + str(comp_coeff),
                                                 override_params={'num_classes': class_num})
        self.encoder = EfficientEncoder(comp_coeff=comp_coeff, bn_momentum=bn_momentum, output_stride=output_stride)
        self.decoder = EfficientDecoder(comp_coeff=comp_coeff, class_num=class_num, bn_momentum=bn_momentum)
        if freeze_bn:
            self.freeze_bn()
            print("freeze bacth normalization successfully!")

    def forward(self, input):
        x, low_level_features = self.effnet(input)

        x = self.encoder(x)
        predict = self.decoder(x, low_level_features)
        # output = F.interpolate(predict, size=input.size()[2:3], mode='bilinear', align_corners=True)
        output = F.upsample(predict, input.size()[2:], mode='bilinear')
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()

    def change_o_stride(self, output_stride):
        new_encoder = Encoder(self.bn_momentum, output_stride)
        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder
        print("output stride changed to " + str(output_stride) + " successfully!")


if __name__ == "__main__":
    model = DeepLab(output_stride=16, class_num=21, pretrained=False, freeze_bn=False)
    model.eval()
    # print(model)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # summary(model, (3, 513, 513))
    # for m in model.named_modules():
    for m in model.modules():
        if isinstance(m, SynchronizedBatchNorm2d):
            print(m)
