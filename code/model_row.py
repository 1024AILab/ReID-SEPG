import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
import numpy as np
import math
import matplotlib.pyplot as plt
from FSIM import FSIM

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class att_resnet(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(att_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.SA = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        f = self.base.layer4(x)
        x = torch.mul(x, self.sigmoid(torch.mean(f, dim=1, keepdim=True)))
        f = torch.squeeze(self.base.avgpool(f))
        out, feat = self.classifier(f)
        return x, out, feat


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        x = self.classifier(x)
        return x, f


class classifier(nn.Module):
    def __init__(self, num_part, class_num):
        super(classifier, self).__init__()
        input_dim = 1024
        self.part = num_part
        self.l2norm = Normalize(2)
        for i in range(num_part):
            name = 'classifier_' + str(i)
            setattr(self, name, ClassBlock(input_dim, class_num))

    def forward(self, x, feat_all, out_all):
        start_point = len(feat_all)
        for i in range(self.part):
            name = 'classifier_' + str(i)
            cls_part = getattr(self, name)
            out_all[i + start_point], feat_all[i + start_point] = cls_part(torch.squeeze(x[:, :, i]))
            feat_all[i + start_point] = self.l2norm(feat_all[i + start_point])

        return feat_all, out_all


class embed_net(nn.Module):
    def __init__(self, class_num=395, part=12, arch='resnet50'):
        super(embed_net, self).__init__()

        self.part = part
        self.base_resnet = base_resnet(arch=arch)
        self.att_v = att_resnet(class_num)
        self.att_n = att_resnet(class_num)
        self.classifier = classifier(part, class_num)
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.seg=FSIM()

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1=self.seg(x1) #光谱增强
            x = torch.cat((x1, x2), 0)
            x = self.base_resnet(x)
            # temp=self.avgpool2(x)
            # temp=torch.reshape(temp,[112,-1])


            x1, x2 = torch.chunk(x, 2, 0)
            x1, out_v, feat_v = self.att_v(x1)
            x2, out_n, feat_n = self.att_n(x2)
            x = torch.cat((x1, x2), 0)

            feat_globe = torch.cat((feat_v, feat_n), 0)
            out_globe = torch.cat((out_v, out_n), 0)
            # temp=self.avgpool2(x)
            # temp=torch.reshape(out_globe,[112,-1])
        elif modal == 1:
            x = self.base_resnet(x1)
            x, _, _ = self.att_v(x)
        elif modal == 2:
            x = self.base_resnet(x2)
            x, _, _ = self.att_n(x)

        x = self.avgpool(x)
        feat = {}
        out = {}
        feat, out = self.classifier(x, feat, out)
        if self.training:
            return feat, out, feat_globe, out_globe
        else:
            for i in range(self.part):
                if i == 0:
                    featf = feat[i]
                else:
                    featf = torch.cat((featf, feat[i]), 1)
            return featf

from thop import profile
from torchstat import stat
inputs1 = torch.randn(2, 3, 384, 192)
print(inputs1.shape)
inputs2 = torch.randn(2, 3, 384, 192)
model=embed_net()

flops, params = profile(model, inputs=(inputs1,inputs2))
print("\nParmas:", round(params / 1.0e6, 3),"M")
print("\nFLOPs:", round(flops / 1.0e6, 3),"M")
