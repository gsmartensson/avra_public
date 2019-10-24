#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual Attention Network module - feature extractor in AVRA
"""
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttentionNet(nn.Module):
    '''
    Convolutional part of AVRA, that inputs a single 2D MRI slice and outputs a 
    flattened vector with relevant features extracted.
    
    Architecture based on "Residual Attention Network for Image Classification" by Wang et al. (2017)
    https://arxiv.org/abs/1704.06904
    '''
    def __init__(self, z=1):
        super(ResidualAttentionNet, self).__init__()
        
        # number of output filters from each block
        num_filters = [8,16,32,64,128]

        k=0 # block number counter
        
        conv1 = nn.Sequential(
            nn.Conv2d(z, num_filters[k], kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(num_filters[k]),
            nn.ReLU(inplace=True)
        )
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resblock1 = ResidualModule(num_filters[k], num_filters[k+1])
        k+=1

        attention_module1 = AttentionModule(num_filters[k], num_filters[k], stage=1)
        resblock2 = ResidualModule(num_filters[k], num_filters[k+1], stride=2)
        k+=1

        attention_module2 = AttentionModule(num_filters[k], num_filters[k], stage=2)
        resblock3 = ResidualModule(num_filters[k], num_filters[k+1], stride=2)
        k+=1
        
        attention_module3 = AttentionModule(num_filters[k], num_filters[k], stage=3)
        resblock4 = ResidualModule(num_filters[k], num_filters[k+1], stride=2)
        
        k+=1
        resblock5= ResidualModule(num_filters[k], num_filters[k])
        resblock6 = ResidualModule(num_filters[k], num_filters[k])
        avgpoolblock = nn.Sequential(
            nn.BatchNorm2d(num_filters[k]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=1)
        )

        self.features=nn.Sequential(
                conv1,maxpool,
                resblock1,attention_module1,
                resblock2,attention_module2,
                resblock3,attention_module3,
                resblock4,resblock5,resblock6,
                avgpoolblock
                      )
        
    def forward(self, x):
        out = self.features(x)
        
        # Flatten before passing to RNN 
        out = out.view(out.size(0), -1)        
        return out


class ResidualModule(nn.Module):
    '''
    A residual module, used in the Residual Attention Network.
    '''
    def __init__(self, inplanes, planes, stride=1):
        super(ResidualModule, self).__init__()
        
        planes_4 = int(planes/4) # bottlenecking
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inplanes,planes_4, kernel_size=1, stride=1, bias = False)
        
        self.bn2 = nn.BatchNorm2d(planes_4)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(planes_4, planes_4, kernel_size=3, stride=stride, padding = 1, bias = False)
        
        self.bn3 = nn.BatchNorm2d(planes_4)
        self.relu3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(planes_4, planes, kernel_size=1, stride=1, bias = False)
        
        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias = False)
        # downsampling?
        self.downsample = (self.inplanes != self.planes) or (self.stride !=1 )
    def forward(self, x):

        residual = x
        out = self.bn1(x)
        out1 = self.relu1(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.conv4(out1)
        out += residual
        return out
    
class AttentionModule(nn.Module):
    '''
    Code for generating Attention Module stage 1, 2, or 3
    '''
    def __init__(self, in_planes, out_planes, stage=1):
        super(AttentionModule, self).__init__()
        # p=1, r=1,t=2, as defined in original paper by Wang et al.
        self.stage=stage # 1,2,3, TODO: assert
        
        self.res1 = ResidualModule(in_planes, out_planes)

        self.trunk_branch = nn.Sequential(# i.e. "t=2"
            ResidualModule(in_planes, out_planes),
            ResidualModule(in_planes, out_planes)
         )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.stage<3:
            self.block1 = ResidualModule(in_planes, out_planes)
            self.skip1 = ResidualModule(in_planes, out_planes)
            self.block5 = ResidualModule(in_planes, out_planes)
            if self.stage==1:
                self.block2 = ResidualModule(in_planes, out_planes)
                self.skip2 = ResidualModule(in_planes, out_planes)
            
                self.block4 = ResidualModule(in_planes, out_planes)

        self.block3 = nn.Sequential( # middle block, included in all stages
            ResidualModule(in_planes, out_planes),
            ResidualModule(in_planes, out_planes)
        )
  
        self.block_sigmoid = nn.Sequential( # conv1x1 + sigmoid
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )

        self.block6 = ResidualModule(in_planes, out_planes)
    def upsample(self,x):
        # upsample tensor with a factor 2
        return F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
    def forward(self, x):
        x = self.res1(x) # H
        trunk_branch = self.trunk_branch(x) # H

        # SOFT MASK BRANCH
        if self.stage<3:
            # downsample 1
            x = self.maxpool(x) # H/2        
            x = self.block1(x)# H/2
            skip1 = self.skip1(x) # H/2, passes through 1 res unit        
            if self.stage==1:
                # downsample 2
                x = self.maxpool(x) # H/4
                x = self.block2(x) # H/4
                skip2 = self.skip2(x) # H/4, passes through 1 res unit
        
        # downsample 3
        x = self.maxpool(x) # H/8
        x = self.block3(x) # H/8
        
        if self.stage==1:
            # upsample 1
            x = self.upsample(x)
            x = x + skip2 # H/4
            x = self.block4(x) # H/4
        if self.stage<3:
            # upsample 2
            x = self.upsample(x)
            x = x + skip1 # H/2
            x = self.block5(x) # H/2
        
        # upsample 3
        x = self.upsample(x)
        mask = self.block_sigmoid(x) # H
        
        # merging trunk branc and soft mask branch
        x = (1 + mask) * trunk_branch # H
        x = x + trunk_branch # H
                
        out_last = self.block6(x) # H
        return out_last

def conv_block(in_planes, out_planes, bigblock,convxd,norm,pooling,fs=3,stride=1,relu=nn.ReLU):
    # conv3->bn3->relu->conv3->bn3->relu->maxpool3
    if bigblock:
        block = nn.Sequential(
            convxd(in_planes, out_planes, fs, 1, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            convxd(out_planes, out_planes, fs, 1, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            convxd(out_planes, out_planes, fs, stride, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            pooling(2, 2)
            )
    else:
        block = nn.Sequential(
            convxd(in_planes, out_planes, fs, 1, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            convxd(out_planes, out_planes, fs, stride, int(fs/2)),
            #norm(out_planes),
            relu(True),
            norm(out_planes),
            pooling(2, 2)
            )
        
    return block
