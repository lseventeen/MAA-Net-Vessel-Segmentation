###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
# from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
#     NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out+x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        # self.query_aap = nn.AdaptiveAvgPool2d(1)
        # self.key_aap = nn.AdaptiveAvgPool2d(1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=2,stride=2)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,  kernel_size=2,stride=2)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C, -1)
        proj_key =self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value =  self.value_conv(x).view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out+x
        return out
class CPAM_fuse(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(CPAM_fuse, self).__init__()
        self.chanel_in = in_dim
        # self.de_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.in_c = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # x = self.de_c(x)
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, C*width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, C*width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize,  -1, C*width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # out = self.in_c(out)
        # out = self.gamma*out+x
        return out
class CPAM(nn.Module):
    def __init__(self, in_channels, shrink = 1):
        super(CPAM, self).__init__()
        inter_channels = in_channels // shrink
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.pam = PAM_Module(inter_channels)
        self.cam = CAM_Module(inter_channels)
        self.conv_c1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        
        self.conv_p1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        
        self.conv_c2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())
        self.conv_p2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        
        self.conv_d = nn.Sequential(nn.BatchNorm2d(inter_channels),
                                    nn.Conv2d(inter_channels, inter_channels*shrink, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels*shrink),
                                    nn.ReLU())

    def forward(self, x):
        feat1 = self.conv_c1(x)
        sa_feat = self.cam(feat1)
        # sa_conv = self.conv_c2(sa_feat)
        

        # feat2 = self.conv_p1(x)
        sc_feat = self.pam(feat1)
        # sc_conv = self.conv_p2(sc_feat)
    

        feat_sum = sa_feat*self.gamma1+sc_feat*self.gamma2+feat1
        
        output = self.conv_d(feat_sum)

     
        return output