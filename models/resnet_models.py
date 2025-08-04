#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import numpy as np


# generator & discriminator
class ResidualBlock(nn.Module):
    def __init__(self,num_filters=64):
        super(ResidualBlock,self).__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=num_filters),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_filters,out_channels=num_filters,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=num_filters)
        )
        
    def forward(self,x):
        return self.layers(x)+x
        
    
class Generator(nn.Module):
    def __init__(self,
                 num_residual_block=16,
                 residual_channels=64,
                 num_upscale_layers=3,
                 upscale_factor=1,
                 num_classes=1):
        super(Generator,self).__init__()
        self.num_residual_block=num_residual_block
        self.num_upscale_layers=num_upscale_layers
        self.residual_channels=residual_channels
        self.upscale_factor=upscale_factor
        
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,
                                         out_channels=residual_channels,
                                         kernel_size=9,stride=1,
                                         padding=4),
                                 nn.PReLU())
        
        conv2=nn.ModuleList()
        for n in range(self.num_residual_block):
            #conv2.append(self._residual_block(num_filters=residual_channels))
            conv2.append(ResidualBlock(num_filters=residual_channels))
        self.conv2=conv2
        
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=residual_channels,
                                      out_channels=residual_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1),
                            nn.BatchNorm2d(num_features=residual_channels))
        conv4=[]
        nchans=1 if upscale_factor==1 else upscale_factor**2
        for layer in range(num_upscale_layers):
            conv4+=[nn.Conv2d(in_channels=residual_channels,
                              out_channels=residual_channels*nchans, 
                              kernel_size=3, 
                              stride=1, 
                              padding=1), 
                            nn.PixelShuffle(upscale_factor=upscale_factor),
                            nn.PReLU()]
        
        self.conv4=nn.Sequential(*conv4)
        
        #self.conv5=nn.Sequential(nn.Conv2d(in_channels=64,
        #                              out_channels=256, kernel_size=3, stride=1, padding=1), 
        #                    nn.PixelShuffle(upscale_factor=2),
        #                    nn.PReLU())
        self.conv5=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=1,padding=0,stride=1))
                                 #nn.BCEWithLogitsLoss())
                                 #nn.Sigmoid())
                        
    def forward(self,x):
        x=self.conv1(x)
        old_x=x
        for layer in self.conv2:
            x=layer(x)
        
        x=self.conv3(x)+old_x
        
        #print(x.shape)
        if self.num_upscale_layers>0:
            x=self.conv4(x)
        #print(x.shape)
        x=self.conv5(x)
        #print(x.shape)
        return x