# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:11:54 2018

@author: USER
"""
import torch.nn as nn


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=6,        
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, 
                                            # padding=(kernel_size-1)/2 if stride=1
            ),                            
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv2 = nn.Sequential(       
            nn.Conv2d(6, 16, 5, 1, 2),   
            nn.ReLU(),                      
            nn.MaxPool2d(2),               
        )
        self.conv3 = nn.Sequential(       
            nn.Conv2d(16, 120, 5, 1,2),     
            nn.ReLU(),                     
            nn.MaxPool2d(2),                
        )
        self.FC = nn.Linear(120*4*4, 84)   
        self.norm = nn.BatchNorm1d(84, affine=False)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        x = conv3.view(conv3.size(0), -1)    # flatten 
        x1 = self.FC(x)
        x = self.norm(x1)      
        output = self.out(x)
        return conv1,conv2,conv3,output    # return x for visualization