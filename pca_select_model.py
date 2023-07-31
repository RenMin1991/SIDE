# -*- coding: utf-8 -*-
"""
DAE model and PCA selection layers
RenMin 20190918
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=True):
        super(EncoderBlock, self).__init__()
        self.shortcut = shortcut
        if shortcut:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels)
                                          )
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1 ,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.nolinear1 = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.nolinear2 = nn.PReLU(out_channels)

        
    def forward(self, x):
        if self.shortcut:
            shortcut_x = self.shortcut_layer(x)
        x = self.nolinear1(self.bn1(self.conv1(x)))
        x = self.nolinear2(self.bn2(self.conv2(x)))
        #x = self.nolinear1(self.conv1(x))
        #x = self.nolinear2(self.conv2(x))
        if self.shortcut:
            x = x + shortcut_x
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class Memory_Block(nn.Module):        
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super(Memory_Block, self).__init__()
        
        self.c = hdim # dim of input feature
        self.k = kdim # number memory item
        
        self.moving_average_rate = moving_average_rate
        
        self.units = nn.Embedding(kdim, hdim)
                
    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1] # (n, )
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype) # (n, k)        
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data
                
    def forward(self, x, update_flag=True):
        '''
          x: (n, c)
          embed: (k, c)
        '''
        
        n, c = x.size()        
        assert c == self.c        
        k, c = self.k, self.c
        
        m = self.units.weight.data # (k, c)
                
        xn = F.normalize(x, dim=1) # (n, c)
        mn = F.normalize(m, dim=1) # (k, c)
        score = torch.matmul(xn, mn.t()) # (n, k)
        
        if update_flag:
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1) # (k, c)
            score = torch.matmul(xn, mn.t()) # (n, k)
        
        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m) # (n, c)
        #out = out.view(b, h, w, c).permute(0, 3, 1, 2)
                                
        return out, score





class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_layer = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, bias=False),
                                      nn.PReLU(64))
        
        self.block1 = EncoderBlock(64, 64, stride=2)
        self.block2 = EncoderBlock(64, 64, stride=1)
        self.block3 = EncoderBlock(64, 128, stride=2)
        self.block4 = EncoderBlock(128, 128, stride=1)
        self.block5 = EncoderBlock(128, 256, stride=2)
        self.block6 = EncoderBlock(256, 256, stride=1)
        self.block7 = EncoderBlock(256, 512, stride=2)
        self.block8 = EncoderBlock(512, 512, stride=1)
        
        #self.out_layer = nn.Sequential(Flatten(),
                                    #nn.Linear(256*7*7, 1024),
                                    #nn.BatchNorm1d(1024),
                                    #nn.PReLU(1024))
        
    def forward(self, x):
        #pdb.set_trace()
        x = self.in_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        #x = self.out_layer(x)
        return x







class PCASelection(nn.Module):
    def __init__(self, num_eig):
        super(PCASelection, self).__init__()
        self.fc1 = nn.Sequential(Flatten(),
                                nn.Linear(512*7*7, 512),
                                nn.BatchNorm1d(512),
                                nn.PReLU(512))
        self.memory = Memory_Block(512, 128)
        self.fc2 = nn.Sequential(nn.Linear(512, num_eig),
                                nn.BatchNorm1d(num_eig),
                                nn.Sigmoid())
        
    def forward(self, x, abl=0):
        #pdb.set_trace()
        x = self.fc1(x)
        if abl==0:
            if self.training:
                x, _ = self.memory(x)
            else:
                x, _ = self.memory(x, update_flag=False)
        x = self.fc2(x)
        
        return x
        
if __name__ == '__main__':
    encoder = Encoder()
    pca_layer = PCASelection()

    x = torch.rand(5, 1, 112, 112)

    hidden = encoder(x)
    recon = pca_layer(hidden)

    print (recon.size())





        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
