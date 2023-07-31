# -*- coding: utf-8 -*-
"""
defense method
RenMin 20191020
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from PIL import Image

from pixelcnn_model import PixelCNN
import pdb



class ImageQuilt(object):
    def __init__(self, dataset, size_patch=11, step=9):
        self.size_patch = size_patch
        self.step = step
        if dataset == 'lfw':
            self.patchset = torch.load('patches/lfw.pth')
        elif dataset == 'mega':
            self.patchset = torch.load('patches/mega.pth')
            
    def smash(self, x):
        num_1d = int((112-self.size_patch)/self.step)+1
        patches = torch.zeros(num_1d**2, self.size_patch**2)
        for j in range(num_1d):
            for k in range(num_1d):
                patch = x[:,:,j*self.step:j*self.step+self.size_patch, k*self.step:k*self.step+self.size_patch]
                patches[j*num_1d+k,:] = patch.contiguous().view(-1)
        return patches
    
    def search(self, patches):
        N, _ = self.patchset.size()
        n, _ = patches.size()
        patches_dataset = torch.zeros(patches.size())
        for i in range(n):
            distances = ((self.patchset - patches[i,:].unsqueeze(0).expand(N, patches.size(1)))**2).sum(1)
            _, indexes = distances.sort()
            index_patch = indexes[0]
            patches_dataset[i,:] = self.patchset[index_patch,:]
        return patches_dataset
    
    def recon(self, patches_dataset):
        #pdb.set_trace()
        count_mat = torch.zeros(1,1,112,112)
        x_quilted = torch.zeros(1,1,112,112)
        num_1d = int((112-self.size_patch)/self.step)+1
        for j in range(num_1d):
            for k in range(num_1d):
                count_mat[:,:,j*self.step:j*self.step+self.size_patch, k*self.step:k*self.step+self.size_patch] = count_mat[:,:,j*self.step:j*self.step+self.size_patch, k*self.step:k*self.step+self.size_patch] + 1.
                x_quilted[:,:,j*self.step:j*self.step+self.size_patch, k*self.step:k*self.step+self.size_patch] = x_quilted[:,:,j*self.step:j*self.step+self.size_patch, k*self.step:k*self.step+self.size_patch] + patches_dataset[j*num_1d+k,:].contiguous().view(1,1,self.size_patch, self.size_patch)
        count_mat[count_mat==0.] = 1.
        x_quilted = x_quilted / count_mat
        return x_quilted
        
            
    def forward(self, x):
        #pdb.set_trace()
        patches = self.smash(x)
        patches_dataset = self.search(patches)
        x_quilted = self.recon(patches_dataset)
        return x_quilted
    

class TVM_loss(nn.Module):
    def __init__(self, lamb, p):
        super(TVM_loss, self).__init__()
        self.lamb = lamb
        uni_dis = torch.rand(1,1,112,112)
        self.X = (uni_dis<p)#.cuda()
        
    def forward(self, x, x_ad):
        #pdb.set_trace()
        recon_loss = ((x[self.X] - x_ad[self.X])**2).sum()
        x_right = x[:,:,:,1:]
        x_down = x[:,:,1:,:]
        tv_loss = ((x[:,:,:,:-1] - x_right)**2).sum() + ((x[:,:,:-1,:] - x_down)**2).sum()
        loss = recon_loss + self.lamb*tv_loss
        return loss
    
class TVM(object):
    def __init__(self, steps=1000, lr=0.01):
        self.loss_fn = TVM_loss(lamb=0.1, p=0.5)
        self.steps = steps
        self.lr = lr
        
    def forward(self, x):
        x_recon = x
        #pdb.set_trace()
        for step in range(self.steps):
            x_recon = Variable(x_recon, requires_grad=True)
            loss = self.loss_fn(x_recon, x)
            loss.backward()
            x_recon = x_recon.detach() - self.lr*x_recon.grad.data
            
        return x_recon
            
            
class PixelDef(object):
    def __init__(self, e=6):
        self.pixelcnn = PixelCNN()
        model_dict = torch.load('checkpoint/pixelcnn.pth', map_location=lambda storage, loc:storage)['model']
        self.pixelcnn.load_state_dict(model_dict)
        self.pixelcnn = self.pixelcnn.cuda()
        self.pixelcnn.eval()
        self.e = e
        
    def forward(self, x):
        _,_,H,W = x.size()
        x = x.cuda()
        x = (x*0.5 + 0.5)*255.
        for i in range(H):
            for j in range(W):
                prediction = self.pixelcnn(x)
                prediction = prediction.detach()
                R_low = max((x[0,0,i,j] - self.e).long(), 0)
                R_high = min((x[0,0,i,j] + self.e).long(), 255)
                if R_high > R_low:
                    pixel_value = prediction[0,R_low:R_high,i,j].argmax() + R_low
                    x[0,0,i,j] = pixel_value
        x = (x/255. - 0.5)/0.5
        
        return x
                
        



















