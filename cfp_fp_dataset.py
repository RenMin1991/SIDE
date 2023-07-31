# -*- coding: utf-8 -*-
"""
attck
RrenMin 20191012
"""

import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import random
import bcolz
from pathlib import Path


import pdb


# parameters
class CFPFPset(object):
    def __init__ (self, data_folder):
        self.data_carray, self.issame = self.get_val_pair(Path(data_folder), 'cfp_fp')

    # pre-process
    def get_val_pair(self, path, name):
        carray = bcolz.carray(rootdir = path/name, mode='r')
        issame = np.load(path/'{}_list.npy'.format(name))
        return carray, issame
        
    def pre_process(self, image):
        img = 0.299*image[:,0,:,:] + 0.587*image[:,1,:,:] + 0.114*image[:,2,:,:]
        return img#.unsqueeze(0)
    
    def __getitem__(self, index):
        img1 = self.data_carray[2*index]
        img2 = self.data_carray[2*index+1]
        img1, img2 = torch.tensor(img1).unsqueeze(0), torch.tensor(img2).unsqueeze(0)
        img1, img2 = self.pre_process(img1), self.pre_process(img2)

        iss = self.issame[index]
        if iss:
            lab = 1.
        else:
            lab = 0.
            
        return img1, img2, lab
    
    def __len__(self):
        return len(self.issame)



    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
