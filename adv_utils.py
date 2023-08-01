
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage import transform
from torch.autograd import Variable
import torchvision.transforms as transforms
import copy

import pca_loss_fn as PLF
from face_utils.arcface import MobiFaceNet, Backbone, remove_module_dict
import pdb


class AverageModel(nn.Module):
    def __init__(self, model, decay, device=None):
        super(AverageModel, self).__init__()
        self.module = copy.deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.decay = decay

        def ema_avg(avg_model_param, model_param, decay):
            return decay*avg_model_param + (1-decay)*model_param
        
        self.avg_fn = ema_avg

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def update_parameters(self, model):
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.decay))

def cos_sim(f_a, f_b):
    #pdb.set_trace()
    sim = torch.mm(f_a, f_b.t()) / ((torch.mm(f_a, f_a.t()) * torch.mm(f_b, f_b.t())))**0.5
    return sim

class FGSM_orig(object):
    def __init__(self, recon_fn, reg_model='r50', step=0.01, encoder=None, pca_layer=None):
        """
        step:           step of iterature
        recon_fn:       
        """
        if reg_model=='mb':
            face_model = MobiFaceNet().cuda()
            face_model.load_state_dict(remove_module_dict(torch.load("/data/renmin/interpretable_face/source_code/checkpoint/arcface_mb.pth")))
        elif reg_model=='r50':
            face_model = Backbone(50).cuda() 
            face_model.load_state_dict(remove_module_dict(torch.load("/data/renmin/interpretable_face/source_code/checkpoint/arcface_r50.pth")))
        self.face_model = face_model
        self.face_model.eval()

        self.encoder = encoder
        self.pca_layer = pca_layer
        self.step = step
        self.recon_fn = recon_fn
        
        
    def attack(self, image, image_anchor=None, Pos=True):
        """
        Fast Gradient Signal Attack for face recognition
    
        inputs:
        image:          pytorch tensor, B X 1 X H X W
        
        outputs:
        image_adv:        image with perturbation
        sim:          similarity between image after attack and anchor
        """
        #pdb.set_trace()
        B,_,H,W = image.size()
        # get anchor feature
        with torch.no_grad():
            if image_anchor is None:
                anchor_feat = self.face_model(image.expand(B, 3, H, W)) # B X 512
            else:
                anchor_feat = self.face_model(image_anchor.expand(B, 3, H, W)) # B X 512

        # forward
        image_grad = image.clone()
        image_grad = Variable(image_grad, requires_grad=True)
        if self.encoder is not None and self.pca_layer is not None:
            hidden = self.encoder(image_grad)
            recon = self.pca_layer(hidden)
        #pdb.set_trace()

        if self.recon_fn is not None:
            image_grad_recon = self.recon_fn(recon, image_grad)
        else:
            image_grad_recon = image_grad

        feat = self.face_model(image_grad_recon.expand(B, 3, H, W)) # B X 512
        if Pos:
            loss = 1. - torch.matmul(feat, anchor_feat.t()).diag().mean()
        else:
            loss = torch.matmul(feat, anchor_feat.t()).diag().mean()

        # backward
        loss.backward()
        #pdb.set_trace()
        pert = self.step * torch.sign(image_grad.grad.data)

        # perturbation
        image_adv = (image + pert).detach()

        # final forward
        feat_adv = self.face_model(image_adv.expand(B, 3, H, W))
        sim_adv = torch.matmul(feat_adv, anchor_feat.t()).diag().mean()

        return image_adv, sim_adv


