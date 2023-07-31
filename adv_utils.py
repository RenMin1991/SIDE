
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


class PGD(object):
    def __init__(self, recon_fn, reg_model='r50', step=1./255., scale_pert=0.04, num_step=40):
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

        self.step = step
        self.scale_pert = scale_pert
        self.num_step = num_step
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
        for i in range(self.num_step):

            image_grad = Variable(image_grad, requires_grad=True)
            #pdb.set_trace()

            if self.recon_fn is not None:
                image_grad_recon = self.recon_fn(image_grad)
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

            # perturbation
            pert_semi = self.step * torch.sign(image_grad.grad.data)
            image_adv_semi = image_grad + pert_semi

            pert = torch.clamp(image_adv_semi.detach() - image, min=-self.scale_pert, max=self.scale_pert)
            image_grad = torch.clamp(image + pert, min=-1, max=1).detach()
        
        image_adv = image_grad.detach()

        # final forward
        feat_adv = self.face_model(image_adv.expand(B, 3, H, W))
        sim_adv = torch.matmul(feat_adv, anchor_feat.t()).diag().mean()

        return image_adv, sim_adv


class DeepFool(object):
    def __init__(self, recon_fn, reg_model='r50', scale_pert=0.04, max_step=40):
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

        self.scale_pert = scale_pert
        self.max_step = max_step
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
        pert_norm = 0.
        i = 0
        while pert_norm < self.scale_pert and i < self.max_step:

            image_grad = Variable(image_grad, requires_grad=True)
            #pdb.set_trace()

            if self.recon_fn is not None:
                image_grad_recon = self.recon_fn(image_grad)
            else:
                image_grad_recon = image_grad

            feat = self.face_model(image_grad_recon.expand(B, 3, H, W)) # B X 512
            sim = torch.matmul(feat, anchor_feat.t()).diag().mean()
            if Pos:
                loss = 1. - sim
            else:
                loss = sim

            # backward
            loss.backward()
            #pdb.set_trace()

            # perturbation
            pert_semi = torch.sign(image_grad.grad.data)
            pert = sim.abs().data * pert_semi / ((pert_semi**2).sum())**0.5

            image_grad = torch.clamp(image_grad + pert, min=-1, max=1).detach()

            pert_norm = (image_grad - image).abs().max()

            i += 1
        
        image_adv = image_grad.detach()

        # final forward
        feat_adv = self.face_model(image_adv.expand(B, 3, H, W))
        sim_adv = torch.matmul(feat_adv, anchor_feat.t()).diag().mean()

        return image_adv, sim_adv
    


class Evolutionary(object):
    def __init__(self, encoder, pca_layer, recon_fn, reg_model='r50', num_iter=10000):
        """
        max_iter:  number of max iteration
        """
        self.num_iter = num_iter
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
        self.recon_fn = recon_fn
    
    def try_check(self, img_try, img, img_raw, anchor_feat, T, Pos):
        # check img_star_try
        B,_,H,W = img_try.size()

        hidden = self.encoder(img_try)
        recon = self.pca_layer(hidden)
        # reconstruction
        recon_img = self.recon_fn(recon, img_try)

        feat = self.face_model(recon_img.expand(B, 3, H, W))
        sim = cos_sim(feat.detach(), anchor_feat)
        if Pos: # positive pair
            if sim<T:
                recg_signal = 1
            else:
                recg_signal = 0
        else: # negative pair
            if sim>T:
                recg_signal = 1
            else:
                recg_signal = 0
        dis = ((img - img_raw)**2).sum()
        dis_try = ((img_try - img_raw)**2).sum()
        if dis_try < dis:
            dis_signal = 1
        else:
            dis_signal = 0
        return recg_signal * dis_signal, sim
    
    def clean_check(self, image, image_anchor, T, Pos):
        B,_,H,W = image.size()
        with torch.no_grad():
            anchor_feat = self.face_model(image_anchor.expand(B, 3, H, W))
            hidden = self.encoder(image)
            recon = self.pca_layer(hidden)
            # reconstruction
            recon_img = self.recon_fn(recon, image)

            feat = self.face_model(recon_img.expand(B, 3, H, W))
            sim = cos_sim(feat.detach(), anchor_feat)

        if Pos:
            if sim < T:
                clean_flag = False
            else:
                clean_flag = True
        else:
            if sim > T:
                clean_flag = False
            else:
                clean_flag = True
        return clean_flag, sim.item()

        

    def attack(self, image, image_anchor=None, Pos=True):
        # initialization
        B,_,H,W = image.size()
        if image_anchor is None:
            image_anchor = image.clone()
        with torch.no_grad():
            if image_anchor is None:
                anchor_feat = self.face_model(image.expand(B, 3, H, W)) # B X 512
            else:
                anchor_feat = self.face_model(image_anchor.expand(B, 3, H, W)) # B X 512

        if Pos:
            img_star = torch.rand(image.size()).cuda()
        else:
            img_star = image_anchor.detach()
        
        n = 112 * 112
        m = 45 * 45
        k = 100
        
        mu = 0.05
        len_mu_list = 5
        C = torch.eye(m)
        p_c = torch.zeros(m)
        c_c = 0.01
        c_cov = 0.001
        sigma_scale = 0.1
        
        scale_perts = []
        try_signals = []

        clean_flag, sim_clean = self.clean_check(image=image, image_anchor=image_anchor,T=0.3,Pos=Pos)
        
        # iteration
        if clean_flag:

            for ite in tqdm(range(self.num_iter), ncols=79):
                
                # sampling
                mse = ((image - img_star)**2).mean()
                scale_perts.append(mse.item())
                
                sigma = sigma_scale * (mse**0.5).item()
                z = np.zeros(m)
                for i in range(m):
                    z[i] = np.random.normal(0., sigma*C[i,i].numpy())
                #z = np.random.multivariate_normal(np.zeros(m), (C*sigma**2).numpy())

                # coordinates selection
                c = C.diagonal()
                c = k * F.softmax(c, dim=0)
                uni_dis = torch.rand(c.size())
                drop_index = uni_dis>c
                z[drop_index] = 0.
                
                # upsampling
                z_upSamp = z.reshape(45,45)
                z_upSamp = transform.resize(z_upSamp, (112, 112))
                z_upSamp = torch.from_numpy(z_upSamp.reshape(1,1,112,112)).cuda()
                
                # add by mu
                z_upSamp = z_upSamp.float() + mu*(image - img_star)
                
                # update img_tar
                img_star_try = img_star + z_upSamp

                # check img_star_try
                try_signal, sim = self.try_check(img_star_try, img_star, image, anchor_feat, T=0.3, Pos=Pos)
                
                # update mu
                if len(try_signals) < len_mu_list:
                    try_signals.append(try_signal)
                else:
                    try_signals.pop(0)
                    try_signals.append(try_signal)
                P_suc = np.array(try_signals).mean()
                mu = mu * np.exp(P_suc - 0.2)
                
                # update
                if try_signal==1:
                    # update img_star
                    img_star = img_star_try
                    # update p_c, C
                    p_c = (1-c_c) * p_c + (c_c*(2-c_c))**0.5 * (torch.from_numpy(z/sigma).float())
                    index_C = range(m)
                    C[index_C, index_C] = (1-c_cov) * C[index_C, index_C] + c_cov * p_c**2

        else:
            sim = sim_clean


                    
        return clean_flag, img_star, sim.detach(), scale_perts