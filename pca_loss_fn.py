# -*- coding: utf-8 -*-
"""
Loss functions for PCA selection model
RenMin 20190918
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from face_utils.arcface import MobiFaceNet, Backbone, remove_module_dict

import pdb


    

class DistributionLoss(nn.Module):
    def __init__(self, T, T_gamma, T_step):
        super(DistributionLoss, self).__init__()
        self.num_eig = 2500
        eigen_data = torch.load('face_eigen/mega_eig.pth')
        eig_face = eigen_data['eig_vec']
        self.eig_face = eig_face[:,:self.num_eig].cuda()
        self.avg_face = eigen_data['avg_face'].cuda()
        #self.weights = torch.load('weight_bce.pth')
        self.T = T
        self.T_gamma = T_gamma
        self.T_step = T_step
    
    def GetCoef(self, image):
        B, _, H, W = image.size()
        image = image.view(B, H*W).t()
        image = image - self.avg_face.unsqueeze(1).expand(H*W, B) # subtraction by average face
        image = F.normalize(image, dim=0) # normalization
        recon_coef = torch.mm(self.eig_face.t(), image) # num_eig X B
        coef = recon_coef.abs().t() # B X num_eig
        coef_mean = coef.mean(1).unsqueeze(1).expand(B, self.num_eig)
        target = (coef > coef_mean).float()
        #coef = coef / (coef.max(1)[0]).unsqueeze(1).expand(B,self.num_eig)
        
        return target
    
    def anneal(self, target, epoch):
        target_mean = target.mean()
        t = self.T * self.T_gamma**(epoch//self.T_step)
        target = target / t
        target = target - target.mean() + target_mean
        return target

    def weight_MSELoss(self, pca_act, target, weight):
        loss = (((pca_act - target)**2) * weight.cuda()).mean()
        return loss

    def forward(self, pca_act, image, index_image, epoch):
        #pdb.set_trace()
        target = self.GetCoef(image)

        losses = torch.zeros(image.size(0)).cuda()
        for i in range(image.size(0)):
            #loss_fn = nn.BCELoss(weight=self.weights[index_image[i],:]).cuda()
            loss_fn = nn.BCELoss().cuda()
            pca_act_row = pca_act[i,:]
            #target_row = self.anneal(target[i,:], epoch)
            loss_sub = loss_fn(pca_act_row, target[i,:])
            #loss_sub = self.weight_MSELoss(pca_act_row, target_row, self.weights[index_image[i],:])
            losses[i] = loss_sub
        
        return losses.mean()

   
class MCLoss(nn.Module):
    def __init__(self, lamb_regular, lamb_ID, num_sample=25, num_eig=2000, reg_model='r50'):
        super(MCLoss, self).__init__()
        eigen_data = torch.load('face_eigen/celebA_eigen.pth')
        self.avg_face = eigen_data['avg_face'].cuda()
        self.eig_face = eigen_data['eig_vec'][:,:num_eig].cuda()
        self.lamb_regular = lamb_regular
        self.lamb_ID = lamb_ID
        self.num_sample = num_sample
        self.num_eig = num_eig

        if reg_model=='mb':
            face_model = MobiFaceNet().cuda()
            face_model.load_state_dict(remove_module_dict(torch.load("checkpoint/arcface_mb.pth")))
        elif reg_model=='r50':
            face_model = Backbone(50).cuda() 
            face_model.load_state_dict(remove_module_dict(torch.load("checkpoint/arcface_r50.pth")))
        self.face_model = face_model
        self.face_model.eval()
        
    def MCSampling(self, pca_act):
        num_eig = pca_act.size(0)
        prob = pca_act.detach().unsqueeze(0).expand(self.num_sample, num_eig)
        uni_dis = torch.rand(prob.size()).cuda()
        samples = prob>uni_dis
        return samples
    
    def face_sim(self, img1, img2):
        B, C1, H, W = img1.size()
        _, C2, _, _ = img2.size()

        if C1==1:
            img1 = img1.expand(B, 3, H, W)
        if C2==1:
            img2 = img2.expand(B, 3, H, W)
        feat1 = self.face_model(img1)
        feat2 = self.face_model(img2)
        sim_mean = torch.matmul(feat1, feat2.t()).diag().mean()
        return sim_mean

    
    def recon_face(self, pca_act_batch, image_noise_batch):
        #pdb.set_trace()
        B, num_eig = pca_act_batch.size()
        uni_dis = torch.rand(pca_act_batch.size()).cuda()
        samples_batch = pca_act_batch>uni_dis
        
        _, _, H, W = image_noise_batch.size()
        D, _ = self.eig_face.size()

        recon_imgs = []
        for i in range(B):
            image_noise = image_noise_batch[i, :, :, :].view(1, H*W).t() # 1 X H*W
            image_noise = image_noise - self.avg_face.unsqueeze(1) # subtraction by average face
            #reconstruction
            samples_exp = samples_batch[i, :].view(1,1,-1).expand(1, D, num_eig)
            eig_face = self.eig_face.unsqueeze(0) # 1, D, num_eig
            eig_face = samples_exp.float() * eig_face
            
            recon_img = torch.matmul(eig_face,torch.matmul(eig_face.permute(0,2,1), image_noise)) # 1 X D X 1
            recon_img = recon_img + self.avg_face.unsqueeze(1)

            recon_img = recon_img.permute(0,2,1).view(1, 1, H, W)

            recon_imgs.append(recon_img)
        recon_imgs = torch.cat(recon_imgs)

        return recon_imgs
    
    def recon_face_1500(self, image_noise_batch, num_eig=1500):
        B, _, H, W = image_noise_batch.size()
        D, _ = self.eig_face.size()

        recon_imgs = []
        for i in range(B):
            image_noise = image_noise_batch[i, :, :, :].view(1, H*W).t() # 1 X H*W
            image_noise = image_noise - self.avg_face.unsqueeze(1) # subtraction by average face
            #reconstruction
            
            eig_face = self.eig_face[:, :num_eig].unsqueeze(0) # 1, D, num_eig
            
            
            recon_img = torch.matmul(eig_face,torch.matmul(eig_face.permute(0,2,1), image_noise)) # 1 X D X 1
            recon_img = recon_img + self.avg_face.unsqueeze(1)

            recon_img = recon_img.permute(0,2,1).view(1, 1, H, W)

            recon_imgs.append(recon_img)
        recon_imgs = torch.cat(recon_imgs)

        return recon_imgs



    
    def Loss(self, samples, image, image_noise):
        #pdb.set_trace()
        with torch.no_grad():
            samples = samples[:,:self.num_eig]
            _, H, W = image.size()
            D, _ = self.eig_face.size()
            image = image.view(1, H*W).t() # 1 X H*W
            
            image_noise = image_noise.view(1, H*W).t() # 1 X H*W
            image_noise = image_noise - self.avg_face.unsqueeze(1) # subtraction by average face
            
            #reconstruction
            samples_exp = samples.unsqueeze(1).expand(self.num_sample, D, self.num_eig)
            eig_face = self.eig_face.unsqueeze(0).expand(self.num_sample, D, self.num_eig)
            eig_face = samples_exp.float() * eig_face
            
            
            recon_samples = torch.matmul(eig_face,torch.matmul(eig_face.permute(0,2,1), image_noise)) # num_sample X D X 1
            recon_samples = recon_samples + self.avg_face.unsqueeze(1)
            

            # recon loss
            recon_loss = ((recon_samples.squeeze(2) - image.t().expand(self.num_sample,D))**2).mean(1)

            # regular loss
            regular = samples.sum(1).float()

            # ID loss
            image = image.t().view(1, 1, H, W).expand(1, 3, H, W)

            recon_samples = recon_samples.permute(0,2,1).view(self.num_sample, 1, H, W).expand(self.num_sample, 3, H, W)
            img_feat = self.face_model(image) # 1 X 512
            recon_feats = self.face_model(recon_samples) # num_sample X 512
            ID_loss = 1. - torch.matmul(img_feat, recon_feats.t()).squeeze(0)
            
            
            loss = recon_loss + self.lamb_regular * regular + self.lamb_ID * ID_loss
        return loss, recon_loss.mean(), ID_loss.mean(), regular.mean()
        
    def LogLikehood(self, pca_act, samples):
        pca_act = pca_act.unsqueeze(0).expand(self.num_sample, samples.size(1))
        samples = samples.float()
        probability = ((1-samples) - pca_act).abs()
        Likehood = torch.log(probability).sum(1)
        # 这个循环可不可以用sum代替
        #for i in range(samples.size(1)-1):
            #Likehood = Likehood + torch.log(probability[:,i+1])
        #pdb.set_trace()
        #LogLikehood = torch.log(Likehood)
        return Likehood
        
        
    def forward(self, pca_act, images, images_noise):
        #pdb.set_trace()
        losses_batch = []
        recon_losses = []
        ID_losses = []
        regular_losses = []
        for i in range(images.size(0)):
            samples = self.MCSampling(pca_act[i]) # num_sample X num_eig
            # 
            losses, recon_loss, ID_loss, regular = self.Loss(samples, images[i], images_noise[i])
            recon_losses.append(recon_loss)
            ID_losses.append(ID_loss)
            regular_losses.append(regular)

            losses = losses - losses.mean()
            likehoods = self.LogLikehood(pca_act[i], samples)
            losslikehoods = (losses * likehoods).sum().unsqueeze(0)
            losses_batch.append(losslikehoods) # num_sample
        
        mc_loss = torch.cat(losses_batch).mean()
        rec_loss = torch.tensor(recon_losses).mean()
        id_loss = torch.tensor(ID_losses).mean()
        sparse_loss = torch.tensor(regular_losses).mean()
        return mc_loss, rec_loss, id_loss, sparse_loss

if __name__ == '__main__':
    pdb.set_trace()
    img = torch.rand(2,1,112,112).cuda()
    img_noise = torch.rand(2,1,112,112).cuda()
    pca_act = torch.rand(2, 1000).cuda()

    mc_loss = MCLoss(0.1, 8.)

    loss = mc_loss(pca_act, img, img_noise)
    print (loss)
    





















      
