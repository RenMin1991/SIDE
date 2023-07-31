# -*- coding: utf-8 -*-
"""
train pca selection model with policy gradient
RenMin 20190918
"""

import imghdr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import time
import csv

from dataset_celeba import CelebaDataset, LFWPair
from pca_select_model import Encoder, PCASelection
import pca_loss_fn as PLF
from adv_utils import FGSM_orig, AverageModel
import pdb


# parameters
ADV_train = True

EPOCHES = 5
BATCH = 4
LR_en = 1e-2
LR_pca = 1e-2
MOMENTUM = 0.9
NOISE_SCALE = 0.04

AVG_DECAY = 0.9

Num_Eig = 1500
#T = 1.25
#T_gamma = 1.
#T_step = 1

Lamb_regular = 3e-3 # weight of sparse regularization
Lamb_ID = 8.  # weight of sparse ID

Print_Step = 10
Push_Step = 200
Avg_Update_Step = 20000

continue_train = False
if continue_train:
    load_file = 'checkpoint/PCAselect_70.pth'
    

# define model

encoder = Encoder()
pca_layer = PCASelection(num_eig=Num_Eig)
# warm up ckpt
if ADV_train:
    warm_data = torch.load('checkpoint/PCAselect_sample30_memory_warm_0.pth', map_location=lambda storage, loc:storage)
    encoder.load_state_dict(warm_data['encoder'])
    pca_layer.load_state_dict(warm_data['pca_layer'])


if continue_train:
    ckpt_data = torch.load(load_file, map_location=lambda storage, loc:storage)
    encoder.load_state_dict(ckpt_data['encoder'])
    pca_layer.load_state_dict(ckpt_data['pca_layer'])

    pre_epoches = ckpt_data['epoches']
    start_epoch = pre_epoches+1
else:
    start_epoch = 0

encoder = encoder.cuda()
pca_layer = pca_layer.cuda()

# EMA model
avg_encoder = AverageModel(encoder, AVG_DECAY)
avg_pca_layer = AverageModel(pca_layer, AVG_DECAY)

# optimizer
params = []
for name, value in encoder.named_parameters():
    params += [{'params':value, 'lr':LR_en}]
for name, value in pca_layer.named_parameters():
    if 'memory'  not in name:
        params += [{'params':value, 'lr':LR_pca}]

optimizer = optim.SGD(params=params, lr=LR_en, momentum=MOMENTUM)

if continue_train:
    optimizer.load_state_dict(ckpt_data['optimizer'])

if continue_train:
    last_epoch = start_epoch - 2
else:
    last_epoch = -1
lr_sch = optim.lr_scheduler.StepLR(optimizer, 100, 0.1, last_epoch=last_epoch)

# pre-process
    
transforms_celebA = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    
# get data
celeba_set = CelebaDataset('/data/renmin/dataset_face/img_align_celeba/', transform=transforms_celebA)
celeba_loader = DataLoader(celeba_set, batch_size=BATCH, shuffle=True, drop_last=True)
print ('length of CelebA Loader:', len(celeba_loader))

lfw_pair_set = LFWPair(txt='/data/renmin/dataset_face/LFW/pairs.txt', data_folder='/data/renmin/dataset_face/LFW/lfw_align_arcface/')
len_lfw = len(lfw_pair_set)
print ('length of LFW:', len_lfw)
lfw_push_index = [1,12,23,34,45,56,67,78]
lfw_imgs_raw = []
for ind in lfw_push_index:
    img1, img2, label = lfw_pair_set[ind]
    img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    lfw_imgs_raw.append(img1)
    lfw_imgs_raw.append(img2)
lfw_imgs_raw = torch.cat(lfw_imgs_raw)



# loss function
#dis_loss = PLF.DistributionLoss(T, T_gamma, T_step)
mc_loss = PLF.MCLoss(lamb_regular=Lamb_regular, lamb_ID=Lamb_ID, num_eig=Num_Eig)




# noise function
def noise_fn(inputs, scale):
    """
    add Gaussion noise to inputs
    sacle is the Sigmma of Gaission distribution
    """
    noise = torch.randn(inputs.size())
    noise = noise*scale
    inputs_noise = inputs + noise
    return inputs_noise


def get_recon_att(recon, save_fn='SSM_sample30.csv'):
    # recon: B X 1500
    recon = recon.detach().cpu()
    B = recon.size(0)
    bin_recon =  recon>0.5
    sparse = (bin_recon*1.).sum(1).mean()
    specific = []
    for i in range(B):
        for j in range(i+1, B):
            hanming_dis = (bin_recon[i,:] ^ bin_recon[j, :]).sum()
            specific.append(hanming_dis.float())
    #pdb.set_trace()
    specific = torch.tensor(specific).mean()
    mutation = (recon - 0.5).abs().mean()
    with open(save_fn, 'a') as  csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sparse.item(), specific.item(), mutation.item()])




#train

#pdb.set_trace()
for epoch in range(start_epoch, EPOCHES):
    running_loss = 0.
    running_recon_loss = 0.
    running_id_loss = 0.
    running_sparse_loss = 0.
    running_sim_adv = 0.
    start_time = time.time()

    # adverarial attacker
    if ADV_train:
        attacker = FGSM_orig(recon_fn=mc_loss.recon_face, step=NOISE_SCALE, encoder=avg_encoder, pca_layer=avg_pca_layer)

    for i, data in enumerate(celeba_loader, 0):

        encoder.train()
        pca_layer.train()
        # input data
        inputs = data.clone()

        # perturbation

        #inputs_noise = noise_fn(inputs, NOISE_SCALE)
        if ADV_train:
            inputs_noise, sim_adv = attacker.attack(inputs.cuda())
        else:
            inputs_noise = noise_fn(inputs, NOISE_SCALE)

        inputs, inputs_noise = inputs.cuda(), inputs_noise.cuda()
        
        # zero the grad
        optimizer.zero_grad()
        
        # forward
        hidden = encoder(inputs_noise)
        recon = pca_layer(hidden)
        
        # loss and backward
        #loss = dis_loss(recon, inputs, index, epoch)# + lambda_mean*recon.mean()
        loss, rec_loss, id_loss, sparse_loss = mc_loss(recon, inputs, inputs_noise)
        
        loss.backward()
        optimizer.step()
        
        # print log
        get_recon_att(recon=recon)

        running_loss += loss.item()
        running_recon_loss += rec_loss.item()
        running_id_loss += id_loss.item()
        running_sparse_loss += sparse_loss.item()
        if ADV_train:
            running_sim_adv += sim_adv.item()
        if i%Print_Step==0 and i>0:
            #print ('epoch', epoch, 'step', i, 'loss', running_loss/(i+1.))
            if ADV_train:
                print ('Epoch:%3i/%3i  Step:%4i/%4i  sim_adv:%8.6f   loss:%8.6f  rec_loss:%8.6f  id_loss:%8.6f  sparse_loss:%8.6f'\
                    %(epoch,EPOCHES,i,len(celeba_loader), running_sim_adv/(i+1.),  running_loss/(i+1.), running_recon_loss/(i+1.), running_id_loss/(i+1.), running_sparse_loss/(i+1.)))
            else:
                print ('Epoch:%3i/%3i  Step:%4i/%4i  loss:%8.6f  rec_loss:%8.6f  id_loss:%8.6f  sparse_loss:%8.6f'\
                       %(epoch,EPOCHES,i,len(celeba_loader),  running_loss/(i+1.), running_recon_loss/(i+1.), running_id_loss/(i+1.), running_sparse_loss/(i+1.)))

        # push examples
        if i%Push_Step==0:
            encoder.eval()
            pca_layer.eval()
            #with torch.no_grad():
            inputs = data.clone()
            
            if ADV_train:
                inputs_noise, sim_adv = attacker.attack(inputs.cuda())
            else:
                inputs_noise = noise_fn(inputs, NOISE_SCALE)

            inputs, inputs_noise = inputs.cuda(), inputs_noise.cuda()
        
            # forward
            hidden = encoder(inputs_noise)
            recon = pca_layer(hidden)
            # reconstruction
            recon_img = mc_loss.recon_face(recon, inputs_noise)
            recon_img = recon_img.expand(inputs.size(0), 3, inputs.size(2), inputs.size(3))
            inputs_noise = inputs_noise.expand(inputs.size(0), 3, inputs.size(2), inputs.size(3))

            # sim
            sim_mean = mc_loss.face_sim(inputs, recon_img)
            if ADV_train:
                print ('celebA sim raw_adv:', sim_adv.item(), 'celebA sim raw_recon:', sim_mean.item())
            else:
                print ('celebA sim raw_recon:', sim_mean.item())

            # save
            resultsample = torch.cat([inputs_noise, recon_img]) * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample.view(-1, 3, 112, 112),
                        'push_example/PCAselect_sample30_celeba_' + str(epoch) + "_" + str(i) + '.png')
                
            
            # test on LFW
            #with torch.no_grad():
            #pdb.set_trace()
            lfw_imgs = lfw_imgs_raw.clone()

            if ADV_train:
                lfw_imgs_noise, sim_adv = attacker.attack(lfw_imgs.cuda())
            else:
                lfw_imgs_noise = noise_fn(lfw_imgs, NOISE_SCALE)

            lfw_imgs, lfw_imgs_noise = lfw_imgs.cuda(), lfw_imgs_noise.cuda()
            # forward
            hidden = encoder(lfw_imgs_noise)
            recon = pca_layer(hidden)
            # reconstruction
            recon_img = mc_loss.recon_face(recon, lfw_imgs_noise)
            recon_img = recon_img.expand(lfw_imgs_noise.size(0), 3, lfw_imgs_noise.size(2), lfw_imgs_noise.size(3))
            lfw_imgs_noise = lfw_imgs_noise.expand(lfw_imgs_noise.size(0), 3, lfw_imgs_noise.size(2), lfw_imgs_noise.size(3))

            # sim
            sim_mean = mc_loss.face_sim(lfw_imgs, recon_img)
            if ADV_train:
                print ('lfw sim raw_adv:', sim_adv.item(), 'lfw sim raw_recon:', sim_mean.item())
            else:
                print ('lfw sim raw_recon:', sim_mean.item())

            # save
            resultsample = torch.cat([lfw_imgs_noise, recon_img]) * 0.5 + 0.5
            resultsample = resultsample.cpu()
            save_image(resultsample.view(-1, 3, 112, 112),
                        'push_example/PCAselect_sample30_lfw_' + str(epoch) + "_" + str(i) + '.png')

        if i%Avg_Update_Step==0 and i>0:
            avg_encoder.update_parameters(encoder)
            avg_pca_layer.update_parameters(pca_layer)
            attacker = FGSM_orig(recon_fn=mc_loss.recon_face, step=NOISE_SCALE, encoder=avg_encoder, pca_layer=avg_pca_layer)

    #pdb.set_trace()
    lr_sch.step(epoch)
    

        
    # save model
    data = dict(
            optimizer = optimizer.state_dict(),
            encoder = encoder.state_dict(),
            pca_layer = pca_layer.state_dict(),
            epoches = epoch,
            )
    save_name = 'checkpoint/PCAselect_sample30_memory_'+str(epoch)+'.pth'
    torch.save(data, save_name)

    # time
    end_time = time.time()
    print ('time:', end_time - start_time)
        




