# -*- coding: utf-8 -*-
"""
train pca selection model with policy gradient
RenMin 20190918
"""


import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from pca_select_model import Encoder, PCASelection
import pca_loss_fn as PLF
import argparse
import pdb


    

# define model

encoder = Encoder()
pca_layer = PCASelection(num_eig=1500)


ckpt_data = torch.load('checkpoint/PCAselect_sample10_memory_4.pth', map_location=lambda storage, loc:storage)
encoder.load_state_dict(ckpt_data['encoder'])
pca_layer.load_state_dict(ckpt_data['pca_layer'])

encoder = encoder.cuda()
pca_layer = pca_layer.cuda()
encoder.eval()
pca_layer.eval()

# pre-process
    
transforms_celebA = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    
# recon
def default_loader(path, rgb=True):
    if rgb:
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path)
    
def img_read(img_path, transform):
    img = default_loader(img_path)
    img = transform(img)
    return img


def img_write(write_path, img):
    img = img*0.5+0.5
    img = img.cpu()
    save_image(img.view(-1, 3, 112, 112), write_path)


mc_loss = PLF.MCLoss(lamb_regular=3e-3, lamb_ID=8., num_eig=1500)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='parameter transfer')
    parser.add_argument('--read_path', type=str, help='image read path')
    parser.add_argument('--write_path', type=str, help='image save path')
    args = parser.parse_args()
    #pdb.set_trace()
    img = img_read(args.read_path, transforms_celebA)
    img = img.unsqueeze(0).cuda()

    hidden = encoder(img)
    recon = pca_layer(hidden)
    recon_img = mc_loss.recon_face(recon, img)
    recon_img = recon_img.expand(recon_img.size(0), 3, recon_img.size(2), recon_img.size(3))

    img_write(args.write_path, recon_img)
    







