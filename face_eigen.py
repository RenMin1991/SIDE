# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
from tqdm import tqdm

import pdb

# parameters
celecb_folder = '/data/renmin/dataset_face/arcface_img_align_celeba/'
num_img = 1000
face_file = 'face_eigen/mega_vec.pth'
eigen_file = 'face_eigen/mega_eig.pth'


# pre-process
transforms_celecb = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])



def default_loader(path, rgb):
    if rgb:
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path)#.convert('RGB')

class CelebADataset(Dataset):
    def __init__ (self, data_folder, num_img=None, transform=None, rgb=True):
        super(CelebADataset, self).__init__()
        #pdb.set_trace()
        fn_list = os.listdir(data_folder)
        if num_img is not None:
            fn_list = random.sample(fn_list, num_img)
        self.fn_list = fn_list
        self.transform = transform
        self.loader = default_loader
        self.rgb = rgb
        self.data_folder = data_folder
        self.num_img = num_img

    def __getitem__(self, index):
        fn = self.fn_list[index]
        img_path = os.path.join(self.data_folder, fn)
        img = self.loader(img_path, self.rgb)
        if self.transform is not None:
            img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.fn_list)


celebA_set = CelebADataset(celecb_folder, transform=transforms_celecb)

len_dataset = len(celebA_set)
print ('number of face image:', len_dataset)

# get face images
pdb.set_trace()
face_matrix = torch.zeros(112*112, len_dataset)
for i, data in tqdm(enumerate(celebA_set, 0)):
    # input data
    image, _ = data
    face_matrix[:,i] = image.view(-1)

#torch.save(face_matrix, face_file) # face_matrix: D X N


# get EigenFace
pdb.set_trace()
print ('get EigenFace.')
avg_face = face_matrix.mean(1)
face_matrix = face_matrix - avg_face.unsqueeze(1).expand(112*112, len_dataset) # substract the mean vector
face_matrix = face_matrix.cuda()
C = torch.mm(face_matrix, face_matrix.t()) # covariance matrix
del face_matrix
C = C.cpu()
eig_val, eig_vec = torch.eig(C, eigenvectors=True)


data = dict(
    eig_val = eig_val,    # D X 2
    eig_vec = eig_vec,    # D X N
    avg_face = avg_face,  # D
    )

torch.save(data, eigen_file)

    
