# data set for face VAE trained on celeba
import torch
from torch.utils.data import Dataset
import torch as t
from PIL import Image
import os
from torchvision import transforms



transform_112=transforms.Compose([
                       transforms.Resize(size=[112,112]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

transform_128=transforms.Compose([
                       transforms.Resize(size=[112,112]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

transforms_pca = transforms.Compose([
        transforms.Resize(size = [112,112]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def default_loader(path, rgb=True):
    if rgb:
        return Image.open(path).convert('RGB')
    else:
        return Image.open(path)#.convert('RGB')
    
class CelebaDataset(Dataset):
    def __init__ (self, data_folder, transform=None, target_transform=None, rgb=True):
        super(CelebaDataset, self).__init__()
        imgs = []
        for img_fn in os.listdir(data_folder):
            img_path = os.path.join(data_folder, img_fn)
            imgs.append(img_path)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.rgb = rgb
        self.data_folder = data_folder

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = self.loader(img_path, self.rgb)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
    

class CelebaDatasetDoubleSize(Dataset):
    def __init__ (self, data_folder, rgb=False):
        super(CelebaDatasetDoubleSize, self).__init__()
        imgs = []
        for img_fn in os.listdir(data_folder):
            img_path = os.path.join(data_folder, img_fn)
            imgs.append(img_path)

        self.imgs = imgs
        self.transform_112 = transforms.Compose([
                       transforms.Resize(size=[112,112]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.transform_128 = transforms.Compose([
                       transforms.Resize(size=[128,128]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.loader = default_loader
        self.rgb = rgb
        self.data_folder = data_folder

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = self.loader(img_path, self.rgb)
        img_112 = self.transform_112(img)
        img_128 = self.transform_128(img)
        
        return img_112, img_128

    def __len__(self):
        return len(self.imgs)
    

class LFWPair(object):
    def __init__ (self, txt, data_folder):
        
        pairs = []
        labels = []
        
        fh = open(txt, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            if len(words)==3:
                pairs.append((words[0], words[1], words[0], words[2]))
                labels.append(1.)
            elif len(words)==4:
                pairs.append((words[0], words[1], words[2], words[3]))
                labels.append(0.)
        fh.close()
        
        self.pairs = pairs
        self.labels = labels
        self.data_folder = data_folder

        self.transform = transforms_pca
        self.loader = default_loader
    
    def GetPath(self, name, ind):
        path = self.data_folder + name + '/' + name + '_' + ind.zfill(4) + '.jpg'
        return path
        
    
    def __getitem__(self, index):
        name1, ind1, name2, ind2 = self.pairs[index]
        label = self.labels[index]
        path1 = self.GetPath(name1, ind1)
        path2 = self.GetPath(name2, ind2)

        if os.path.exists(path1):
            img1 = self.loader(path1)
            img1 = self.transform(img1)
        else:
            img1 = torch.rand(1, 112, 112)
            label = 0.

        if os.path.exists(path2):
            img2 = self.loader(path2)
            img2 = self.transform(img2)
        else:
            img2 = torch.rand(1, 112, 112)
            label = 0.

        return img1, img2, label
        
    def __len__(self):
        return len(self.pairs)
    
class MegaFacePair(object):
    def __init__ (self, txt, data_folder):
        
        pairs = []
        labels = []
        
        fh = open(txt, 'r')
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()

            pairs.append((words[0], words[1]))
            labels.append(int(words[2]))

        fh.close()
        
        self.pairs = pairs
        self.labels = labels
        self.data_folder = data_folder

        self.transform = transforms_pca
        self.loader = default_loader
        
    
    def __getitem__(self, index):
        img1_fn, img2_fn = self.pairs[index]
        label = self.labels[index]
        path1 = self.data_folder + img1_fn
        path2 = self.data_folder + img2_fn

        if os.path.exists(path1):
            img1 = self.loader(path1)
            img1 = self.transform(img1)
        else:
            img1 = torch.rand(1, 112, 112)
            label = 0.

        if os.path.exists(path2):
            img2 = self.loader(path2)
            img2 = self.transform(img2)
        else:
            img2 = torch.rand(1, 112, 112)
            label = 0.

        return img1, img2, label
        
    def __len__(self):
        return len(self.pairs)