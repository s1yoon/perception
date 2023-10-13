import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

class KittiDataset(Dataset):
    def __init__ (self, img_dir, anno_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = os.listdir(self.img_dir)       
        self.anno_dir = anno_dir        
        self.anno_files = os.listdir(self.anno_dir)
        self.classdict =  {'Car':1, 'Van':2, 'Truck':3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6, 'Tram':7, 'Misc':8, 'DontCare':0}
        self.transform = transform

    def __len__(self):
        return len(self.anno_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = read_image(img_path)
        anno_path = os.path.join(self.anno_dir, self.anno_files[idx])
        anno = self.read_annotation(anno_path)     
        return {'image' : img, 'anno' : anno}
    
    def read_annotation(self, anno_path):
        with open(anno_path) as f:
            anno = [self.str2float_annotation(line) for line in f.readlines()]  
        return anno

    def str2float_annotation(self, line):
        line = line.strip('\n').split(' ')
        line[0] = self.classdict[line[0]]
        line = [float(ele) for ele in line]
        return line       

class KittiCollate(object):
    def __init__ (self, img_w=1242, img_h=375):
        self.img_w = img_w
        self.img_h = img_h

    def __call__(self, batch):
        self.images = [self.pad_image(sample['image']) for sample in batch]
        lines_max = max([len(sample['anno']) for sample in batch])
        self.annos = [self.pad_annotation(sample['anno'], lines_max) for sample in batch]
        return self.images, self.annos
    
    def pad_image(self, img):
        if list(img.shape[-2:]) != [self.img_h, self.img_w]:
            print("HERE")
            print(img.shape[-2:])
            img = v2.Resize([self.img_h, self.img_w])(img)
        return(img)

    def pad_annotation(self, anno, lines_max):
        anno = np.pad(np.array(anno), ((0,(lines_max-len(anno))), (0,0)), 'constant', constant_values=0)
        print(anno.shape)
        return(anno)

