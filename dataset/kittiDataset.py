import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class KittiDataset(Dataset):
    def __init__ (self, img_dir, anno_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = os.listdir(self.img_dir)        

        self.anno_dir = anno_dir        
        self.anno_files = os.listdir(self.anno_dir)

        self.transform = transform

    def __len__(self):
        return len(self.anno_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = read_image(img_path)

        anno_path = os.path.join(self.anno_dir, self.anno_files[idx])
        with open(anno_path) as f:
            anno = [self.parseAnnotation(line) for line in f.readlines()]            

        return {'image' : image, 'anno' : anno}
    
    def parseAnnotation(self, anno):
        anno = anno.strip('\n').split(' ')
        anno[0] = 99.0
        anno = [float(anno) for anno in anno]

        return anno       

class KittiCollate(object):
    def __init__ (self, img_w=1242, img_h=375):
        self.img_w = img_w
        self.img_h = img_h

    def __call__(self, batch):
        self.images = [sample['image'] for sample in batch]
        self.annos = [sample['anno'] for sample in batch]
        print(self.annos)
    
    def padImage(self):
        pass

    def padAnnotation(self):
        pass