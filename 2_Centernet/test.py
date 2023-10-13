from torch.utils.data import DataLoader
from dataset.kittiDataset import KittiDataset, KittiCollate


img_dir = r"D:\git\perception_external\data_object_image_2\training\image_2"
anno_dir = r"D:\git\perception_external\data_object_label_2\training\label_2"



if __name__ == '__main__':
    mydataset = KittiDataset(img_dir, anno_dir)
    collate_fn = KittiCollate()
    mydataloader = DataLoader(mydataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=False)

    train_features, train_labels = next(iter(mydataloader))