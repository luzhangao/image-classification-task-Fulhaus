# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/11/02
@description:
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    from utils.gerenal_tools import open_yaml
    paras = open_yaml("../data/data_for_test.yaml")
    from torchvision import transforms
    dataset = ImageDataset(paras["datasets_path"] + paras["train_labels"] + paras["train_labels_file"], paras["datasets_path"] + paras["train_images"],  transforms.Compose([
        transforms.Resize(paras["scaled_size"]),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),)
    for elem in dataset:
        print(elem[0], elem[1])
