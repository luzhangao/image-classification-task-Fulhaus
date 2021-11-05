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
        """

        :param annotations_file: string
               path of the pandas.DataFrame, the DataFrame example is illustrated below:
                                                            image_name  label
               0             Allure 2 Piece Sofa and Armchair Set.jpg      2
               1               Amira Queen Performance Velvet Bed.jpg      0
               2    Baxton Studio Ally Modern and Contemporary Bei...      2
        :param img_dir: string,
               directory path of the images, e.g. "../datasets/images/train/"
        :param transform: torchvision.transforms.transforms.Compose
               transform for the input images
               e.g. Compose(
                        ToPILImage()
                        Resize(size=[224, 224], interpolation=bilinear, max_size=None, antialias=None)
                        ToTensor()
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    )
        :param target_transform: torchvision.transforms.transforms.Compose
               transform for the output labels
        """
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
    from torchvision import transforms
    paras = open_yaml("../data/data_for_test.yaml")
    dataset = ImageDataset(
        paras["datasets_path"] + paras["train_labels"] + paras["train_labels_file"],
        paras["datasets_path"] + paras["train_images"],
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(paras["scaled_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        )
    for elem in dataset:
        print(elem)
        break
