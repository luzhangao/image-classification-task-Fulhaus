# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/10/28
@description: parse the raw data and convert into train dataset and val dataset.
parent
├── control
└── datasets
    └── images
        └── train
        └── val
    └── labels
        └── train
        └── val
└── raw_data
    └── Data for test
        └── Bed
        └── Chair
        └── Sofa
"""

import os
import math
import pandas as pd
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from utils.gerenal_tools import open_yaml
from data.rescale_images import Rescale
from skimage import io


paras = open_yaml("data_for_test.yaml")


def generate_datasets():
    """
    parse the raw images from Data for test and save them into datasets
    :return:
    """
    raw_path = paras["raw_path"]
    categories = paras["categories"]
    labelled_data = {"image_name": [], "label": [], "label_name": []}
    for key in categories:
        category_path = raw_path + key
        for filename in os.listdir(category_path):
            labelled_data["image_name"].append(filename)
            labelled_data["label"].append(categories[key])
            labelled_data["label_name"].append(key)
    df = pd.DataFrame(labelled_data)
    df = df.sample(frac=1.0)
    df = df.reset_index(drop=True)
    scale = math.floor(paras["train_size"] * df.shape[0])
    train = df.loc[0: scale-1]
    val = df.loc[scale:]
    resize_images(train, paras["raw_path"], paras["datasets_path"] + paras["train_images"])
    resize_images(val, paras["raw_path"], paras["datasets_path"] + paras["val_images"])
    train[["image_name", "label"]].to_csv(paras["datasets_path"] + paras["train_labels"] + paras["train_labels_file"])
    val[["image_name", "label"]].to_csv(paras["datasets_path"] + paras["val_labels"] + paras["val_labels_file"])


def resize_images(df, source_path, destination_path):
    """
    Resize images by the image names
    :param df: pd.DataFrame, it contains two columns: "image_name" and "label".
    :param source_path: string
    :param destination_path: string
    :return: None
    """
    scale = Rescale(tuple(paras["scaled_size"]))  # Scale the image to fit VGG16
    for index, row in df.iterrows():
        image_name = row["image_name"]
        label_name = row["label_name"]
        image_source_path = source_path + label_name + "/" + image_name
        image_destination_path = destination_path + image_name
        temp = io.imread(image_source_path)
        # Display the image
        # fig = plt.figure()
        # plt.imshow(temp)
        # plt.show()
        scaled = scale(temp)
        print(temp.shape, scaled.shape)
        io.imsave(image_destination_path, scaled)


if __name__ == '__main__':
    generate_datasets()
