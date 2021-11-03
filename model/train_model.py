# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/11/02
@description:
"""


import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, Sequential
from torch.optim.lr_scheduler import StepLR
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.gerenal_tools import open_yaml
from model.load_datasets import ImageDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
paras = open_yaml("../data/data_for_test.yaml")


def run():
    train_dataset = ImageDataset(
        paras["datasets_path"] + paras["train_labels"] + paras["train_labels_file"],
        paras["datasets_path"] + paras["train_images"],
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(paras["scaled_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    )

    val_dataset = ImageDataset(
        paras["datasets_path"] + paras["val_labels"] + paras["val_labels_file"],
        paras["datasets_path"] + paras["val_images"],
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(paras["scaled_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    )

    train_dataloader = DataLoader(train_dataset, batch_size=paras["batch_size"], shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=paras["batch_size"], shuffle=True, num_workers=0)
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    model = models.vgg16_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        model = model.cuda()

    model.classifier[6] = Sequential(Linear(4096, 3))
    # model.classifier
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    # print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=20)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                loader = dataloaders["train"]
            else:
                model.eval()  # Set model to evaluate mode
                loader = dataloaders["val"]

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in loader:

                # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / (len(loader) * paras["batch_size"])
            epoch_acc = running_corrects.double() / (len(loader) * paras["batch_size"])

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    run()


