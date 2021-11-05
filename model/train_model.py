# coding:utf8

"""
@author: Zhangao Lu
@contact: zlu2@laurentian.ca
@time: 2021/11/02
@description:
"""


import copy
import numpy as np
import matplotlib.pyplot as plt
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, Sequential
from torch.optim.lr_scheduler import StepLR
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils.gerenal_tools import open_yaml
from model.load_datasets import ImageDataset


wandb_switch = True
paras = open_yaml("../data/data_for_test.yaml")
if wandb_switch:
    wandb.init(project="image-classification-fulhaus", entity="luzhangao")
    wandb.config = {
        "learning_rate": paras["lr"],
        "epochs": paras["epochs"],
        "batch_size": paras["batch_size"]
    }
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run():
    """
    Run the training process
    :return:
    """
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

    train_dataloader = DataLoader(train_dataset, batch_size=paras["batch_size"], shuffle=True, num_workers=paras["number_workers"])
    val_dataloader = DataLoader(val_dataset, batch_size=paras["batch_size"], shuffle=True, num_workers=paras["number_workers"])
    dataloaders = {"train": train_dataloader, "val": val_dataloader}

    model = models.vgg16_bn(pretrained=True)  # Load pretrained VGG16 model.
    # Freeze model weights.
    for param in model.parameters():
        param.requires_grad = False
    # Update the weights for the last layer because there are 3 categories need to be classify,
    # which is different from the original VGG16 model.
    model.classifier[6] = Sequential(Linear(4096, 3))
    # Activate the model weights for the last layer
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    # print(model)

    # Move model to GPU.
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=paras["lr"], momentum=paras["momentum"])
    scheduler = StepLR(optimizer, step_size=paras["step_size"], gamma=paras["gamma"])

    new_model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=paras["epochs"])
    torch.save(new_model, paras["model_saved_path"] + paras["model_name"])
    new_model = torch.load(paras["model_saved_path"] + paras["model_name"])
    new_model.eval()
    for _ in range(10):
        visualize_model(new_model, dataloaders["val"])


def imshow(inp, title=None):
    """
    Imshow for Tensor.
    :param inp: torch.Tensor
    :param title: string
    :return:
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)


def visualize_model(model, dataloader, num_images=6):
    """

    :param model: torchvision.models
    :param dataloader: torch.utils.data.dataloader.DataLoader
    :param num_images: int
    :return:
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    class_names = {v: k for k, v in paras["categories"].items()}
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # Predict from input tensors.
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, label: {}'.format(class_names[preds[j].item()], class_names[labels[j].item()]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """

    :param model: torchvision.models
    :param dataloaders: dict, {key: torch.utils.data.dataloader.DataLoader}
    :param criterion: e.g. torch.nn.modules.loss.CrossEntropyLoss
    :param optimizer: e.g. torch.optim.sgd.SGD
    :param scheduler: e.g. torch.optim.lr_scheduler
    :param num_epochs: int
    :return:
    """
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
                    wandb.log({"loss": loss}) if wandb_switch else None

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
            if wandb_switch:
                wandb.log({"epoch loss": epoch_loss})
                wandb.log({"epoch acc": epoch_acc})

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



