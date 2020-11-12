import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse 

import timm
from pprint import pprint

from dataset import TrainDataset
from configs import *
from checkpoint import CheckPointStore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initalize_model(pretrain_model_name, num_classes):
    """
     Initialize pretrained model
    """
    print(f'pretrain_model_name: {pretrain_model_name}')
    
    # resnet50
    if (pretrain_model_name == "resnet50"):
        """ ResNet-50 """
        model_init = models.resnet50(pretrained=True) # Initialize the pretrained model

        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)  # replace final fully connected layer
    elif (pretrain_model_name == "resnet101"):
        """ ResNet-101 """
        model_init = models.resnet101(pretrained=True)

        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)  

    elif (pretrain_model_name == "tresnet_l"):
        """ tresnet_l """
        model_init = timm.create_model('tresnet_l', pretrained=True, num_classes=num_classes)
       
        # list all avaliable pretrianed model name
        #model_names = timm.list_models(pretrained=True)
        #pprint(model_names)
    
    elif (pretrain_model_name == "tresnet_m"):
        """ tresnet_m """
        model_init = timm.create_model('tresnet_m', pretrained=True, num_classes=num_classes)

    elif (pretrain_model_name == "densenet121"):
        """ Desnet-121 """
        model_init = models.densenet121(pretrained=True) 
        num_ftrs = model_init.classifier.in_features
        model_init.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif (pretrain_model_name == "resnext50_32x4d"):
        """ ResNeXt-50-32x4d """
        model_init = models.resnext50_32x4d(pretrained=True)
        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)     
    
    elif (pretrain_model_name == "resnext101_32x8d"):
        """ resnext101_32x8d """
        model_init = models.resnext101_32x8d(pretrained=True)
        num_ft = model_init.fc.in_features
        model_init.fc = nn.Linear(num_ft, num_classes)    

    # print model architecture.
    #print(model_init)

    # freeze first n layer 
    ct = 0
    for name, child in model_init.named_children():
        print(f'name={name}, child={child}')
        if ct < NUM_FREEZE_LAYERS:
            print(f'Freeze->{ct} layer')
            for name2, params in child.named_parameters():
                params.requires_grad = False
        ct += 1

    return model_init


def load_model(model):
    checkPointStore = CheckPointStore()

    # load trained model if needed. 
    if (IS_RESUME_TRAINIG):
        print("Load trained model...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

        # save checkpoint info in object
        checkPointStore.total_epoch = checkpoint['total_epoch']
        checkPointStore.Best_val_Acc = checkpoint['Best_val_Acc']
        checkPointStore.epoch_loss_history = checkpoint['epoch_loss_history']
        checkPointStore.epoch_acc_history = checkpoint['epoch_acc_history']
        checkPointStore.model_state_dict = checkpoint['model_state_dict']
        print(f'Trained epochs previously : {checkPointStore.total_epoch}\nBest val Acc : {checkPointStore.Best_val_Acc}')
        print("Successfully loaded trained model.")
    else:
        print("No trained model found, using built-in pretrained model.")

    print(f'Send model to {device}')
    model.to(device)
    return model, checkPointStore


# train model
def train_model(model, criterion, optimizer, scheduler, checkPointStore, num_epochs=25):
    print(f'Training on Device : {device}')

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    trained_epochs = 0
    best_acc = 0.0
    epoch_loss_history = {x: list() for x in ["train", "val"]} # record the loss in each epoch
    epoch_acc_history = {x: list() for x in ["train", "val"]} # record  the accuracy in each epoch 
    
    if (IS_RESUME_TRAINIG):
        # reloaded trained infomation 
        trained_epochs = checkPointStore.total_epoch
        best_acc = checkPointStore.Best_val_Acc
        epoch_loss_history = checkPointStore.epoch_loss_history 
        epoch_acc_history = checkPointStore.epoch_acc_history
        
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                

                inputs = inputs.to(device)
                # print(labels)
                labels = labels.to(device)

                 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss_history[phase].append(epoch_loss)
            epoch_acc_history[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {}/{}({:.4f})'.format(
                phase, epoch_loss, running_corrects, dataset_sizes[phase], epoch_acc))

            # deep copy the model
            if IS_SELECT_MODEL_BY_VAL_ACC and phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save({
                            'total_epoch': trained_epochs + epoch, 
                            'Best_val_Acc': best_acc,
                            'model_state_dict': model.state_dict(),
                            'epoch_loss_history': epoch_loss_history,
                            'epoch_acc_history': epoch_acc_history
                           }, MODEL_SAVE_ROOT_PATH + MODEL_SAVE_NAME + ".pth") # save on GPU mode
                print("Save model info based on val acc.")
            elif ((not IS_SELECT_MODEL_BY_VAL_ACC) and (epoch%5 == 0)):
                # save model when final epoch and not accroding validation accuracy
                torch.save({
                            'total_epoch': trained_epochs + epoch, 
                            'Best_val_Acc': best_acc,
                            'model_state_dict': model.state_dict(),
                            'epoch_loss_history': epoch_loss_history,
                            'epoch_acc_history': epoch_acc_history
                           }, MODEL_SAVE_ROOT_PATH +MODEL_SAVE_NAME + ".pth")
                print("End final epoch - Save model info.")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(10)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Car Classifier')
    parser.add_argument("-m", dest="model_name", default="resnet50", type=str)
    return parser.parse_args()
    
torch.manual_seed(42)

args = parse_args()

# Data augmentation and normalization for training
# Just normalization for validation
transform_options = [
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.RandomRotation(degrees=[-15, 15]),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomAffine(0, shear=20)
]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
             transforms.RandomChoice(transform_options)
        ], p=0.9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = {"train": "data/train", "val": "data/val"}
#data_dir = {"train": "data/training_data", "val": "data/val"}
#data_dir = {"train": "data/val", "val": "data/train"}
csv_file_path = {"train": "data/train.csv", "val": "data/val.csv"}
#csv_file_path = {"train": "data/training_labelsWithInt.csv", "val": "data/val.csv"}
#csv_file_path = {"train": "data/val.csv", "val": "data/train.csv"}
image_datasets = {x: TrainDataset(data_dir[x], csv_file_path[x], data_transforms[x])
                for x in ["train", "val"]}

weightedRandomSampler = image_datasets["train"].getWeightedRandomSampler()

print(f'(Data Size) Training data: {len(image_datasets["train"])}, Valiation data: {len(image_datasets["val"])}')

dataloaders = {}

# dataloader no class balance
#dataloaders = {x: DataLoader(dataset=image_datasets[x], batch_size=BATCHSIZE, 
#                             shuffle=True, pin_memory=True, num_workers=0)

# dataloader using WeightedRandomSampler for class balance
dataloaders["train"] = DataLoader(dataset=image_datasets["train"], batch_size=BATCHSIZE, 
                             pin_memory=True, num_workers=0, sampler = weightedRandomSampler)
dataloaders["val"] = DataLoader(dataset=image_datasets["val"], batch_size=BATCHSIZE, 
                             shuffle=True, pin_memory=True, num_workers=0)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# show img
inputs, labels = next(iter(dataloaders['train'])) # Get a batch of training data
out = torchvision.utils.make_grid(inputs) # Make a grid from batch
imshow(out, title=labels)

init_model = initalize_model(args.model_name, NUM_CLASSES) # initalize built-in pretrained model
model, checkPointStore = load_model(init_model) # load trained model and trained info in CheckPointStore object

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

# Learning rate scheduler
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=4)

model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    checkPointStore, num_epochs=NUM_EPOCHS)
