import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import argparse 
import numpy as np

import timm

from dataset import TestDataset
from configs import *
from checkpoint import CheckPointStore
"""
Inference testing data
"""
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

    return model_init

# load trained model
def load_model(model_path, model):
    checkPointStore = CheckPointStore()

    print("Load trained model...")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # save checkpoint info in object
    checkPointStore.total_epoch = checkpoint['total_epoch']
    checkPointStore.Best_val_Acc = checkpoint['Best_val_Acc']
    checkPointStore.epoch_loss_history = checkpoint['epoch_loss_history']
    checkPointStore.epoch_acc_history = checkpoint['epoch_acc_history']
    checkPointStore.model_state_dict = checkpoint['model_state_dict']
    print(f'Trained epochs previously : {checkPointStore.total_epoch}\nBest val Acc : {checkPointStore.Best_val_Acc}')
    print("Successfully loaded trained model.")
    
    print(f'Send model to {device}')

    model.to(device)

    return model, checkPointStore

def test_model(model):
    model.eval()   # Set model to evaluate mode

    print("Device : ", device)
    all_pred = []
    with torch.no_grad():
        # Iterate over data.
        for inputs, img_name in testloaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_pred.extend(predicted.cpu().tolist())
    
    #print(all_pred)

    return all_pred

def write_pred_to_csv(file_ids, all_pred):
    with open("test_result/test_pred.csv", "w") as file:
        file.write("id,label\n")
        for file_id, pred in zip(file_ids, all_pred):
            file.write(file_id[:-4] + "," + LABELS[pred] + "\n")
    print("Done.")

def parse_args():
    parser = argparse.ArgumentParser(description='Car Classifier')
    parser.add_argument("-m", dest="model_name", default="resnet50", type=str)
    parser.add_argument("-mp", dest="model_path", type=str)
    return parser.parse_args()

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize([400, 400]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = {"test": "data/testing_data"}

args = parse_args()

image_datasets = {x: TestDataset(data_dir[x], data_transforms[x]) 
                 for x in ["test"]}

print(f'(Data Size) testing data: {len(image_datasets["test"])}')

testloaders = {x: DataLoader(dataset=image_datasets[x], batch_size=BATCHSIZE, 
                             pin_memory=True, num_workers=0)
              for x in ['test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

init_model = initalize_model(args.model_name, num_classes=NUM_CLASSES)
model, checkPointStore = load_model(args.model_path, init_model)

# test 
all_pred = test_model(model)

# write prediction result to csv file
file_ids = image_datasets["test"].get_file_ids()
write_pred_to_csv(file_ids, all_pred)

