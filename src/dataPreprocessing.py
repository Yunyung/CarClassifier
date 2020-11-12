import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
from shutil import copyfile

def split_train_val():
    '''
        Split training data to training set and validatoin set
        , save train.csv and val.csv file.
        Finally, move the imgs to coressponding folder
    '''
            
    data = pd.read_csv("data/training_labelsWithInt.csv", dtype=str)


    train, val = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['label'])
    print(train)
    print(val)
    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    # copy_img_basedOnCSV(train, isTrain=True)
    # copy_img_basedOnCSV(val, isTrain=False)

def copy_img_basedOnCSV(csv_file, isTrain):
    img_ids = csv_file['id']
    sourceFolderPath = "data/training_data/"

    if (isTrain):
        destFolderPath = "data/train/"
    else:
        destFolderPath = "data/val/"

    for img_id in img_ids:
        sourcePath = sourceFolderPath + img_id + ".jpg"
        destPath = destFolderPath + img_id + ".jpg"
        copyfile(sourcePath, destPath)


def uniqueLabel_Transform(data):
    pass

def LabelEnconde(data):
    le = preprocessing.LabelEncoder()
    le.fit(data["label"])
    print("All unique labels:")
    # print(le.classes_)
    for idx, class_name in enumerate(le.classes_):
        print(f'{idx}: "{class_name}",')
    print("The number of unique labels: ", len(le.classes_))
    encodedLabels = le.transform(data["label"])
    
    return encodedLabels
    
    # labels = data['label']
    # unique_labels = pd.unique(labels)
    # unique_labels.sort()
    # print(unique_labels)
    # print("The number of unique label:", len(unique_labels))

'''
    Encond label to int for network training
'''
data = pd.read_csv("data/training_labels.csv", dtype=str)
data["label_int"] = LabelEnconde(data)
data.to_csv("data/training_labelsWithInt.csv", index=False)


'''
    Split data to training and validation data
'''
split_train_val()

'''
    copy img to corresponding folder based on csv file
'''
train_data = pd.read_csv("data/train.csv", dtype=str)
val_data = pd.read_csv("data/val.csv", dtype=str)
copy_img_basedOnCSV(train_data, isTrain=True)
copy_img_basedOnCSV(val_data, isTrain=False)


    