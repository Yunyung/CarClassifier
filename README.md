# CarClassifier
This repository gathers the code for car image classification from the [in-class Kaggle challenge](https://www.kaggle.com/c/cs-t0828-2020-hw1/).

## Reproducing Submission
Our model achieve 95.04% accuracy in testing set.

To reproduct my submission without retrainig, do the following steps:
1. [Installation](#Installation)
2. [Ensemble Prediction](#Ensemble-Prediction)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n carClassifier python=3.7
source activate carClassifier
pip install -r requirements.txt
```

## Download Official Image
Official image can be downloaded from [Kaggle challenge](https://www.kaggle.com/c/cs-t0828-2020-hw1/) or just downloaded from my repository.

## Dataset Preparation
After downloading images from [Kaggle challenge](https://www.kaggle.com/c/cs-t0828-2020-hw1/), we expect the data directory is structured as:
```
data
  +- training_data        # all training data from kaggle
  +- testing_data         # all testing data from kaggle
  - training_labels.csv   # csv file contain img's id and label
```

Run the following command to build the data directory above
Run:
```
mkdir data/train
mkdir data/val
python dataPreprocessing.py
```

After run the command, the data directory should be following struture:
```
data
    +- training_data        # all training data from kaggle
    +- testing_data         # all testing data from kaggle
    +- train                # training set split
    +- val                  # validatoin set split from training_data
    - train.csv             # record img's id and label in train folder
    - val.csv               # record img's id and albel in val folder
    - training_labels.csv   # # csv file contain img's id and label

```


## Training 
Training configuration can be specified in ```src/configs.py```.
Then, run:
```
python src/train.py 
```
Default model will use pretrain ResNet-50.

In addition, you can use parameter ```-m "model_name"``` specify what kinds of pretrained model you want to train.

For instance:
```
python src/train.py -m "resnet50"
```


Pretrained model come from [PyTorch office[1]](https://pytorch.org/docs/stable/torchvision/models.html), [rwightman/pytorch-image-models[2]](https://github.com/rwightman/pytorch-image-models). Avaliable pretrained model name in this task are showed below: 
```
"resnet50", "resnet101", "tresnet_l", "tresnet_m", "densenet121", "resnext50_32x4d", "resnext101_32x8d"
```

Trained model will be save as ```src/model/trained_model/model.pth```
## Inference
If trained model are prepared, use it to infer your testing data.

Run:
```
python src/infer.py -mp "trained_model_path"
```
This will save the testing predictions in ```test_result/test_pred.csv```.



## Ensemble Prediction
In the task, we apply bagging (Majority vote) method, one kinds of ensemble, to improve our accuracy in testing set.

Make sure your testing results (*.csv) in [Inference](#Inference) stage inside ```./test_result```. The program will use all .csv file in ```./test_result``` to perform bagging (Majority vote) method.

run: 
```
python src/ensemble.py
```

Result will be save in ```./ensemble.csv```.

**For Reproducing Submission**: There are some testing result in ```./test_result``` that we tested before. You can directly run ```python src/ensemble.py``` without traning.
## Future Work.
Provide more argument parsing parameters to make the user easier to use our program.

## References
[1] [Pretrained model in PyTorch Office WebSite](https://pytorch.org/docs/stable/torchvision/models.html)

[2] [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).