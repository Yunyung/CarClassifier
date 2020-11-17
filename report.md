Project repositoy: https://github.com/Yunyung/CarClassifier

###### tags: `Image Classification`

---
## Environment
Framework: PyTorch
## Introdunction
&nbsp;&nbsp;&nbsp;&nbsp;The challenge is a car image classification task with 196 classes. The first difficult in this challenge is high inter-class similarity that means images between different classes are very similar. The second problem is that training dataset only contain 11,185 images. To label an image to one of 196 classes require large training data for high accuracy performace. In this task, various deep learning models and powerful techniques are used to achieve as high accuracy as possible.

## Methodology

### Data preprocessing
- **Split dataset** - To verify our model performance, split training data to training dataset and validation data is important. It's about 80% of data for training and 20% for validation done in this task. It is notable that class balance are quite important in both dataset.
```python
# straify option are used for class balance after spliting data
train, val = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['label'])
```

- **Resize Image** - The images in dataset do not have the same size, so we resize all image to 224x224 before data fed into model. The process is applied to all dataset included training, validation, testing set.

- **Normalization** - Originally, all images are represented as tensors with pixel values ranging from 0 to 255. Because our model is pretrianed model trained from ImageNet, the images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. More details see [[1]](https://pytorch.org/docs/stable/torchvision/models.html). Normalization also apply to all dataset.
```python
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
- **Class Balance** - The classes are unbalanced in training data, as figure below. To deal with this problem, we apply *WeightedRandomSampler* [[2]](https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/34) that sample each class with nearly equal probability.

<img src="https://i.imgur.com/SxZ2y6N.jpg" title="Figure1." />

- **Data Augmentation** - As deep networks perform and generalize better with a large amount of training data, we perform data augmentation. The data augmentation we perform are showed below:

<img src="https://i.imgur.com/yBFYbpQ.jpg" title="Figure2." />

<hr>

### Transfer Learning
&nbsp;&nbsp;&nbsp;&nbsp;Because our data contains images similar to those in ImageNet, we will start off with a CNN model that has been pretrained on ImageNet. We replace the find fully connected layer to 196 classes and fine-tune the model for better perforamce. In this task, various state-of-the-art models are tested to get higher performace, such as ResNet-50, ResNet-101, DesNet-121, ResNeXt, TResNet[[3]](https://github.com/mrT23/TResNet), etc.

<hr>

### Ensemble - Bagging
![](https://i.imgur.com/vJouRqk.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;Ensemble learning methods, that often trust the top rankings of many machine learning competitions, are based on the hypothesis that combining multiple models together can often produce a much more powerful model. Bagging is one of ensemble method, as figure above. The idea of bagging is then simple: we want to fit several models and “average” their predictions in order to obtain a model with a lower variance. 

&nbsp;&nbsp;&nbsp;&nbsp;In practice, we sample data from training dataset to build the subsets of training data. Each subsets used to train the corresponding classifier. Finally, aggregate all trained classifiers to do <font color="#c00">majority vote</font>, that is, choose the label that most classifier predict. In our experiment, <font color="#c00">ensemble learning method dramatically improve our testing accuracy</font> that will be discuess later.
<hr>

### Model architecture
The model architecture are summarized below:

![](https://i.imgur.com/slUlZOA.png)





&nbsp;&nbsp;&nbsp;&nbsp;The table below are shown the performace of different model and ensemble (Bagging) strategy:
| Model | ResNet-50 | DenseNet-121 | TResNet [[3]](https://github.com/mrT23/TResNet)|
| -------- | -------- | -------- | ---------|
| Accuracy    | 0.9240     | 0.9202     | 0.9304|
|**Model**| Bagging-5 | Bagging-9 | Bagging-12|
|Accuracy| 0.9388 | 0.94800 | <font color="#c00">0.95040</font>|

&nbsp;&nbsp;&nbsp;&nbsp;The <font color="#c00">bagging strategy with 2 trained model</font> acheive <font color="#c00">95%</font> accuracy in testing set.
<hr>

### Hyperparameters
- **loss function** - CrossEntropyLoss
- **Optimizer** - SGD, learning rate = 0.001, momentum=0.9
- **epoch** - 20 - 50 epoch
- **Learning Rate Scheduling** - CosineAnnealingLR. How it work it's show in figure below:

![](https://i.imgur.com/fIBrO2X.jpg)

## Summary
&nbsp;&nbsp;&nbsp;&nbsp;In the task, purely adopt state-of-the-art pretrianed model can achieve about 90% accuracy in testing set. After integrating a lot of techiques such as data augmentation, class balance, transfer learning and ensemble learning method, we achieve 95% high testing accuracy.
## References
[1] https://pytorch.org/docs/stable/torchvision/models.html

[2] [Some problems with WeightedRandomSampler (For class balance)](https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/34)

[3] [TResNet: High Performance GPU-Dedicated Architecture](https://github.com/mrT23/TResNet).
