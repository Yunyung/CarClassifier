import pandas as pd 
import numpy as np
import os
from configs import *
"""
Ensemble - Bagging (Majority Voting)
"""
def MajorityVoting(fileNames_csv):
    for i, fn in enumerate(fileNames_csv):
        df = pd.read_csv("test_result/" + fn, dtype=str)
        labels_str = df["label"].to_numpy()
        labels_int = [LABELS_TO_INT[label] for label in labels_str] 
        numOflabels = len(labels_int)

        # Apply one hot encoding to deal with majority voting
        # Convert array of indices to 1-hot encoded numpy array
        # initalize np 2D-array with all zeros
        if i == 0:
            one_hot_Mat = np.zeros((numOflabels, NUM_CLASSES)) # 2D array -> (# of data in testing set, # of classes)
            img_id = df["id"] 
            
        one_hot_Mat[np.arange(numOflabels), labels_int] += 1 # Accumulate voting from each model to corresponding label in 1-hot encoding 

    # pick up the majority label in 1-hot encoding
    ensemble_result = np.argmax(one_hot_Mat, axis=1)    

    # tranlate label(int) to label(str)
    label_result = [LABELS[label_int] for label_int in ensemble_result]

    # write final result to csv 
    d = {'id': img_id, 'label': label_result}
    df = pd.DataFrame(data=d)
    df.to_csv("ensemble.csv", index=False)
    print("Successfully complete ensemble prediction.")

fileNames_csv = os.listdir("test_result")

MajorityVoting(fileNames_csv)




