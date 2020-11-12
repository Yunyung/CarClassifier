import numpy as np
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv("data/training_labelsWithInt.csv")

# Class Distribution
sns.countplot(x = "label_int", data=df)
plt.show()