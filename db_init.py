import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
dataset = pd.read_csv('/kaggle/input/datasets/prikandel/train-csv/train.csv')
dataset = np.array(dataset)
num_samples, num_features = dataset.shape
np.random.shuffle(dataset)
val_set = dataset[0:1000].T
val_labels = val_set[0]
val_images = val_set[1:num_features]
val_images = val_images / 255.

train_set = dataset[1000:num_samples].T
train_labels = train_set[0]
train_images = train_set[1:num_features]
train_images = train_images / 255.
_, num_train_samples = train_images.shape