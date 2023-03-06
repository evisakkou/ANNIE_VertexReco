import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#--------- File with events for reconstruction:
#--- evts for training:
#infile = "tankPMT_forVetrexReco.csv"
infile = "tankPMT_forVetrexReco_withRecoV.csv"
#

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
#print((pd.read_csv(filein)).head())
Dataset=np.array(pd.read_csv(filein))
features, rest, recovertex, labels, gridpoint = np.split(Dataset,[4400,4402,4405,4408],axis=1)
print("rest :", rest[0])
print("features: ",features[0])
print("recovertex: ", recovertex[0])
print("labels: ", labels)
np.random.shuffle(Dataset) #shuffling the data sample to avoid any bias in the training
df = pd.DataFrame(Dataset)
print(df.head())
df.to_csv("shuffledtankPMT_forVetrexReco_withRecoV.csv", float_format = '%.3f')

