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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

# infile= 'middleMerged.csv'
infile = "predictionsVertexDRfor10cm.csv"
# df = pd.read_csv('middleMerged.csv')
# filein = open(str(infile))
# print("evts for training in: ",filein)

# Dataset=np.array(pd.read_csv(filein))
# vtx,grid,predicted_grid,rest=np.split(Dataset, [2,3,4], axis=1)

import pandas as pd

def find_tank(truevtxx_tank, truevtxy_tank, truevtxz_tank, step, Gridpoint):
    # make grid for 10cm
    i = 1
    gridpoint = {}
    for z in range(30):
        for r in range(40):
            for c in range(30):
                gridpoint[i] = (z, r, c)
                i += 1

    # find vtx coordinates for given gridpoint
    z, r, c = gridpoint[int(Gridpoint)]
    vtx_z = -140 + z * step
    vtx_y = -190 + r * step
    vtx_x = -140 + c * step

    return vtx_x, vtx_y, vtx_z


# read CSV file into DataFrame
df = pd.read_csv("predictionsVertexDRfor10cm.csv")

# check the length of the DataFrame
print(df.shape)

# apply find_tank function to each row
new_columns = df['Gridpoint'].apply(lambda gp: find_tank(0, 0, 0, 10, gp))

# check the length of the new columns
print(len(new_columns))
print(len(new_columns[0]))

# create new columns for vtx_x, vtx_y, and vtx_z
df[['vtx_x', 'vtx_y', 'vtx_z']] = new_columns

# check the length of the DataFrame again
print(df.shape)

print(df.head())




