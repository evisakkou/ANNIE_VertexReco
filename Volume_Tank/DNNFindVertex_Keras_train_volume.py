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

infile='shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp.csv'

# Set TF random seed to improve reproducibility
seed = 170
np.random.seed(seed)

print( "--- opening file with input variables!")
#--- events for training - MC events
filein = open(str(infile))
print("evts for training in: ",filein)
Dataset=np.array(pd.read_csv(filein))
# example of a normalization
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

# define min max scaler
scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(Dataset)
print(scaled)


features,hitT, nhits, labels, gridpoint, cm, texp = np.split(Dataset,[60, 80, 81,84,85,88],axis=1)


print('nhits', nhits)
print("features: ",features)
print("labels: ", labels)
print("gridpoint ", gridpoint)
print('texp', texp)

#split events in train/test samples:
num_events, num_pixels = features.shape
# print(num_events, num_pixels)
np.random.seed(0)
train_x = cm[:3000]
train_y = labels[:3000]
num_events, num_pixels = train_x.shape
print(num_events, num_pixels)

print("train sample features shape: ", train_x.shape," train sample label shape: ", train_y.shape)

# Scale data (training set) to 0 mean and unit standard deviation.
# scaler = preprocessing.StandardScaler()
# train_x = scaler.fit_transform(train_x)


def custom_loss_function(y_true, y_pred):
    #dist = math.dist(y_true, y_pred)
    dist = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))
    return dist
   #squared_difference = tf.square(y_true - y_pred)
   #return tf.reduce_mean(squared_difference, axis=-1)

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=3, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss=custom_loss_function, optimizer='ftrl', metrics=custom_loss_function)
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=20, batch_size=4, verbose=0)
'''
kfold = KFold(n_splits=10)#, random_state=seed)
results = cross_val_score(estimator, train_x, train_y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''
# checkpoint
filepath="weights_bets20.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
callbacks_list = [checkpoint]

# Fit the model
#history = estimator.fit(train_x, train_y, validation_split=0.33, epochs=22, batch_size=1, callbacks=callbacks_list, verbose=0)
#kfold = KFold(n_splits=4)
#results = cross_val_score(estimator, train_x, train_y, cv=kfold)
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
history = estimator.fit(train_x, train_y, validation_split=0.33,batch_size=1, callbacks=callbacks_list, verbose=1)
#-----------------------------
# summarize history for loss
f, ax2 = plt.subplots(1,1)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Performance')
ax2.set_xlabel('Epochs')
ax2.set_xlim(0.,20.)
ax2.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig("keras_train_test_volume_cm.pdf")