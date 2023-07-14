import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
from array import array
import math

infile ='/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank_Texpected.csv'
df = pd.read_csv(infile)
print(df.head())
#print(df.tail())

# Read file with gridpoints vertices to calculate the expected hit time #
# Each row corresponds to the gridpoint number:
infile2 = '~/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/gridpoint_coords.csv'
df2 = pd.read_csv(infile2, header = None)
print(df2.head())
coord = np.array(df2)
print('coord[0][:] ',coord[0][:])
print('coordlen', coord.shape)

hitx_cols = [f'X_{i}' for i in range(1,21)]
hity_cols = [f'Y_{i}' for i in range(1,21)]
hitz_cols = [f'Z_{i}' for i in range(1,21)]
hitT_cols = [f'T_{i}' for i in range(1,21)]

hitx = df[hitx_cols]
hity = df[hity_cols]
hitz = df[hitz_cols]
hitT = df[hitT_cols]

print(hitT.shape)
print('hitT ', hitT.head())

c = 29.9792458  # in cm/ns

def min_Texp(texp_arr, hittime):
    dt = abs(texp_arr - hittime)
    mindtexp = np.amin(dt)  # find min(dt)
    ind = np.argmin(dt)  # return its index
    return ind, mindtexp

def find_nextMin(index, texp_arr):
    new_texp_foreachhit = np.delete(texp_arr, index)
    return new_texp_foreachhit

texp_foreachhit = []
MinGridpoint=[]
df['MinGrid_X'] = np.nan
df['MinGrid_Y'] = np.nan
df['MinGrid_Z'] = np.nan
df['Dist'] = np.nan


for n in range(len(5)):
    if (n + 1) % 100 == 0:
        print(f"Processed {n+1} events.")
    for i in range(20):
        texp_foreachhit = []
        for j in range(1000):
            s = np.sqrt((hitx.iloc[n, i] - coord[j, 0])**2 + (hity.iloc[n, i] - coord[j, 1])**2 + (hitz.iloc[n, i] - coord[j, 2])**2)
            texp = s / c
            texp_foreachhit.append(texp)
        
        #find the index and value of the expected time for each hit(out of 20 hits before median)
        index_DT, mindt = min_Texp(texp_foreachhit, hitT.iloc[n, i])
        df.loc[n, f'texp_{i+1}'] = texp_foreachhit[index_DT]

        new_texp_foreachhit = find_nextMin(index_DT, texp_foreachhit) #array without the 1st Dt minimum
        index_DT2, mindt2 = min_Texp(new_texp_foreachhit, hitT.iloc[n, i]) #find 2nd Dt minimum 
        

        # Store minimum dt and corresponding grid points
        
        df.loc[n, f'MinGrid_X'] = coord[index_DT][0]
        df.loc[n, f'MinGrid_Y'] = coord[index_DT][1]
        df.loc[n, f'MinGrid_Z'] = coord[index_DT][2]
        # df.loc[n, f'Min_dt'] = mindt

    # Calculate distance between (xc, yc, zc) and (MinGrid_X, MinGrid_Y, MinGrid_Z)
    dist = np.sqrt((df.loc[n, 'xc'] - df.loc[n, 'MinGrid_X'])**2 + (df.loc[n, 'yc'] - df.loc[n, 'MinGrid_Y'])**2 + (df.loc[n, 'zc'] - df.loc[n, 'MinGrid_Z'])**2)
    df.loc[n, 'Dist'] = dist

    # print('dist', dist)

df.to_csv('tankPMT_withonlyMRDcut_insidevolume_withTexp.csv', index=False, float_format = '%.3f')
print(df.head())
print(f'Saved the modified data to {df}.')