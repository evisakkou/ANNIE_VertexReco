import sys
import glob
import numpy as np
import pandas as pd

infile='shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp.csv'

print( "--- opening file with input variables!")

filein = open(str(infile))
print("evts for training in: ",filein)

Dataset=np.array(pd.read_csv(filein))

features, nhits, labels, gridpoint, cm, texp = np.split(Dataset,[80,81,84,85,88],axis=1)


print('nhits', nhits)
print("features: ",features)
print("labels: ", labels)
print("gridpoint ", gridpoint)
print('texp', texp)

