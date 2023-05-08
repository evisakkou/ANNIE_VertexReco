import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import math
import matplotlib.pyplot as plt
from array import array


infile = "predictionsVertex_3vars.csv"
# filein = open(str(infile))
# df=np.array(pd.read_csv(filein))
df = pd.read_csv(infile)
print(df.head())
hitX,hitY,hitZ,hitT, labels, recovertex, DR_reco, DR, nhits0=np.split(df, [1100,2200,3300,4400,4403,4406,4407,4408], axis=1)

dr = np.empty(np.shape(hitX))

hitX = np.array(hitX)
hitZ = np.array(hitZ)

for i in range(0, 2143):
    for j in range(0, 1100):
        dr[i][j] = np.linalg.norm(hitX[i][j] - hitZ[i][j])


for i in range(0, 5):
    for j in range(0, 5):
        print(dr[i][j], ' ')
    print('\n')

print('dr shape', dr.shape)
print('hity shape', hitY.shape)
dr_df = pd.DataFrame(dr, columns=[f"dr_{i+1}" for i in range(1100)])
dff = pd.concat([df, dr_df], axis=1)

dff.to_csv("predictionsVertex_3vars_with_dr.csv", index=False)
print(dff.head())


#Make Scatter Plot
n_lines = 5
plot_df = dff.iloc[:n_lines, :]
for i in range(5):
    label = f"Line {i+1}"
    plt.scatter(plot_df[f"dr_{i+1}"], plot_df[f"hitY_{i+1}"], label=label)
    
plt.xlabel("hitX-Z")
plt.ylabel("hitY")
plt.legend()
plt.title("Scatter plot of hitY / hitX-Z")
plt.show()
