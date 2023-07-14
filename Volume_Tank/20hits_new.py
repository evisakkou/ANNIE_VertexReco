import sys
import glob
import numpy as np
import pandas as pd
import tempfile
import random
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from array import array

infile = 'tankPMT_withonlyMRDcut_insidevolume2.csv'

print("--- opening file with input variables!")
# --- events for training - MC events
filein = open(str(infile))
print("evts for training in: ", filein)

Dataset = np.array(pd.read_csv(filein))

hitx, hity, hitz, hitT, totalPMTs, labels, gridpoint, cm = np.split(Dataset, [1100, 2200, 3300, 4400, 4401, 4404, 4405], axis=1)

def hits_medianT(dt, dx, dy, dz):
    d00 = pd.concat([dt, dx, dy, dz], axis=1)
    print('d00', d00.head())
    d0 = d00[d00.columns[d00.columns.astype(str).str.startswith('T_')]]    
    print(d0)
    # calculate median and its index:
    for col in d0.columns:
        medianT = d0[col].median()
        posmedian = d0.loc[d0[col] == medianT]
        print("posmedian: ", posmedian, " with value: ", medianT)

        d0.sort_values(by=col, inplace=True)
        print("d0 after sorting: \n", d0)
        pos1 = d0[d0[col] > medianT].iloc[0]
        pos2 = d0[d0[col] < medianT].iloc[-1]
        print("pos1: ", pos1, " pos2: ", pos2)
        df_short = d0[d0[col] < medianT]
        print(df_short)

        df_ToUse = df_short[df_short[col] < medianT].iloc[:20]  # Select the first 20 rows
        print('df to use', df_ToUse)

        df_ToUse = pd.DataFrame(df_ToUse.values, columns=df_ToUse.columns)  # Convert df_ToUse to DataFrame

        return df_ToUse


# Create the modified DataFrame
df_ToUse = hits_medianT(hitT, hitx, hity, hitz)

output_df = pd.DataFrame(
    {
        'nhits': totalPMTs[:, 0],
        'truevtxX': labels[:, 0],
        'truevtxY': labels[:, 1],
        'truevtxZ': labels[:, 2],
        'Gridpoint': gridpoint[:, 0],
        'xc': cm[:, 0],
        'yc': cm[:, 1],
        'zc': cm[:, 2],
    }
)

final_df = pd.concat([df_ToUse, output_df[['nhits', 'truevtxX', 'truevtxY', 'truevtxZ', 'Gridpoint', 'xc', 'yc', 'zc']]], axis=1)
print(final_df)

# Save the modified DataFrame to a new CSV file
final_file = '/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv'
final_df.to_csv(final_file, float_format='%.3f', index=False)
print(f'Saved the modified data to {final_file}.')
