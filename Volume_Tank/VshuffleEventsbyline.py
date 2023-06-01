
import numpy as np
import pandas as pd

infile='tankPMT_withonlyMRDcut_insidevolume_withTexp.csv'

df=pd.read_csv('tankPMT_withonlyMRDcut_insidevolume_withTexp.csv')
# shuffle the dataframe rows
df_shuffled = df.sample(frac=1)

# save the shuffled dataframe to a new CSV file
df_shuffled.to_csv('shuffled_tankPMT_withonlyMRDcut_insidevolume_withTexp.csv', index=False)

