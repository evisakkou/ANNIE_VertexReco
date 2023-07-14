import pandas as pd

#REARANGE THE X Y Z T COLUMNS

df = pd.read_csv('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv')

# Get the number of sets (e.g., 20)
num_sets = 20

# Create a list to store the column order
column_order = []

# Rearrange columns by grouping X, Y, Z, and T values together
for i in range(1, num_sets + 1):
    column_order.extend(['X_' + str(i)])

for i in range(1, num_sets + 1):
    column_order.extend(['Y_' + str(i)])

for i in range(1, num_sets + 1):
    column_order.extend(['Z_' + str(i)])

for i in range(1, num_sets + 1):
    column_order.extend(['T_' + str(i)])

# Select the columns in the desired order
new_df = df[column_order]


# Save the rearranged columns to a new CSV file
new_df.to_csv('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv', index=False)


#ADD COLUMNS FOR NHITS, GRIDPOINT, AND TRUE GRID COORDS

# Read the existing CSV file into a DataFrame
existing_df = pd.read_csv('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv')

# Read the additional CSV file into a DataFrame
extra_df = pd.read_csv('tankPMT_withonlyMRDcut_insidevolume2.csv')

# Extract the desired columns from the extra DataFrame
add_columns = ['totalPMTs', 'truevtxX', 'truevtxY', 'truevtxZ', 'Gridpoint', 'xc', 'yc', 'zc']
extra_columns = extra_df[add_columns]

# Concatenate the existing DataFrame with the extra columns
combined_df = pd.concat([existing_df, extra_columns], axis=1)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('/home/evi/Desktop/ANNIE-THESIS-2/VolumeTank/ANNIE_VertexReco/Volume_Tank/VolumeTank_Texpected1.csv', index=False)
