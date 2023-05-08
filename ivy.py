import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Load the data from the CSV file
infile = "predictionsVertex_3vars_with_dr.csv"
df = pd.read_csv(infile)
print(df.head())
#Get initial conditions
# cuts_df = df[(df["DR"] > 100) & (df["DR_reco"] > 100)] 
cuts_df = df[(df["DR"] <30) & (df["DR_reco"] >30)] 

print(cuts_df["nhits"])

print('cuts:', cuts_df)
# Set the number of lines to plot
n_lines = 5

# Loop over the first n_lines rows of the dataframe
for i in range(n_lines):

    # Extract the data for the current row
    # line_data = df.iloc[i]

    # Extract the hitY and dr values for the current row
    # hitY = line_data.loc["hitY_1":"hitY_1100"]
    # dr = line_data.loc["dr_1":"dr_1100"]

    hitY = cuts_df[f"hitY_{i+1}"]
    dr = cuts_df[f"dr_{i+1}"]

    # Create a scatter plot for the current row
    plt.scatter(dr, hitY)

    # Add axis labels and a title for the current plot
    plt.xlabel("hit X-Z")
    plt.ylabel("hit Y")
    plt.title(f"Scatter plot for event {i+1}")
    plt.savefig(f"Plots0805/D/event_{i+1}.png")
    # plt.show()

