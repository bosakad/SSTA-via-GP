import pandas as pd
from pathlib import Path
from os import listdir
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
This module plots real data of a delay of a inverter
"""

this_dir = Path(__file__).resolve().parent

data_files = listdir(this_dir/"Inputs.outputs/data")

print(data_files)

# for file in data_files:
# 	data = pd.read_csv(this_dir/"data"/file)

data = pd.read_csv(this_dir/"Inputs.outputs/data"/data_files[0])

# print(data.head())

data = data.drop(['Test', 'Spec', 'Weight', 'Pass/Fail'], axis=1)

# print(data.head())

names_of_experiments = data.Output.unique()
# print(names_of_experiments)

# Create a DataFrame with the desired columns,
# where each column has data for a particular gate (inverter).
df = pd.DataFrame(columns=names_of_experiments)

# Extract the simulations data as an array and write to the DataFrame.
for name in data.Output.unique():
	df[f"{name}"] = data[data['Output']==f'{name}']['Nominal'].to_numpy()

# print(df.head())


	# plot data
fig, ax = plt.subplots(1, 1, gridspec_kw={'wspace':0.5,'hspace':0.5})
xmin = np.min(df.iloc[:,0])
xmax = np.max(df.iloc[:,0])
x = np.linspace(xmin, xmax, 31)

xmin = np.min(df.iloc[:,0] * 10000000000000)
xmax = np.max(df.iloc[:,0] * 10000000000000)
xHelp = np.linspace(xmin, xmax, 31)

data = df.iloc[:,0] * 10000000000000
hist, edges = np.histogram(data, xHelp, normed=True)

plt.hist(x[:-1], x, weights=hist * 10000000000000, alpha=0.2, color='blue')

xmin = np.min(df.iloc[:,0] * 10000000000000)
xmax = np.max(df.iloc[:,0] * 10000000000000)
xHelp = np.linspace(xmin, xmax, 100)
mu, std = norm.fit(df.iloc[:,0] * 10000000000000)

print(mu / 10000000000000, std / 10000000000000)

xmin = np.min(df.iloc[:,0])
xmax = np.max(df.iloc[:,0])
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(xHelp, mu, std) * 10000000000000

plt.plot(x, p, 'k', linewidth=2)
ax.set_xlabel('Delay(seconds)', fontsize=12)
ax.set_ylabel('PDF', fontsize=12)
# plt.show()
plt.savefig("Inputs.outputs/inverterReal.jpeg", dpi=800, bbox_inches='tight')



