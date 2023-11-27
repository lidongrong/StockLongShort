import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Exploratory Data Analysis
"""

# Load data
df = pd.read_csv('data/data.csv')

# Pivot the DataFrame
df['date'] = pd.to_datetime(df['date'])
df = df.pivot(index='date', columns='ticker', values='last')

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle (optional)
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Remove x and y labels
plt.xticks(color='white')
plt.yticks(color='white')

plt.show()
