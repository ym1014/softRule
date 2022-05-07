%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns;
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm



df = pd.read_csv('Book3.csv')
print(df.head(10))
fig, ax = plt.subplots(figsize=(15,5))
divnorm = TwoSlopeNorm(vmin=df.min().min(), vcenter=0, vmax=df.max().max())
ax = sns.heatmap(df,cmap="PiYG", norm=divnorm)
helix.columns
couple_columns = helix[['pred:no_relation','pred:org:alternate_names', 'helix1 phase']]
couple_columns.head()
phase_1_2 = helix.columns
print(phase_1_2.shape)
phase_1_2 = phase_1_2.reset_index()
phase_1_2.head()
plt.figure(figsize=(9,9))
pivot_table = phase_1_2.pivot('pred:no_relation', 'pred:org:alternate_names','pred:org:founded_by')
plt.xlabel('helix 2 phase', size = 15)
plt.ylabel('helix1 phase', size = 15)
plt.title('Energy from Helix Phase Angles', size = 15)
sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');
# Default heatmap
p1 = sns.heatmap(helix)

