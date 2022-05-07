graph.py
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
cars = pd.read_csv('Book3.csv')
cars.head()
cars.corr()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
xlabels = ['pred:no_relation', 'pred:org:alternate_names', 'pred:org:city_of_headquarters', 'pred:org:country_of_headquarters', 'pred:org:dissolved','pred:org:founded','pred:org:founded_by']
ylabels = ['no_relation', 'org:alternate_names', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:dissolved', 'org:founded','org:founded_by']
orders = np.array([[-478, -53, 35, 70, 23,20,7], 
                   [-1, -258, 0, 0, 0,0,0],
                   [-2, 1, 1, -2, 0,0,0],
                   [-14, 1, 4, 7, 0,0,0],
                  [0, 0, 0, 0, -1,1,0],
                  [0, 0, 0, 0, -2,2,0],
                  [-1, 0, 0, 0, 0,0,-11]]
                 )

plt.figure(figsize=(8,5))
sns.heatmap(orders, 
            cmap='YlOrBr',
            vmin=0,
            xticklabels=xlabels,
            yticklabels=ylabels,
            annot=True,
            alpha=0,
            square=True,
            annot_kws={'fontsize':14, 'fontweight': 'bold', 'color': 'black'}
           )
plt.yticks(rotation=0)
plt.tick_params(
    which='both',      
    bottom=False,      
    left=False,      
    labelbottom=False,
    labeltop=True) 
plt.tight_layout();
plt.figure(figsize=(8,7))
sns.heatmap(orders, 
            cmap='YlOrBr', 
            vmin=0, 
            xticklabels=xlabels, 
            yticklabels=ylabels, 
            annot=True, 
            square=True, 
            annot_kws={'fontsize':10, 'fontweight': 'bold', 'color': 'black'} ) 
sns.set(font_scale=1.1) 
plt.yticks(rotation=0) 
plt.xticks(rotation=30) 
plt.autofmt_xdate() 
plt.tick_params( which='both',
bottom=True,
left=True,
labelbottom=True, labeltop=True) 
plt.tight_layout();