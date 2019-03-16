import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import pandas as pd
from pylab import savefig
import numpy as np


def correlation_table(data):
    le = preprocessing.LabelEncoder() 
    for col in data.columns: 
        if data[col].dtype == pd.np.object: 
            data[col] = le.fit_transform(data[col].astype(dtype=pd.np.str)) 
    corr = data.corr()
    
        # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    pic = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
          
    figure = pic.get_figure()    
     # Thus we have to give more margin:
    figure.subplots_adjust(bottom=0.2)
    
    figure.savefig('data/corr.png', dpi=400)