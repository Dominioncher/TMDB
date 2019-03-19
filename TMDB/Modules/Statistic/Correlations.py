import pandas as pd
import seaborn as sns
import numpy as np
from TMDB.Modules.Helpers.LabelEncoding import label_encode
import matplotlib.pyplot as plt


def correlation_table(data: pd.DataFrame, path: str=None) -> None:
    data = label_encode(data)
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    pic = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    if path:
        figure = pic.get_figure()
        figure.subplots_adjust(bottom=0.35, top=1)
        figure.savefig(path, dpi=600)
    plt.show()
