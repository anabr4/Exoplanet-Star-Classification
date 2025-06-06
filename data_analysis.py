'''
PAIR PLOTS OF FISRT 5 FLUX VALUES BEFORE AND AFTER REMOVING OUTLIERS
'''

# Import standard packages
import os
import pathlib
import numpy as np
import pandas as pd

# For Plotting Purpose and statistics
import matplotlib.pyplot as plt
import seaborn as sn

# For Outliers Search
import collections as cls

# Read the Train dataframe
current_path = pathlib.Path().absolute()
exoTrain = pd.read_csv(os.path.join(current_path, "data/", "exoTrain.csv"))

# PairPlot for first 5 columns (first 5 features) with all oobservations (stars)
subset_data = exoTrain.iloc[:, [0] + list(range(1, 6))]
# We write explicitely the order of the concatenation to plot the stars labelled with 2 above the one with label 1
fig1 = sn.pairplot(pd.concat([subset_data[subset_data['LABEL'] != 2], subset_data[subset_data['LABEL'] == 2]]), hue='LABEL')


def StDev_method (df,n,features):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations
    containing more than n outliers according to the standard deviation method.
    """
    outlier_indices = []
    outlier_counts = cls.defaultdict(int)

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()

        # calculate the cutoff value (we selected the std after analysing stars containing outliers)
        cut_off = data_std * 60

        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[(df[column] < data_mean - cut_off) | (df[column] > data_mean + cut_off)].index

        # appending the found outlier indices for column to the list of outlier indices
        outlier_indices.extend(outlier_list_column)

        for idx in outlier_list_column:
            outlier_counts[idx] += 1

    # Converto to Pandas Series, aligned with the original DataFrame
    outlier_series = pd.Series(0, index=df.index)
    for idx, count in outlier_counts.items():
        outlier_series.loc[idx] = count

    # selecting observations containing more than x outliers
    outlier_indices = cls.Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )


    return multiple_outliers, outlier_series


# detecting outliers and removing them
Outliers_StDev = StDev_method(exoTrain,1,exoTrain.columns.drop("LABEL").tolist())
print('The total number of stars with at least one outlier is',len(Outliers_StDev[0]))
exoTrain_out = exoTrain.drop(Outliers_StDev[0], axis = 0).reset_index(drop=True)

# PairPlot for first 5 columns after removing the outlier
custom_palette = {
    1: "lightgray",
    2: "red"
}

subset_data_out = exoTrain_out.iloc[:, [0] + list(range(1, 6))]
fig3 = sn.pairplot(pd.concat([subset_data_out[subset_data_out['LABEL'] != 2], subset_data_out[subset_data_out['LABEL'] == 2]]), hue='LABEL', palette=custom_palette)

# Kernel Density Estimate Plot (KDEplot) for FLUX.1 (w/o outliers)
fig2, ax = plt.subplots()
sn.kdeplot(data = pd.concat([subset_data_out[subset_data_out['LABEL'] != 2], subset_data_out[subset_data_out['LABEL'] == 2]]), hue = 'LABEL',x='FLUX.1', ax=ax, palette=custom_palette)


# Save as a png
fig1.savefig(str(current_path) + "/plots/pairplot.png")
fig2.savefig(str(current_path) + "/plots/kde1plot.png")
fig3.savefig(str(current_path) + "/plots/pairplot_out.png")
print("The three plots have been created with the first 5 fluxes of all stars")
