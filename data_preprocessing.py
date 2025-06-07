'''
DATASETS PREPROCESSING BEFORE INTRODUCING THEM TO CNN
'''

# Import standard packages
import os
import pathlib
import numpy as np
import pandas as pd
import pickle

# For Plotting Purpose and statistics
import matplotlib.pyplot as plt
import seaborn as sn

# For Outliers Search
import collections as cls

# For Imbalanced Data treatment
import imblearn.over_sampling as ovsm
import imblearn.pipeline as ppl
import imblearn.under_sampling as unsm

# Data smoothing and scaling
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

# Read the Train and Test dataframes
current_path = pathlib.Path().absolute()
exoTrain = pd.read_csv(os.path.join(current_path, "data/", "exoTrain.csv"))
exoTest = pd.read_csv(os.path.join(current_path, "data/", "exoTest.csv"))

# Print the number of empty flux values in Train and Test datasets
print('The number of missing values in the Train dataset is:', exoTrain.isna().sum().sum())
print('The number of missing values in the Test dataset is:', exoTest.isna().sum().sum())

# Function to plot each star (row) flux into a different figure
def flux_graph(dataset, target):
  # Title, labels, size...
  plt.rcParams["font.family"] = "Times"
  plt.style.use('dark_background')
  fig, axs = plt.subplots(6, 1)
  fig.set_size_inches(18.5, 16.5)

    
  # Arrays of indexes considering the two cases (taking only the first 3 stars)
  w_planet = target[target == 2].head(3).index
  wo_planet = target[target == 1].head(3).index
  
  # Combining indices from both classes
  all_rows = list(w_planet) + list(wo_planet)
  labels = ['At least One Exoplanet'] * 3 + ['No Exoplanet'] * 3

  # Iterate through the 6 stars we are considering
  for i, row in enumerate(all_rows):
    # Obtain flux values depending on the type of dataset
    # Pandas DataFrame
    if isinstance(dataset, pd.DataFrame):
        flux_values = dataset.iloc[row]
    # Numpy Array, List...
    else:
        flux_values = dataset[row]
        
    # Plot flux values in terms of time units
    axs[i].plot(range(1, len(flux_values) + 1), flux_values, color='#00ffff')
    axs[i].tick_params(colors='#00ffff', labelsize=14)
    axs[i].set_title(f"{labels[i]} in Star_{row}", fontsize=16, color='white')
    axs[i].set_xlabel('Time units', fontsize=14, color='white')
    axs[i].set_ylabel('Light intensity', fontsize=14, color='white')
  
  # Visualization on final png
  plt.tight_layout()

# Outlier detection method
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

    # selecting observations containing more than x outliers (list of indices)
    outlier_indices = cls.Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )


    return multiple_outliers, outlier_series

#Changing the labels from (1 ---> 0) and (2 ---> 1)
def change_labels(y_train, y_test):
    label_changer = lambda x: 1 if x == 2 else 0
    y_train_temp = y_train.apply(label_changer)
    y_test_temp = y_test.apply(label_changer)

    return y_train_temp, y_test_temp
    
# Handling the Imbalance of datasets by oversampling the class 1 and undersampling the class 0
def smote(x_train, y_train):
    over = ovsm.SMOTE(sampling_strategy=0.1)
    under = unsm.RandomUnderSampler(sampling_strategy=0.2)
    steps = [('o', over), ('u', under)]
    # First we oversample the minority class to be 1/10 of the majority class and then we undersample the majority one to have 1/5 of minority class wrt majority class
    pipeline = ppl.Pipeline(steps=steps)
    x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)

    return x_train_res, y_train_res
 
# Printing the new dimensions of each class after under- and over-sampling
x_train_res, y_train_res = smote(exoTrain.drop(['LABEL'], axis = 1), exoTrain['LABEL'])
y_train_res.reset_index(drop=True, inplace=True)

print(y_train_res.value_counts())

##############################################################################################
# Pre-processing data

x_train, y_train = exoTrain.loc[:, exoTrain.columns != 'LABEL'], exoTrain.loc[:, 'LABEL']
x_test, y_test = exoTest.loc[:, exoTest.columns != 'LABEL'], exoTest.loc[:, 'LABEL']

# Detecting and dropping outliers from the Train dataset
Outliers_StDev = StDev_method(exoTrain,1,exoTrain.columns.drop("LABEL").tolist())
x_train = x_train.drop(Outliers_StDev[0], axis = 0).reset_index(drop=True)
y_train = y_train.drop(Outliers_StDev[0], axis = 0).reset_index(drop=True)

# Under-sampling the majority class using RandomUnderSampler and over-sampling the minority one with SMOTE in Train Dataset
x_train, y_train = smote(x_train, y_train)
#Changing the labels (1 --> 0) and (2 --> 1) for both datasets
y_train, y_test = change_labels(y_train, y_test)

###############################################################################################

# Performing a gaussian filter and Standar Scaler to visually see how the spectra change (they performed worse than the ones without applying them)
flux_columns = [col for col in exoTrain.columns if "FLUX" in col]
x_train1 = exoTrain.drop(['LABEL'], axis = 1)[flux_columns].apply(lambda row: gaussian_filter1d(row, sigma=3), axis=1, result_type='expand')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x_train1)

flux_graph(dataset = scaled_data, target = exoTrain['LABEL'])


# Save as a png
plt.savefig(str(current_path) + "/plots/spectra_gausSS.png")

# Save x_train, y_train, x_test, y_test as a pickle
with open("data/preprocessed_data.pkl", "wb") as f:
    pickle.dump((x_train, y_train, x_test, y_test), f)
    
print("The plots have been created and the pre-processed data have been saved")
