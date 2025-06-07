'''
LIGHT FLUX PLOTS OF STARS WITH AND WITHOUT EXOPLANETS
'''

# Import standard packages
import os
import pathlib
import numpy as np
import pandas as pd

# For Plotting Purpose
import matplotlib.pyplot as plt
import seaborn as sn

# Read the Train dataset and convert it to Pandas DataFrame
current_path = pathlib.Path().absolute()
exoTrain = pd.read_csv(os.path.join(current_path, "data/", "exoTrain.csv"))
exoTest = pd.read_csv(os.path.join(current_path, "data/", "exoTest.csv"))

# Showing the number of rows and columns of both datasets
print('The shape of the Trainset is (rows,columns)=', exoTrain.shape,'\n and the one of the Testset is (rows,columns)=', exoTest.shape)

# Printing the number of non-exoplanet-stars and exoplanet-stars
# 1 ---> No Exoplanet
# 2 ---> Exoplanet
print('Traning dataset:\n', exoTrain['LABEL'].value_counts())
print('Testing dataset:\n', exoTest['LABEL'].value_counts())

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



# Separate the dataset into predictive features and prediction target
X = exoTrain.drop(['LABEL'], axis = 1)
y = exoTrain['LABEL']

# Call the function
flux_graph(dataset = X, target = y)

# Save as a png
pathlib.Path(current_path / "plots").mkdir(parents=True, exist_ok=True)
plt.savefig(str(current_path) + "/plots/starfluxspectra.png")
print("plots/starfluxspectra.png has been created with the first 3 fluxes of each star classes")
