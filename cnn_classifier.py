'''
CONVOLUTIONAL NEURAL NETWORK CLASSIFIER TRAINED TO CLASSIFY BETWEEN TWO SKEWED CLASSES
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

# Scikit-learn, model evaluation
from sklearn.utils import shuffle
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

# Tensorflow, model creation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import EarlyStopping


# Load pre-processed data
current_path = pathlib.Path().absolute()
with open("data/preprocessed_data.pkl", "rb") as f:
    x_train, y_train, x_test, y_test = pickle.load(f)

# ------------------------- USEFUL FUNCTIONS -------------------------------

# Printing the Confusion matrix
def plot_confusion_matrix(y_test, y_pred):

    plt.rcParams["font.family"] = "Times"
    # Create the matrix with real and predicted flux values
    matrix = confusion_matrix(y_test, y_pred,normalize='true')
    # Plot the confusion matrix
    df = pd.DataFrame(matrix, columns=[0, 1], index = [0, 1])
    df.index.name = 'Real Values'
    df.columns.name = 'Predicted Values'
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    sn.heatmap(df, cmap="BuGn", annot=True, ax=ax)

    return matrix, fig
    
# Graph train and test accuracy
def graph_acc(history):

    plt.rcParams["font.family"] = "Times"
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    
    # Plot loss during training with train set and with predictions made with the validation set
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot recall during training with train set and with predictions made with the validation set
    ax2.set_title('Recall')
    ax2.plot(history.history['recall'], label='Train')
    ax2.plot(history.history['val_recall'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall')
    ax2.legend()
    
    fig.tight_layout()
    
    return fig

# Print prediction metrics
def display_predictions(y_test, y_pred, y_class_pred, matrix):

  #TP ---> TRUE POSITIVE
  #TN ---> TRUE NEGATIVE
  #FP ---> FALSE POSITIVE
  #FN ---> FALSE NEGATIVE
  TN = matrix[0][0]
  TP = matrix[1][1]
  FN = matrix[0][1]
  FP = matrix[1][0]


  # Recall
  rec = TP/(TP+FN)
  # Accuracy (not preferred for classification problems)
  accuracy = (TP+TN)/(TP+FP+TN+FN)
  # Precision
  precision = TP/(TP+FP)
  # F1 Score (preferred): harmonic mean of precision and recall
  f1 = (2*precision*rec)/(precision+rec)
  # ROC curve (Area under Receiver Operating Characteristic curve)
  auc = roc_auc_score(y_test, y_pred)

  # Print metrics
  print('\t\t Prediction Metrics\n')
  print("Accuracy:\t", "{:0.4f}".format(accuracy))
  print("Precision:\t", "{:0.4f}".format(precision))
  print("Recall:\t\t", "{:0.4f}".format(rec))
  print("ROC AUC:\t", "{:0.4f}".format(auc))
  print("F1:\t", "{:0.4f}".format(f1))
  


# ----------------------- DATA PREPARATION ---------------------------

# shuffle the data to try to avoid 0.0000e+00 val_recall (classes are ordered)
x_train, y_train = shuffle(x_train, y_train)

# Define number of features that will be introduced in the CNN
n_features = x_train.shape[1]

# Classes are still unbalanced so we impose a class weight to give a higher representation to the minority class
class_weight = {
    0: 1.0,  # weight for class 0
    1: 2.5,  # weight for class 1 (minoritary class)
    }

# ------------------------ DEEP LEARNING MODEL ----------------------------

# Architecture
# Sequential allows stacking multiple layers sequentially
model = keras.Sequential()
# In order to make a convolutional network, we need to use a dataset with 3 dimensions: data, number of imputs, number of features per imput.
model.add(keras.Input(shape=(n_features,1)))
model.add(layers.Normalization())
model.add(layers.Conv1D(filters=11, kernel_size=2, activation='relu', kernel_regularizer='l2'))
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(filters=7, kernel_size=2, activation='relu', kernel_regularizer='l2'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2, strides=2))
model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(30, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# Representation of architecture
print(model.summary())

# Compile model specifying the learning rate of Adam optimizer
lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.8)
model.compile(optimizer = Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=[metrics.Recall()])

# Add an early stop to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Fit model to the train data selecting validation split of 20%, 128 batch sample, max 40 epochs, adding class weight defined before
history = model.fit(x_train, y_train, validation_split = 0.2, batch_size=128, callbacks=[early_stop], epochs=40, verbose=2, class_weight=class_weight)


# -------------------- TRAINING VALIDATION ---------------------
print('Training Prediction:')

# Sigmoid output (probability)
y_val = model.predict(x_train)

# Choose a threshold to maximize F1 (to obtain a balance between precision and recall):
precision, recall, thresholds = precision_recall_curve(y_train, y_val)

f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[f1_scores.argmax()]

print(f"Best threshold for F1: {best_threshold:.2f}")
# Obtain validation predictions considering the obtained threshold
y_val_adj = (y_val > best_threshold).astype(int)
    
# Calculating the Confusion Matrix in train dataset validation
matrix, fig1 = plot_confusion_matrix(y_train, y_val_adj)

# Displaying the Output Predictions
display_predictions(y_train, y_val, y_val_adj, matrix)


# -------------------- TEST PREDICTIONS -----------------------

print('Testing Prediction:')

y_pred = model.predict(x_test)
# Obtain predictions considering the threshold chosen above
y_pred_adj = (y_pred > best_threshold).astype(int)

# Confusion matrix in test dataset
matrix, fig2 = plot_confusion_matrix(y_test, y_pred_adj)

# Recall, loss evolution plot during epochs
fig3 = graph_acc(history)

# Test Metrics
display_predictions(y_test, y_pred, y_pred_adj, matrix)


# Save as a png confusion matrix and train evolution
fig1.savefig(str(current_path) + "/plots/CMtrain.png")
fig2.savefig(str(current_path) + "/plots/CMtest.png")
fig3.savefig(str(current_path) + "/plots/Trainevo.png")
    
print("The confusion matrices and training evolution have been created and saved in 'plots' folder ")
