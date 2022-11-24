# Data preprocessing for PyTorch binary classification tasks, provides:
# - Creation of datasets with appropriate labelling and dimensionality, given "raw" datasets
# - Input scaling
# - Creation of train/validation/test datasets
# - Transformation to tensor formats for later ingestion

import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class BinaryClassificationPreprocessor:

  def __init__(self):
      self.scaler = StandardScaler()  # We use standardization as the default option

  def train_val_test_split(self, X, y, val_ratio, test_ratio):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=True)
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=len(X)*val_ratio/len(X_train), shuffle=True)

      return X_train, X_val, X_test, y_train, y_val, y_test

  def standardize(self, X):
      # Transform all values in each row of X according to the (already fitted) self.scaler
      # S is the length of each time series (which is 1 when we only have one long series in X)
      return self.scaler.transform(X.ravel().reshape(-1,1)).reshape(X.shape)
  
  def create_tensor_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
      train_input = torch.Tensor(X_train)
      train_output = torch.Tensor(y_train)
      val_input = torch.Tensor(X_val)
      val_output = torch.Tensor(y_val)
      test_input = torch.Tensor(X_test)
      test_output = torch.Tensor(y_test)

      return (TensorDataset(train_input, train_output), TensorDataset(val_input, val_output), TensorDataset(test_input, test_output))
  
  def split_and_transform(self, X, y, val_ratio=0.2, test_ratio=0.2):
      
      (X_train, X_val, X_test, y_train, y_val, y_test) = self.train_val_test_split(X, y, val_ratio, test_ratio)

      self.scaler.fit_transform(X_train.ravel().reshape(-1,1))  # Only fit on the train set
      X_train_std = self.standardize(X_train)
      X_val_std = self.standardize(X_val)
      X_test_std = self.standardize(X_test)
      if len(y_train.shape) == 1:                # If y only contains one label for each time series (e.g., when using load_data_by_class)
          y_train = y_train.reshape(-1, 1)       # We need to reshape to get y in the same form as X
          y_val = y_val.reshape(-1, 1)
          y_test = y_test.reshape(-1, 1)
      else:
          y_train = y_train[...,np.newaxis]      # Need to add one dimension to match the model output later (y == y_pred)
          y_val = y_val[...,np.newaxis]
          y_test = y_test[...,np.newaxis]
      
      print("Preprocessed datasets:")
      print(f"Train: {X_train_std.shape}, range [{np.min(X_train_std)}, {np.max(X_train_std)}]")
      print(f"Validation: {X_val_std.shape}, range [{np.min(X_val_std)}, {np.max(X_val_std)}]")
      print(f"Test: {X_test_std.shape}, range [{np.min(X_test_std)}, {np.max(X_test_std)}]")

      # Finally, prepare for PyTorch usage
      return self.create_tensor_datasets(X_train_std, X_val_std, X_test_std, y_train, y_val, y_test)

  def load_data_by_class(self, X, Y, val_ratio=0.2, test_ratio=0.2):
      # X should be a list of time series belonging to class 1 and Y series belonging to class 2
      # X.shape[0] = # series, X.shape[1] = # features, X.shape[2] = length of each series

      print("Process 1: using", X.shape[0], "time series with", X.shape[1], "features, of length", X.shape[2])
      print("Process 2: using", Y.shape[0], "time series with", Y.shape[1], "features, of length", Y.shape[2])
              
      X_merge = np.concatenate((X, Y), axis=0)
      y_merge = np.concatenate((np.ones(len(X)), np.zeros(len(Y))), axis=0)
      
      return self.split_and_transform(X_merge, y_merge, val_ratio=val_ratio, test_ratio=test_ratio)

  def load_test_data(self, X, y):
      # Load separate test dataset (X, y), apply same scaling as was done for the current training set

      X = self.standardize(X)
      if len(y.shape) == 1:          # If y only contains one label for each time series
          y = y.reshape(-1, 1)       # We need to reshape to get y in the same form as X
      else:
          y = y[...,np.newaxis]      # Need to add one dimension to match the model output later (y == y_pred)

      return TensorDataset(torch.Tensor(X), torch.Tensor(y))

  def load_continuous_data(self, X, labels, L, W, test_ratio=0.2):
      # X should be a long sequence of continuous data belonging to both class 1 and class 2
      # labels should be the desired output for each timestep in X
      # L is the length of each subsequence that will be created from X
      # W is the window size for RNN processing (i.e., we get one output for every W input steps)
      # X.shape[0] = length of entire sequence, X.shape[1] = # features

      N = len(X) // L       
      X_split = np.zeros((N, X.shape[1], L))
      y_split = np.zeros((N, L // W))
      for i in range(N):
          start = i * L
          X_split[i,:,:] = X[start:start+L,:].reshape(1,-1)
          y_split[i,:] = labels[start:start+L:W]

      print(X_split.shape, y_split.shape)
      return self.split_and_transform(X_split, y_split, test_ratio=test_ratio)