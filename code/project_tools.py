from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.95
RANDOM_STATE = 13
TIME_STEP = 15
CURRENT_DATE = "2026-03-29"
MODEL_PATH = "../models"
DATA_PATH = "../data"
IMAGE_PATH = "../visualisations"


def create_windows(data, time_step=TIME_STEP):
  data = np.array(data)
  X, y = [], []

  for i in range(len(data) - time_step - 1):
    X.append(data[i:(i + time_step), 0])
    y.append(data[i + time_step, 0])

  X, y = np.array(X), np.array(y)
  X = X.reshape(X.shape[0], X.shape[1], 1)

  return X, y


def split_and_scale_data(X, y, train_split, val_split):
  train_inx = int(train_split * len(X))
  val_inx = int(val_split * len(X))

  X_train, X_val, X_test = X[:train_inx], X[train_inx:val_inx], X[val_inx:]
  y_train, y_val, y_test = y[:train_inx], y[train_inx:val_inx], y[val_inx:]

  scaler_X = MinMaxScaler()
  scaler_y = MinMaxScaler()

  X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).reshape(-1, TIME_STEP, 1)
  X_val_scaled = scaler_X.transform(X_val.reshape(-1, 1)).reshape(-1, TIME_STEP, 1)
  X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).reshape(-1, TIME_STEP, 1)

  y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
  y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1)
  y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

  joblib.dump(scaler_X, f"{MODEL_PATH}/scaler_X.save")
  joblib.dump(scaler_y, f"{MODEL_PATH}/scaler_y.save")

  return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled