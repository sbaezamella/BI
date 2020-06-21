import pandas as pd
import numpy as np


def get_pre_data(type):
  data = pd.read_csv('data.csv', sep=',', header=None)
  param_sae = pd.read_csv('param_sae.csv', sep=',', header=None)
  
  porcentaje_train = param_sae.iloc[0][0]
  rows_number = len(data.index)
  train_number = int(np.round(porcentaje_train*rows_number))
  testing_number = rows_number - train_number

  # TODO reordenar aleatoriamente la data
  if type == 'train':
    training_data = data.iloc[0:train_number]
    return training_data
  elif type == 'test':
    testing_data = data.iloc[train_number:]
    return testing_data
  
print(get_pre_data('test'))







