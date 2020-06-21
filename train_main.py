import pandas as pd
import numpy as np
# from pre_proc import *

def get_pre_data(type):
  data_df = pd.read_csv('data.csv', sep=',', header=None)
  param_sae = pd.read_csv('param_sae.csv', sep=',', header=None)
  
  porcentaje_train = param_sae.iloc[0][0]
  rows_number = len(data_df.index)
  train_number = int(np.round(porcentaje_train*rows_number))
  testing_number = rows_number - train_number

  # TODO reordenar aleatoriamente la data
  data_df_copy = data_df.sample(frac=1)

  if type == 'train':
    training_data = data_df_copy.iloc[0:train_number]
    return training_data
  elif type == 'test':
    testing_data = data_df_copy.iloc[train_number:]
    return testing_data
  
print(get_pre_data('test'))
print(get_pre_data('train'))
# print(percent_trainning)







