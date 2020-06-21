import pandas as pd
import numpy as np
from pre_proc import *

data_input, parametros = pre_proc()


def get_pre_data(type):
    param_sae = parametros['sae']

    porcentaje_train = param_sae['percent_training']
    rows_number = len(data_input.index)
    train_number = int(np.round(porcentaje_train*rows_number))
    testing_number = rows_number - train_number


    data_df_copy = data_input.sample(frac=1)

    if type == 'train':
        training_data = data_df_copy.iloc[0:train_number]
        return training_data
    elif type == 'test':
        testing_data = data_df_copy.iloc[train_number:]
        return testing_data


print(get_pre_data('test'))
print(get_pre_data('train'))
