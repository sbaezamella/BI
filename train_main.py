import pandas as pd
import numpy as np
from pre_proc import *

data_input, data_label,parametros = pre_proc()


def get_pre_data(type):
    param_sae = parametros['sae']

    porcentaje_train = param_sae['percent_training']
    rows_number = len(data_input.index)
    train_number = int(np.round(porcentaje_train*rows_number))
    testing_number = rows_number - train_number

 
    np.random.shuffle(data_label)
    data_df_copy = data_input.sample(frac=1)



    training_data_label= np.zeros((train_number, 10))
    test_data_label=np.zeros((testing_number, 10))
    if type == 'train_input':
        training_data_input = data_df_copy.iloc[0:train_number]
        return training_data_input

    elif type == 'test_input':
        testing_data_input = data_df_copy.iloc[train_number:]
        return testing_data_input

    elif type == 'train_label':
        for i in range(0, train_number):
            for j in range(0,10):
                training_data_label[i][j]= data_label[i][j]
        return training_data_label

    elif type == 'test_label':
        for i in range(train_number,rows_number):
            for j in range(0,10):
                test_data_label[i-1280][j]=data_label[i][j]
        return test_data_label




print(get_pre_data('train_input'))
print(get_pre_data('test_input'))
print(get_pre_data('train_label'))
print(get_pre_data('test_label'))