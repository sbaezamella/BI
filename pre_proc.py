import pandas as pd
import numpy as np


def pre_proc():
    data_df = pd.read_csv('data.csv', sep=',', header=None)

    data_df = data_df.sample(frac=1).reset_index(drop=True)

    data_input = data_df.iloc[0:1600, 0:256]

    # Ultima columna (etiquetas) 
    data_label = data_df.iloc[0:1600, 256]

    binarios = np.identity(10)

    aux = []
    for i in data_label:
        aux.append(binarios[i-1])

    aux = pd.DataFrame(aux)

    aux.to_csv('data_label.csv', index=False)

    vectores_label = pd.read_csv('data_label.csv') 



    ### PARÁMETROS
    ## SAE
    param_sae = np.genfromtxt('param_sae.csv', delimiter=',')
    data_param_sae = {
        'percent_training': param_sae[0],
        'param_pinv': param_sae[1],
        'max_epoch': param_sae[2],
        'size_mini_batch': param_sae[3],
        'tasa_aprendizaje_mu': param_sae[4],
        'hidden_node_ae1': param_sae[5],
        'hidden_node_ae2': param_sae[6]
    }

    # Softmax
    param_softmax = np.genfromtxt('param_softmax.csv', delimiter=',')
    data_param_softmax = {
        'max_iteracion': param_softmax[0],
        'tasa_aprendizaje': param_softmax[1],
        'penalidad_pesos': param_softmax[2]
    }

    # Backpropagation
    param_bp = np.genfromtxt('param_bp.csv', delimiter=',')
    data_param_bp = {
        'max_iteracion': param_bp[0],
        'tasa_aprendizaje': param_bp[1]
    }

    parametros = {
        'sae': data_param_sae,
        'softmax': data_param_softmax,
        'bp': data_param_bp
    }

    return data_input, vectores_label, parametros
