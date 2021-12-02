"""
Grupo 13:
- Sebastián Baeza M
- Andrés Guerrero H.
- Jorge Pacheco M.
- Juan Mansilla C. 
"""

import pandas as pd
import numpy as np


def pre_proc():
    """
    Funcion que retorna la data, las etiquetas y los parametros

    output: data_input, vectores_label, parametros
    """

    # Lectura de datos desde data.csv
    data_df = pd.read_csv("data.csv", sep=",", header=None)

    # Reordenar
    data_df_copy = data_df.sample(frac=1).reset_index(drop=True)

    L = len(data_df_copy.index)
    D = len(data_df_copy.columns)

    # data_input= N filas por [D-1]columnas
    data_input = data_df_copy.iloc[0:L, 0 : D - 1]

    # Label, contiene la última columna (257-1)
    data_label = data_df_copy.iloc[0:L, D - 1]

    # Identidad binaria
    matriz_identidad = np.identity(10)

    # Reemplazo en Columna  256, de cada etiqueta numérica a etiqueta Binaria.
    vectores_label = [matriz_identidad[i - 1] for i in data_label]

    vectores_label = pd.DataFrame(vectores_label)

    # Parámetros SAE
    param_sae = np.genfromtxt("param_sae.csv", delimiter=",")
    data_param_sae = {
        "percent_training": param_sae[0],
        "param_pinv": param_sae[1],
        "max_epoch": param_sae[2],
        "size_mini_batch": param_sae[3],
        "tasa_aprendizaje_mu": param_sae[4],
        "hidden_node_ae1": param_sae[5],
        "hidden_node_ae2": param_sae[6],
    }

    # Softmax
    param_softmax = np.genfromtxt("param_softmax.csv", delimiter=",")
    data_param_softmax = {
        "max_iteracion": param_softmax[0],
        "tasa_aprendizaje": param_softmax[1],
        "penalidad_pesos": param_softmax[2],
    }

    # Backpropagation
    param_bp = np.genfromtxt("param_bp.csv", delimiter=",")
    data_param_bp = {"max_iteracion": param_bp[0], "tasa_aprendizaje": param_bp[1]}

    parametros = {
        "sae": data_param_sae,
        "softmax": data_param_softmax,
        "bp": data_param_bp,
    }

    return data_input, vectores_label, parametros
