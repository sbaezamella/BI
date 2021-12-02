"""
Grupo 13:
- Sebastián Baeza M
- Andrés Guerrero H.
- Jorge Pacheco M.
- Juan Mansilla C. 
"""

import numpy as np
import pandas as pd

from utils import pre_proc

data_input, data_label, parametros = pre_proc()

# Constantes
AUTOENCODER_NUM = 2


def get_pre_data(type):
    param_sae = parametros["sae"]

    porcentaje_train = param_sae["percent_training"]
    rows_number = len(data_input.index)
    train_number = int(np.round(porcentaje_train * rows_number))

    if type == "train":
        training_data_input = data_input.iloc[0:train_number]
        training_data_label = data_label.iloc[0:train_number]
        return training_data_input, training_data_label

    elif type == "test":
        test_data_input = data_input.iloc[train_number:]
        test_data_label = data_label.iloc[train_number:]
        return test_data_input, test_data_label


def train_fine(a1, ye, a2, a3, mu, max_iter):

    for i in range(int(max_iter)):

        print(i)
        print(a1, a2, a3)
        error = np.subtract(a1, ye)
        costo = np.square(np.subtract(a1, ye)).mean()
        print(costo)
        derivateAct = sigmoid_derivate(ye)

        delta = np.dot(error, np.transpose(derivateAct))
        w2 = np.dot(np.transpose(a3 - mu), delta)
        error = np.dot(w2, delta)
        derivateAct = sigmoid_derivate(a2)
        w1 = np.dot(np.transpose(a2 - mu), delta)

        a2 = np.transpose(w1)
        a3 = np.transpose(w2)


def sigmoid_derivate(x):
    return x * (1 - x)


training_data_input, training_data_label = get_pre_data("train")

predict = pd.read_csv("resultado_prediccion.csv", header=None)
predict = np.array(predict)

max_iteracion = parametros["bp"]["max_iteracion"]
tasa_aprendizaje = parametros["bp"]["tasa_aprendizaje"]

listaAE = []

training_data_label = np.array(training_data_label)
listaAE.append(training_data_label)
listaAE.append(predict)

for i in range(AUTOENCODER_NUM):
    aux = pd.read_csv(f"pesos{str(i+1)}.csv")
    aux = np.array(aux, dtype=float)
    listaAE.append(aux)

a1 = listaAE[0]
ye = listaAE[1]
a2 = listaAE[2]
a3 = listaAE[3]


train_fine(a1, ye, a2, a3, tasa_aprendizaje, max_iteracion)
