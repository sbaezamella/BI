import pandas as pd
import numpy as np
from pre_proc import *

data_input, data_label, parametros = pre_proc()

# Constantes
AUTOENCODER_NUM = 2


def get_pre_data(type):
    param_sae = parametros['sae']

    porcentaje_train = param_sae['percent_training']
    rows_number = len(data_input.index)
    train_number = int(np.round(porcentaje_train*rows_number))
    testing_number = rows_number - train_number

    if type == 'train':
        training_data_input = data_input.iloc[0:train_number]
        training_data_label = data_label.iloc[0:train_number]
        return training_data_input, training_data_label

    elif type == 'test':
        test_data_input = data_input.iloc[train_number:]
        test_data_label = data_label.iloc[train_number:]
        return test_data_input, test_data_label


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def softmax(z):

    N, nc = np.shape(z)

    z = np.array(z, dtype=float)
    y = z
    suma = 0
    for i in range(N):
        for j in range(nc):
            for k in range(nc):
                suma = suma + np.exp(z[i][k])
            y[i][j] = np.exp(z[i][j]) / suma
            suma = 0

    return y


def actualizar_peso(T, Y, X, mu, lamd, w, N):
    mu = (mu)
    lamd = (lamd)
    X = np.transpose(X)
    resta = np.subtract(T, Y)
    resta = np.transpose(resta)
    mult = np.dot(resta, X)
    gradiente = ((-1/N) * mult) + lamd*w
    wnew = w-mu*(gradiente)
    return wnew


def train(parametros):
    listaCosto = []

    training_data_input, training_data_label = get_pre_data('train')

    nh = [parametros['sae']['hidden_node_ae1'],
          parametros['sae']['hidden_node_ae2']]

    param_inv = parametros['sae']['param_pinv']

    lista = []
    for i in range(AUTOENCODER_NUM):

        hidden_nodes = nh[i]

        ni, N = training_data_input.shape

        r = np.sqrt(6/(ni+hidden_nodes))
        w = np.random.rand(int(hidden_nodes), int(ni)) * 2 * r - r
        x = training_data_input

        b = np.random.rand(int(hidden_nodes), 1) * 2 * r - r

        x = np.array(x, dtype=float)
        w = np.array(w, dtype=float)
        b = np.array(b, dtype=float)
        capa_salida = np.dot(w, x)+b

        # ENCODER
        H = capa_salida
        H = sigmoid(H)

        # PESOS DECODER
        w_aux1 = np.dot(x, np.transpose(H))
        w_aux2 = np.dot(H, np.transpose(H)) + (1/param_inv)
        w_aux2 = np.linalg.inv(w_aux2)
        w = np.dot(w_aux1, w_aux2)

        training_data_input = w
        w = pd.DataFrame(w)
        lista.append(w)
        # w.to_csv("deepl_pesos"+str(i+1)+".csv")

    x = lista[AUTOENCODER_NUM-1]

    x = np.transpose(x)
    nc, N = x.shape

    training_data_label = np.array(training_data_label, dtype=float)
    nmuestras, nclases = training_data_label.shape
    r = np.sqrt(6/(nc+nclases))
    w = np.random.rand(nclases, nc) * 2 * r - r
    z = np.dot(w, x)
    z = np.transpose(z)
    max_iteracion = parametros['softmax']['max_iteracion']
    tasa_aprendizaje = parametros['softmax']['tasa_aprendizaje']
    penalidad_pesos = parametros['softmax']['penalidad_pesos']

    for i in range(int(max_iteracion)):
        print(f'Iteración numero: {i}')
        y = softmax(z)
        w = actualizar_peso(training_data_label, y, x,
                            tasa_aprendizaje, penalidad_pesos, w, N)

        z = np.dot(w, x)
        z = np.transpose(z)

        suma1 = 0
        for i in range(nmuestras):
            for j in range(nclases):
                suma1 = suma1 + (training_data_label[i][j] * np.log(y[i][j]))

        wnorma = (np.linalg.norm(w))

        costo = -(1/N) * suma1 + ((penalidad_pesos/2)*(wnorma))
        print(f'Costo: {costo}')
        # listaCosto.append(costo)

        # with open("deepl_costo.csv", "w") as f:
        #     wr = csv.writer(f, delimiter="\n")
        #     wr.writerow(listaCosto)
        suma1 = 0

    # y = pd.DataFrame(y)
    # y.to_csv("predict.csv", index=False, header=None)


train(parametros)
