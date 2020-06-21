"""
"""
import pandas as pd
import numpy as np


data = pd.read_csv('data.csv', sep=',', header=None)
data_input= data.iloc[0:1600, 0:256]

print(data_input)

data_label = np.zeros((1600, 10)) 
data_label_prev= data.iloc[0:1600, 256]

print("\n")
#df = pd.DataFrame(data_label)

#Llenado de vectores binarios
for j in range (0,1600):
    for pos in range (1,11):
        if data_label_prev.loc[j] == pos :
            data_label[j][pos-1]=1

print(data_label)


#Lectura de param_sae.csv
param_sae=np.genfromtxt('param_sae.csv',delimiter=',')

#PARÁMETROS SAE.CS
percent_trainning=param_sae[0]
param_pinv=param_sae[1]
max_epoch=param_sae[2]
size_mini_batch=param_sae[3]
tasa_aprendizaje_mu=param_sae[4]
hidden_node_ae1=param_sae[5]
hidden_node_ae2=param_sae[6]

print(percent_trainning,param_pinv,max_epoch,size_mini_batch,tasa_aprendizaje_mu,hidden_node_ae1,hidden_node_ae2)

#Lectura de param_sae.csv
param_softmax=np.genfromtxt('param_softmax.csv',delimiter=',')
max_iteracion=param_softmax[0]
tasa_aprendizaje=param_softmax[1]
penalidad_pesos=param_softmax[2]

print(max_iteracion,tasa_aprendizaje,penalidad_pesos)
 
