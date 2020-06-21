import pandas as pd
import numpy as np


data=np.genfromtxt('data.csv',delimiter=',')

# Data input
data_input = data.iloc[0:1600, 0:256]
print(data_input)

# Data label
data_label= data.iloc[0:1600, 256]
print(data_label)

