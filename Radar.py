import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Radar_Traffic_Counts.csv")
df_new = df.copy()


#Transform object data to integer classes

mappings = {}

for i in range(df_new.shape[1]):
    if df_new.iloc[:,i].dtypes == 'O':
        labels_list=list(df_new.iloc[:,i].unique())
        mapping = dict(zip(labels_list,range(len(labels_list))))
        mappings[df_new.columns[i]] = (mapping)
        df_new.iloc[:,i] = df_new.iloc[:,i].map(mapping)


#divide data into features and labels
        
X = df_new.drop("Volume", axis = 1)
y = df_new.Volume

scaler = StandardScaler()
X_scaled = scaler.fit(X)

