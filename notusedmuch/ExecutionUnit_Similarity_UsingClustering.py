# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:43:02 2019

@author: u65988
"""

import pandas as pd
import gensim
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans



dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\whse.csv')
dataframe= dataframe[['Data_Operation','Execution_Block_Name','Execution_Unit_ID']]
dataframe.drop_duplicates(keep='first', inplace= True)
dataframe= dataframe.reset_index(drop=True)

le= LabelEncoder()
dataframe_encoded = dataframe[['Data_Operation','Execution_Block_Name']].apply(le.fit_transform)
dataframe_encoded['Execution_Unit_ID']= dataframe['Execution_Unit_ID']


kmeans = KMeans(n_clusters=4, random_state=0).fit(dataframe_encoded)
dataframe['Cluster_Center']=""

for index in range(0,len(dataframe)):
    cluster_center = kmeans.predict(dataframe_encoded.iloc[[index]])
    dataframe['Cluster_Center'].loc[index]= int(cluster_center) 