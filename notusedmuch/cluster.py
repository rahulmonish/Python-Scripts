

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\dbmodules\dm_restAPI_applicationmodule.csv')
dataframe= dataframe.dropna(axis=1, how= 'any')

le= LabelEncoder()
dataframe_encoded = dataframe.apply(le.fit_transform)


kmeans = KMeans(n_clusters=5, random_state=0).fit(dataframe_encoded)

dataframe['Cluster_Center']=""

for index in range(0,len(dataframe_encoded)):
    
    cluster_center = kmeans.predict(dataframe_encoded.iloc[[index]])
    dataframe['Cluster_Center'].loc[index]= cluster_center    