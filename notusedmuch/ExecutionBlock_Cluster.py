from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#Reading and Cleaning the Data
dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\dbmodules\dm_restAPI_executionblock.csv')
dataframe= dataframe.dropna(axis=1, how= 'all')
dataframe["Cluster_Center"] = np.nan
dataframe = dataframe.fillna('0')

#Label Encoding the Data
le= LabelEncoder()
#dataframe['name'] = le.fit_transform(dataframe.name.values)
dataframe_encoded = dataframe[['statementCount','complexity','entryCondition_id','exitCondition_id']]
#KMeans Clustering Model 
kmeans = KMeans(n_clusters=5, random_state=0).fit(dataframe_encoded)

for index in range(0,len(dataframe)):
    cluster_center = kmeans.predict(dataframe_encoded.iloc[[index]])
    dataframe['Cluster_Center'].loc[index]= cluster_center    
    