from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Reading the Data
dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\dbmodules\dm_restAPI_applicationmodule.csv')
dataframe= dataframe.dropna(axis=1, how= 'any')


#Label Encoding the Data
le= LabelEncoder()
dataframe_encoded = dataframe[['location','portfolio_id','technology_id']].apply(le.fit_transform)




#KMeans Clustering Model 
kmeans = KMeans(n_clusters=3, random_state=0).fit(dataframe_encoded)

dataframe['Cluster_Center']=""

for index in range(0,len(dataframe_encoded)):
    cluster_center = kmeans.predict(dataframe_encoded.iloc[[index]])
    dataframe['Cluster_Center'].loc[index]= int(cluster_center)   



#Standardising the Data
#dataframe_encoded = StandardScaler().fit_transform(dataframe_encoded)

#Using PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dataframe_encoded)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

FinalDataframe = pd.concat([principalDf, dataframe[['Cluster_Center']]], axis = 1)

#Plotting the graph
figure = plt.figure(figsize = (4,4))
ax = figure.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = FinalDataframe['Cluster_Center'] == target
    print(indicesToKeep)
    ax.scatter(FinalDataframe.loc[indicesToKeep, 'principal component 1']
               , FinalDataframe.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()


    
