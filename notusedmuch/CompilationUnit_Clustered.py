from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score




#Reading the Data
dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\dbmodules\dm_restAPI_compilationunit.csv')
dataframe= dataframe.dropna(axis=1, how= 'all')
dataframe = dataframe.fillna('0')
#Converting object data column to int 
dataframe['statementCount'] = dataframe['statementCount'].astype(int)

#Label Encoding the Data
le= LabelEncoder()

dataframe_encoded = dataframe[['statementCount','complexity']].apply(le.fit_transform)
X= dataframe_encoded
y= dataframe['module_id']
#KMeans Clustering Model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))  
    
