import pandas as pd
import gensim
from sklearn.preprocessing import LabelEncoder
import collections
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\whse.csv')
dataframe= dataframe[['Data_Operation','Execution_Block_Name','Execution_Unit_ID']]
dataframe.drop_duplicates(keep='first', inplace= True)
dataframe= dataframe.reset_index(drop=True)
structure=''
structure_list= []
structure_dict=collections.OrderedDict()

for index in range(0,len(dataframe)-1):
    if dataframe['Execution_Unit_ID'].iloc[index]== dataframe['Execution_Unit_ID'].iloc[index+1]:
        if  dataframe['Data_Operation'].iloc[index]!='NIL':
            structure += " " + dataframe['Execution_Block_Name'].iloc[index] + ' DO '
        else:
            structure += " " + dataframe['Execution_Block_Name'].iloc[index]
    else:
        if  dataframe['Data_Operation'].iloc[index]!='NIL':
            structure += " " + dataframe['Execution_Block_Name'].iloc[index] + ' DO '
        else:
            structure += " " + dataframe['Execution_Block_Name'].iloc[index]
        
        structure_list.append(structure)
        structure_dict[dataframe['Execution_Unit_ID'].iloc[index]]= structure
        structure=''

#Cosine Similarity       
similarity_ratio= pd.DataFrame(columns=['ExUnit1','ExUnit2','Similarity'])
count=0
for i in structure_list:
    for j in structure_list:
        count+=1
        expressions = (i,j)
        tfidf_vectorizer=TfidfVectorizer(analyzer="char")
        tfidf_matrix=tfidf_vectorizer.fit_transform(expressions)
        cs=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
        similarity_ratio.loc[count, 'ExUnit1'] = i
        similarity_ratio.loc[count, 'ExUnit2'] = j
        similarity_ratio.loc[count, 'Similarity'] = cs
        print('similarities are',cs)
        
#Kmeans Algorithm
kmeans_dataframe= pd.DataFrame(columns=['ExUnit','Cluster_Center'])
from sklearn.cluster import KMeans
x = TfidfVectorizer().fit_transform(structure_list)
km = KMeans(n_clusters=3).fit(x)

count=0
for index in range(0,len(structure_list)):
    count+=1
    kmeans_dataframe.loc[count, 'ExUnit'] = structure_list[index]
    kmeans_dataframe.loc[count, 'Cluster_Center'] = km.predict(x[index])

        
        
