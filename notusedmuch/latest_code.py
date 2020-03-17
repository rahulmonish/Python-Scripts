# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import copy
from sklearn.preprocessing import StandardScaler
import tensorflow_hub as hub


import pickle

pickle_in = open(r"C:\Users\u65988\Downloads\db2.pickle","rb")
db2 = pickle.load(pickle_in)

pickle_in = open(r"C:\Users\u65988\Downloads\notdb2.pickle","rb")
notdb2 = pickle.load(pickle_in)

pickle_in = open(r"C:\Users\u65988\Downloads\dataframe_db.pickle","rb")
df1 = pickle.load(pickle_in)
pickle_in2 = open(r"C:\Users\u65988\Downloads\dataframe_notdb_new.pickle","rb")
df3=  pickle.load(pickle_in2)

pickle_in3 = open(r"C:\Users\u65988\Downloads\df.pickle","rb")
df4=  pickle.load(pickle_in3)






dataframe= pd.read_csv(r'D:\Datasets\cluster_latest.csv')
dataframe= dataframe[['Block ID', 'Expression Type ID']]
#dataframe.drop_duplicates(keep='first', inplace= True)
dataframe = dataframe.dropna(axis=0, subset=['Block ID','Expression Type ID'])
dataframe= dataframe.fillna('NIL')
structure_list=[]
structure=''
structure_dict={}


#Getting all the execution Block Structures
for index in range(0,len(dataframe)-1):
    if dataframe['Block ID'].iloc[index]== dataframe['Block ID'].iloc[index+1]:
        structure += " " + str(int(dataframe['Expression Type ID'].iloc[index]))
    else:
        structure += " " + str(int(dataframe['Expression Type ID'].iloc[index]))
        structure_list.append(structure)
        structure_dict[dataframe['Block ID'].iloc[index]]= structure
        structure=''

print("There are ", len(structure_list)," no of Execution Blocks")


#Creating Execution Unit Structure
dataframe_new= pd.read_csv(r'D:\Datasets\cluster_latest.csv')
dataframe_new= dataframe_new[['Execution Unit ID', 'Block ID', 'Symbol Type Name', 'Symbol Datatype']]
dataframe_new = dataframe_new.dropna(axis=0, subset=['Execution Unit ID'])


dataframe_final = pd.DataFrame(index= range(0,len(dataframe_new)), columns= ['Execution Unit ID', 'Symbol Entry Datatype', 'Input Datatype'])
input_data=''
symbol_entry=''
count=0
for index in range(0,len(dataframe_new)-1):
    
    if dataframe_new['Execution Unit ID'].iloc[index]== dataframe_new['Execution Unit ID'].iloc[index+1]:
        if dataframe_new['Symbol Type Name'].iloc[index]=='INPUT_DATA':
            input_data+= str(dataframe_new['Symbol Datatype'].iloc[index])
        
        elif dataframe_new['Symbol Type Name'].iloc[index]=='SYMBOL_ENTRY':
            symbol_entry+= str(dataframe_new['Symbol Datatype'].iloc[index])
    
    else:
        dataframe_final['Execution Unit ID'].iloc[count]= dataframe_new['Execution Unit ID'].iloc[index]
        dataframe_final['Symbol Entry Datatype'].iloc[count]= symbol_entry
        dataframe_final['Input Datatype'].iloc[count]= input_data
        count+=1
        input_data= symbol_entry= ''
    
    
            
#Merging with Execution Block structure

dataframe= pd.read_csv(r'D:\Datasets\cluster_latest.csv')
dataframe = dataframe.dropna(axis=0, subset=['Block ID'])
dataframe= dataframe[['Execution Unit ID','Block ID', 'Symbol Datatype', 'Data Operation Type', 'Data Source Type', 'Data Source Infrastructure', 'Data Source System', 'Data Source Name', 'Data Field Type', 'Data Field Name']]
dataframe= dataframe.fillna('NIL')
dataframe_latest= pd.DataFrame(index= range(0,len(dataframe)), columns= ['Execution Unit ID','Block ID', 'Symbol Datatype', 'Data Operation Type', 'Data Source Type', 'Data Source Infrastructure', 'Data Source System', 'Data Source Name', 'Data Field Type', 'Data Field Name'])
data_operation=''
count=0
flag=0

for index in range(0,len(dataframe)-1):
    if dataframe['Block ID'].iloc[index]== dataframe['Block ID'].iloc[index+1]:
        if dataframe['Data Operation Type'].iloc[index]!= 'NIL':
            dataframe_latest.iloc[count]= dataframe.iloc[index]
            flag=1
            count+=1
            #data_operation+= str(dataframe['Data Operation Type'].iloc[index]) + str(dataframe['Data Source Type'].iloc[index]) + str(dataframe['Data Source Infrastructure'].iloc[index]) + str(dataframe['Data Source System'].iloc[index]) + str(dataframe['Data Source Name'].iloc[index]) + str(dataframe['Data Field Type'].iloc[index]) + str(dataframe['Data Field Name'].iloc[index])
    else:
         if dataframe['Data Operation Type'].iloc[index]!= 'NIL':
             dataframe_latest.iloc[count]= dataframe.iloc[index]
             flag=1
             count+=1
             #data_operation+= str(dataframe['Data Operation Type'].iloc[index]) + str(dataframe['Data Source Type'].iloc[index]) + str(dataframe['Data Source Infrastructure'].iloc[index]) + str(dataframe['Data Source System'].iloc[index]) + str(dataframe['Data Source Name'].iloc[index]) + str(dataframe['Data Field Type'].iloc[index]) + str(dataframe['Data Field Name'].iloc[index])
        
         if flag==0:
            dataframe_latest.iloc[count]= dataframe.iloc[index]
            count+=1
         else:
            flag=0


#consolidating all the data into a single dataframe
dataframe_latest['Block Structure']= np.nan
dataframe_latest['Input Datatype']= np.nan
dataframe_final= dataframe_final.dropna(how= 'all')
dataframe_latest= dataframe_latest.dropna(how= 'all')
for index in range(0,len(dataframe_latest)):
    
    unit_id= dataframe_latest['Execution Unit ID'].iloc[index]
    row = dataframe_final[dataframe_final['Execution Unit ID']==unit_id].index.values.astype(int)[0]
    dataframe_latest['Input Datatype'].iloc[index]= dataframe_final['Input Datatype'][row]
    if int(dataframe['Block ID'].iloc[index]) in structure_dict:
        dataframe_latest['Block Structure'].iloc[index]= structure_dict[int(dataframe['Block ID'].iloc[index])]



#Splitting the execution Units with DB and without DB
dataframe_latest= dataframe_latest.sort_values('Data Operation Type', ascending= False)
dataframe_db= pd.DataFrame(index= range(0,len(dataframe_latest)), columns= dataframe_latest.columns)
dataframe_notdb= pd.DataFrame(index= range(0,len(dataframe_latest)), columns= ['Execution Unit ID', 'Block ID', 'Symbol Datatype', 'Block Structure', 'Input Datatype', 'Cluster Center'])
count1= count2= 0
for index in range(0,len(dataframe_latest)):
    if dataframe_latest['Data Operation Type'].iloc[index]=='NIL':
        dataframe_notdb['Execution Unit ID'].iloc[count1]= dataframe_latest['Execution Unit ID'].iloc[index]
        dataframe_notdb['Block ID'].iloc[count1]= dataframe_latest['Block ID'].iloc[index]
        dataframe_notdb['Symbol Datatype'].iloc[count1]= dataframe_latest['Symbol Datatype'].iloc[index]
        dataframe_notdb['Block Structure'].iloc[count1]= dataframe_latest['Block Structure'].iloc[index]
        dataframe_notdb['Input Datatype'].iloc[count1]= dataframe_latest['Input Datatype'].iloc[index]
        count1+=1
    else:
        dataframe_db.iloc[index]= dataframe_latest.iloc[index]

dataframe_db= dataframe_db.dropna(how='all')
dataframe_notdb= dataframe_notdb.dropna(how='all')


#For non Db items, combining all Execution Blocks into a Execution Unit
dataframe_notdb= dataframe_notdb.sort_values('Execution Unit ID')
dataframe_notdb = dataframe_notdb.reset_index(drop=True)
dataframe_notdb_new= pd.DataFrame(index= range(0,len(dataframe_notdb)), columns= dataframe_notdb.columns)
block=''
count=0
for index in range(0,len(dataframe_notdb)-1):
    if dataframe_notdb['Execution Unit ID'].iloc[index]== dataframe_notdb['Execution Unit ID'].iloc[index+1]:
        block+= str(dataframe_notdb['Block Structure'].iloc[index])

    else:
         block+= str(dataframe_notdb['Block Structure'].iloc[index])  
         dataframe_notdb_new.iloc[count]= dataframe_notdb.iloc[index]
         dataframe_notdb_new['Block Structure'].iloc[count]= block
         count+=1
         block=''




#KMeans Clustering (DB)
#df= copy.deepcopy(dataframe_db)
#df = pd.get_dummies(df, columns=['Symbol Datatype', 'Data Operation Type', 'Data Source Type', 'Data Source Infrastructure', 'Data Source System', 'Data Source Name', 'Data Field Type', 'Data Field Name','Block Structure', 'Input Datatype'])

#pca = PCA(n_components=2)
#result = pca.fit_transform(dataframe_latest)
#df = StandardScaler().fit_transform(df) 
df= db2.values
model = KMeans(n_clusters=3, random_state=0).fit(df)
dataframe_db['Cluster Center']=np.nan

for index in range(0,len(dataframe_db)):
    print(index)
    cluster_center = model.predict(df[index].reshape(1,-1))
    dataframe_db['Cluster Center'].loc[index]= int(cluster_center)




df= notdb2.values
print("before")
model = KMeans(n_clusters=200, random_state=0).fit(df)
dataframe_notdb_new = dataframe_notdb_new.dropna(how= 'any')
print("after")

for index in range(0,len(notdb2)):
    print(index)
    cluster_center = model.predict(df[index].reshape(1,-1))
    dataframe_notdb_new['Cluster Center'].loc[index]= int(cluster_center)

export_csv = dataframe_notdb_new.to_csv (r'D:\Datasets\ExecutionUnits_withoutDBOperations.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
export_csv = dataframe_db.to_csv (r'D:\Datasets\ExecutionUnits_withDBOperations.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

