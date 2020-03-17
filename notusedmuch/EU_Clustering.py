# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import copy
import matplotlib.pyplot as plt
import math
from difflib import SequenceMatcher
from minisom import MiniSom    
from sklearn import preprocessing








def calc_distance(x1,y1,a,b,c):
    
    d= abs((a*x1 + b*y1 +c)) / (math.sqrt(a*a  + b*b))
    return d


def similarity(df_list, acc):
  copy_list = copy.deepcopy(df_list)
  added_index = []
  count = 0
  
  for i in range(0, len(df_list)):

    if i not in added_index:
      index_list = []
      index_list.append(i)
      for j in range(i + 1, len(df_list)):

        ratio= SequenceMatcher(None, df_list[i],  df_list[j]).ratio()
        ratio*=100
        
        if ratio > acc:
          index_list.append(j)
          added_index.append(j)

      

      for index in index_list:
        copy_list[index] = count

    count += 1

  return copy_list


def get_optimum_k(df):
    
    data= df.values
    max_length= len(df)-1
    dist_points_from_cluster_center=[]
    K = range(1,max_length)
    for no_of_clusters in K:
        k_model= KMeans(n_clusters= no_of_clusters)
        k_model.fit(data)
        dist_points_from_cluster_center.append(k_model.inertia_)
    
    plt.plot(K, dist_points_from_cluster_center)
    plt.plot([K[0], K[max_length-2]], [dist_points_from_cluster_center[0], dist_points_from_cluster_center[max_length-2]], 'ro-' )
    
    a= dist_points_from_cluster_center[0]- dist_points_from_cluster_center[max_length-2]
    b= K[max_length-2] - K[0]
    c1= K[0] * dist_points_from_cluster_center[max_length-2]
    c2= K[max_length-2] * dist_points_from_cluster_center[0]
    c= c1-c2
    
    distance_of_points_from_line = []
    for k in range(max_length-1):
        distance_of_points_from_line.append(calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c ))
        
    plt.plot(K, distance_of_points_from_line)
    
    print("Optimum Value of K is ", int(distance_of_points_from_line.index(max(distance_of_points_from_line))+1))
    k = int(distance_of_points_from_line.index(max(distance_of_points_from_line))+1)
    
    return k





if __name__=="__main__":
    filepath= r'C:\Users\u65988\cluster_new.csv'
    dataframe= pd.read_csv(filepath)
    dataframe= dataframe[['Block ID', 'Expression Type ID']]
    #dataframe.drop_duplicates(keep='first', inplace= True)
    dataframe = dataframe.dropna(axis=0, subset=['Block ID','Expression Type ID'])
    dataframe= dataframe.fillna('NIL')
    structure_list=[]
    structure=''
    structure_dict={}
    dataframe= dataframe.reset_index(drop=True)
    
    
    #Getting all the execution Block Structures
    print('#Getting all the execution Block Structures')
    for index in range(0,len(dataframe)-1):
        if dataframe['Block ID'].at[index]== dataframe['Block ID'].at[index+1]:
            structure += " " + str(int(dataframe['Expression Type ID'].at[index]))
        else:
            structure += " " + str(int(dataframe['Expression Type ID'].at[index]))
            structure_list.append(structure)
            structure_dict[dataframe['Block ID'].at[index]]= structure
            structure=''
    
    print("There are ", len(structure_list)," no of Execution Blocks")
    
    
    
    
    
    
    
    
    #Creating Compilation Unit Structures
    print('#Creating Compilation Unit Structures')
    dataframe_new= pd.read_csv(filepath)
    dataframe_new= dataframe_new[['Compilation Unit ID', 'Execution Unit ID', 'Block ID', 'Symbol Type Name', 'Symbol Datatype']]
    dataframe_new= dataframe_new.fillna('NIL')
    dataframe_new= dataframe_new.reset_index(drop= True)
    compilation_unit_list = dataframe_new
    compilation_unit_list= compilation_unit_list.fillna(0)
    
    count=0
    
    for index in range(0, len(dataframe_new)):
        if dataframe_new['Execution Unit ID'].at[index]== 'NIL':
            compilation_unit_list.iloc[count]= dataframe_new.iloc[index]
            count+=1
    compilation_unit_list= compilation_unit_list[:count]
    
    CU_symbol_dict={}
    CU_input_dict= {}
    input_data=[]
    symbol_entry=[]
    count=0
    for index in range(0,len(compilation_unit_list)-1):
        
        if compilation_unit_list['Compilation Unit ID'].at[index]== compilation_unit_list['Compilation Unit ID'].at[index+1]:
            if compilation_unit_list['Symbol Type Name'].at[index]=='INPUT_DATA':
                input_data.append(str(compilation_unit_list['Symbol Datatype'].at[index]))
            
            elif compilation_unit_list['Symbol Type Name'].at[index]=='SYMBOL_ENTRY':
                symbol_entry.append(str(compilation_unit_list['Symbol Datatype'].at[index]))
        
        else:
            if compilation_unit_list['Symbol Type Name'].at[index]=='INPUT_DATA':
                input_data.append(str(compilation_unit_list['Symbol Datatype'].at[index]))
            
            elif compilation_unit_list['Symbol Type Name'].at[index]=='SYMBOL_ENTRY':
                symbol_entry.append(str(compilation_unit_list['Symbol Datatype'].at[index]))
            
            input_data= list(set(input_data))
            symbol_entry= list(set(symbol_entry))
            input_data_string= ''.join(map(str, input_data))
            symbol_entry_string= ''.join(map(str, symbol_entry))
            CU_symbol_dict[compilation_unit_list['Compilation Unit ID'].at[index]]= symbol_entry_string
            CU_input_dict[compilation_unit_list['Compilation Unit ID'].at[index]]= input_data_string
            count+=1
            input_data=[]
            symbol_entry= []
    
    
    
    
    
    #Creating Execution Unit Structure
    print('#Creating Execution Unit Structure')
    dataframe_new= pd.read_csv(filepath)
    dataframe_new= dataframe_new[['Execution Unit ID', 'Block ID', 'Symbol Type Name', 'Symbol Datatype', 'Data Source Name','Data Operation Type']]
    dataframe_new = dataframe_new.dropna(axis=0, subset=['Execution Unit ID'])
    dataframe_new= dataframe_new.sort_values(by=['Execution Unit ID'])
    dataframe_new= dataframe_new.fillna('NIL')
    dataframe_new= dataframe_new.reset_index(drop=True)
    dataframe_new= dataframe_new.reset_index(drop= True)
    
    
    dataframe_final = pd.DataFrame(index= range(0,len(dataframe_new)), columns= ['Execution Unit ID', 'Symbol Entry Datatype', 'Input Datatype', 'Data Operation Type', 'Data Source Name'])
    input_data=[]
    symbol_entry=[]
    data_operation_list=[]
    data_source_name_list= []
    data_operation_string=''
    data_source_name_string=''
    count=0
    for index in range(0,len(dataframe_new)-1):
        
        if dataframe_new['Execution Unit ID'].at[index]== dataframe_new['Execution Unit ID'].at[index+1]:
            if dataframe_new['Symbol Type Name'].at[index]=='INPUT_DATA':
                input_data.append(str(dataframe_new['Symbol Datatype'].at[index]))
            
            elif dataframe_new['Symbol Type Name'].at[index]=='SYMBOL_ENTRY':
                symbol_entry.append(str(dataframe_new['Symbol Datatype'].at[index]))
            
            if dataframe_new['Data Operation Type'].at[index]!= "NIL":
                print(dataframe_new['Data Operation Type'].at[index])
                data_operation_list.append(dataframe_new['Data Operation Type'].at[index])
                data_source_name_list.append(dataframe_new['Data Source Name'].at[index])
        
        else:
            data_operation_list= list(set(data_operation_list))
            data_source_name_list= list(set(data_source_name_list))
            data_operation_string= '-'.join(data_operation_list)
            data_source_name_string= '-'.join(data_source_name_list)
            input_data= list(set(input_data))
            symbol_entry= list(set(symbol_entry))
            input_data_string= '-'.join(map(str, input_data))
            symbol_entry_string= '-'.join(map(str, symbol_entry))
            dataframe_final['Execution Unit ID'].at[count]= dataframe_new['Execution Unit ID'].at[index]
            dataframe_final['Symbol Entry Datatype'].at[count]= symbol_entry_string
            dataframe_final['Input Datatype'].at[count]= input_data_string
            dataframe_final['Data Source Name'].at[count]= data_source_name_string
            dataframe_final['Data Operation Type'].at[count]= data_operation_string
            data_operation_string=''
            data_source_name_string=''
            count+=1
            data_operation_list=[]
            data_source_name_list=[]
            input_data= []
            symbol_entry= []
        
        
                
# =============================================================================
#     #Merging with Execution Block structure
#     print('#Merging with Execution Block structure')
#     dataframe= pd.read_csv(filepath)
#     dataframe = dataframe.dropna(axis=0, subset=['Block ID'])
#     dataframe= dataframe[['Execution Unit ID', 'Symbol Datatype', 'Data Operation Type', 'Data Source Type', 'Data Source Infrastructure', 'Data Source System', 'Data Source Name', 'Data Field Type', 'Data Field Name']]
#     dataframe= dataframe.fillna('NIL')
#     dataframe_latest= pd.DataFrame(index= range(0,len(dataframe)), columns= ['Execution Unit ID', 'Symbol Datatype', 'Data Operation Type', 'Data Source Type', 'Data Source Infrastructure', 'Data Source System', 'Data Source Name', 'Data Field Type', 'Data Field Name'])
#     data_operation=''
#     dataframe= dataframe.sort_values(by=['Execution Unit ID'])
#     dataframe=dataframe.reset_index(drop=True)
#     count=0
#     
# 
#     for i in range(0,len(dataframe)):
#         if dataframe['Execution Unit ID'].at[i]== dataframe['Execution Unit ID'].at[i+1]:
#             
#         
#         else:
# =============================================================================
            
            
                
    
# =============================================================================
#     for index in range(0,len(dataframe)-1):
#         if dataframe['Execution Unit ID'].at[index]== dataframe['Execution Unit ID'].at[index+1]:
#             if dataframe['Data Operation Type'].at[index]!= 'NIL':
#                 dataframe_latest.iloc[count]= dataframe.iloc[index]
#                 flag=1
#                 count+=1
#         else:
#              if dataframe['Data Operation Type'].at[index]!= 'NIL':
#                  dataframe_latest.iloc[count]= dataframe.iloc[index]
#                  flag=1
#                  count+=1
#             
#              if flag==0:
#                 dataframe_latest.iloc[count]= dataframe.iloc[index]
#                 count+=1
#              else:
#                 flag=0
# =============================================================================
    
    
    #consolidating all the data into a single dataframe
# =============================================================================
#     print('#consolidating all the data into a single dataframe')
#     dataframe_latest['Block Structure']= np.nan
#     dataframe_latest['Input Datatype']= np.nan
#     dataframe_final= dataframe_final.dropna(how= 'all')
#     dataframe_latest= dataframe_latest.dropna(how= 'all')
#     dataframe_latest= dataframe_latest.fillna('NIL')
#     for index in range(0,len(dataframe_latest)):
#         
#         unit_id= dataframe_latest['Execution Unit ID'].at[index]
#         row = dataframe_final[dataframe_final['Execution Unit ID']==unit_id].index.values.astype(int)[0]
#         dataframe_latest['Input Datatype'].at[index]= dataframe_final['Input Datatype'][row]
#         dataframe_latest['Symbol Datatype'].at[index]= dataframe_final['Symbol Entry Datatype'][row]
#         if int(dataframe['Block ID'].at[index]) in structure_dict:
#             dataframe_latest['Block Structure'].at[index]= structure_dict[int(dataframe['Block ID'].at[index])]
# =============================================================================
    
    
    
    
    #Splitting the execution Units with DB and without DB
    print('#Splitting the execution Units with DB and without DB')
    dataframe_latest= dataframe_final
    dataframe_latest= dataframe_latest.sort_values('Data Operation Type', ascending= False)
    dataframe_latest= dataframe_latest.reset_index(drop=True)
    dataframe_db= pd.DataFrame(index= range(0,len(dataframe_latest)), columns= dataframe_latest.columns)
    dataframe_notdb= pd.DataFrame(index= range(0,len(dataframe_latest)), columns= ['Execution Unit ID', 'Symbol Entry Datatype', 'Input Datatype', 'Cluster Center'])
    count1= 0
    count2=0
    for index in range(0,len(dataframe_latest)):
        if dataframe_latest['Data Operation Type'].at[index]=='':
            dataframe_notdb['Execution Unit ID'].at[count1]= dataframe_latest['Execution Unit ID'].at[index]
            #dataframe_notdb['Block ID'].at[count1]= dataframe_latest['Block ID'].at[index]
            dataframe_notdb['Symbol Entry Datatype'].at[count1]= dataframe_latest['Symbol Entry Datatype'].at[index]
            #dataframe_notdb['Block Structure'].at[count1]= dataframe_latest['Block Structure'].at[index]
            dataframe_notdb['Input Datatype'].at[count1]= dataframe_latest['Input Datatype'].at[index]
            count1+=1
        else:
            dataframe_db['Execution Unit ID'].at[count2]= dataframe_latest['Execution Unit ID'].at[index]
            #dataframe_db['Block ID'].at[count2]= dataframe_latest['Block ID'].at[index]
            dataframe_db['Symbol Entry Datatype'].at[count2]= dataframe_latest['Symbol Entry Datatype'].at[index]
            #dataframe_db['Block Structure'].at[count2]= dataframe_latest['Block Structure'].at[index]
            dataframe_db['Input Datatype'].at[count2]= dataframe_latest['Input Datatype'].at[index]
            dataframe_db['Data Source Name'].at[count2]= dataframe_latest['Data Source Name'].at[index]
            dataframe_db['Data Operation Type'].at[count2]= dataframe_latest['Data Operation Type'].at[index]
            count2+=1
    
    dataframe_db= dataframe_db[['Execution Unit ID', 'Symbol Entry Datatype',  'Input Datatype', 'Data Source Name', 'Data Operation Type']]
    dataframe_db= dataframe_db.dropna(how='all')
    dataframe_notdb= dataframe_notdb.dropna(how='all')
    
    
    
    
    #For non Db items, combining all Execution Blocks into a Execution Unit
# =============================================================================
#     print('#For non Db items, combining all Execution Blocks into a Execution Unit')
#     dataframe_notdb= dataframe_notdb.sort_values('Execution Unit ID')
#     dataframe_notdb = dataframe_notdb.reset_index(drop=True)
#     dataframe_notdb_new= pd.DataFrame(index= range(0,len(dataframe_notdb)), columns= dataframe_notdb.columns)
#     block=''
#     count=0
#     for index in range(0,len(dataframe_notdb)-1):
#         if dataframe_notdb['Execution Unit ID'].at[index]== dataframe_notdb['Execution Unit ID'].at[index+1]:
#             block+= str(dataframe_notdb['Block Structure'].at[index])
#     
#         else:
#              block+= str(dataframe_notdb['Block Structure'].at[index])  
#              dataframe_notdb_new.iloc[count]= dataframe_notdb.iloc[index]
#              dataframe_notdb_new['Block Structure'].at[count]= block
#              count+=1
#              block=''
# =============================================================================
    
    
    
    
    
    #Vectorizing the Data for Clustering
    print('#Vectorizing the Data for Clustering')
    notdb= dataframe_notdb[['Symbol Entry Datatype', 'Input Datatype']]
    notdb = notdb.dropna(how='any')
    notdb.reset_index(drop=True)
    db= dataframe_db[['Symbol Entry Datatype', 'Data Operation Type',  'Data Source Name', 'Input Datatype']]
    db2= pd.DataFrame(index= range(0,len(dataframe_db)), columns=[['Symbol Entry Datatype', 'Data Operation Type', 'Data Source Name', 'Input Datatype']])
    notdb2= pd.DataFrame(index= range(0,len(notdb)), columns=[['Symbol Entry Datatype', 'Input Datatype',]])
    
    
    
    db= db.fillna('')
    label_encoder = preprocessing.LabelEncoder() 
    #db2[['Symbol Datatype']]= label_encoder.fit_transform((list(db['Symbol Datatype'])))
    db2[['Data Operation Type']]= label_encoder.fit_transform((list(db['Data Operation Type'])))
    db2[['Data Source Name']]= label_encoder.fit_transform((list(db['Data Source Name'])))
    #db2[['Block Structure']]= label_encoder.fit_transform(list(db['Block Structure']))
    #db2[['Input Datatype']]= label_encoder.fit_transform(list(db['Input Datatype']))
    notdb2[['Input Datatype']]= label_encoder.fit_transform(list(notdb['Input Datatype']))
    notdb2[['Symbol Entry Datatype']]= label_encoder.fit_transform(list(notdb['Symbol Entry Datatype']))
    #notdb2[['Block Structure']]= label_encoder.fit_transform(list(notdb['Block Structure']))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #Clustering Execution Units (DB) using SOM
    db2=db2[['Data Operation Type', 'Data Source Name']]
    print('#Clustering Execution Units (DB) using SOM')
    data= db2.values
    
    #Standard Scaling
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    db2 = pd.DataFrame(data)
    
    data= db2.values
    data[:, 1] *= 10
    som = MiniSom(3,3, 2, sigma=0.5, learning_rate=0.5)
    som.train_random(data, 1000)
    x= som.win_map(data)
    dataframe_db['Cluster Center']= np.nan     
    dataframe_db= dataframe_db.reset_index(drop=True)  
    cluster_centers= {}
    count=0
    for cnt,xx in enumerate(data):
        w = som.winner(xx)
        if w not in cluster_centers.keys():
            cluster_centers[w]=count
            count+=1
        dataframe_db['Cluster Center'].at[cnt]= cluster_centers[w]
    
    
    
    
    #Clustering Execution Units (Not DB) using SOM
    print('#Clustering Execution Units (Not DB) using SOM')
    
    data= notdb2.values
        
    som = MiniSom(10,10, 2, sigma=0.5, learning_rate=0.5)
    som.train_random(data, 1000)
    x= som.win_map(data)
    dataframe_notdb['Cluster Center']= np.nan       
    cluster_centers= {}
    count=0
    for cnt,xx in enumerate(data):
        w = som.winner(xx)
        if w not in cluster_centers.keys():
            cluster_centers[w]=count
            count+=1
        dataframe_notdb['Cluster Center'].at[cnt]= cluster_centers[w]
    
    
    
    
    
    
    
    
    
    #Capability Mapping
# =============================================================================
#     print('Capability Mapping')
#     dataframe_notdb['Capability']= np.nan
#     dataframe_notdb= dataframe_notdb.fillna(' ')
#     capability_dict= {'Purchase Order': 187, 'Slot Assignment': 924} #Input Dict for mapping capability
#     EU_Cluster_map= {}
#     for index in range(0,len(dataframe_notdb_new)):
#         
#         EU_Cluster_map[dataframe_notdb_new['Execution Unit ID'].at[index]]= dataframe_notdb_new['Cluster Center'].at[index]
#         
#     for item in capability_dict.keys():
#         
#         cluster_center= EU_Cluster_map[capability_dict[item]]
#         
#         for index in range(0, len(dataframe_notdb_new)):
#             
#             if dataframe_notdb_new['Cluster Center'].at[index]== cluster_center:
#                 
#                 dataframe_notdb_new['Capability'].at[index]= item
#     
#     dataframe_notdb_new= dataframe_notdb_new.rename(columns={"Execution Unit ID":"Execution_Unit_ID"})
# 
#     dataframe_notdb_new= dataframe_notdb_new[dataframe_notdb_new.Execution_Unit_ID!= " "]
# =============================================================================
        
    #return dataframe_db, dataframe_notdb_new




#x,y= cluster(r'C:\Users\u65988\cluster_final.csv')

