# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:55:38 2019

@author: U65988
"""
import pickle
import pandas as pd
import copy
from difflib import SequenceMatcher
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")






def similarity(df_list, acc):
  copy_list = copy.deepcopy(df_list)
  added_index = []
  count = 0
  
  for i in range(0, len(df_list) - 1):

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






def cluster(filepath):

    import numpy as np
    
    pickle_in4 = open(r"caller_old.pickle","rb")
    pr=  pickle.load(pickle_in4)
    
    pickle_in4 = open(r"callee_old.pickle","rb")
    sr=  pickle.load(pickle_in4)
    
    pickle_in4 = open(r"block_old.pickle","rb")
    block=  pickle.load(pickle_in4)
     
    pickle_in4 = open(r"dataframe.pickle","rb")
    dataframe=  pickle.load(pickle_in4)
    
    
    
    
    #Getting all the execution Block Structures
    print('Getting all the execution Block Structures')
    dataframe= pd.read_csv(filepath)
    dataframe= dataframe[['Block ID', 'Expression Type ID']]
    dataframe.drop_duplicates(keep='first', inplace= True)
    dataframe = dataframe.dropna(axis=0, subset=['Block ID'])
    dataframe= dataframe.fillna('')
    dataframe= dataframe.reset_index(drop=True)
    structure_list=[]
    structure=''
    structure_dict={}
    
    for index in range(0,len(dataframe)):
        structure_dict[dataframe['Block ID'].at[index]]=''
    
    for index in range(0,len(dataframe)-1):
        if dataframe['Block ID'].at[index]== dataframe['Block ID'].at[index+1]:
            structure += " " + str((dataframe['Expression Type ID'].at[index]))
        else:
            structure += " " + str((dataframe['Expression Type ID'].at[index]))
            structure_list.append(structure)
            structure_dict[dataframe['Block ID'].at[index]]= structure
            structure=''
    
    print("There are ", len(structure_list)," no of Execution Blocks")
    
    
    
    
    
    
    
    
    
    
    
    # Get the symbol entries and Input Datatypes of all Compilation Units
    print('Get the symbol entries and Input Datatypes of all Compilation Units')
    dataframe= pd.read_csv(filepath)
    dataframe= dataframe.replace(np.nan, "")
    indexes = dataframe.index[dataframe['Execution Unit ID'] == ""].tolist()
    
    CU_symbol_dict={}
    CU_input_dict={}
    
    symbol_data=''
    input_data=''
    
    for index in range(0,len(dataframe)):
        
        CU_symbol_dict[index]= ''
        CU_symbol_dict[index]= ''
    
    for index in range(0, len(indexes)-1):
    
        if dataframe['Compilation Unit ID'].at[indexes[index]]== dataframe['Compilation Unit ID'].at[indexes[index+1]]:
            
            if dataframe['Symbol Type Name'].at[indexes[index]]== "SYMBOL_ENTRY":
                symbol_data+= dataframe['Symbol Datatype'].at[indexes[index]] + ", "
            else:
                input_data+= dataframe['Symbol Datatype'].at[indexes[index]] + ", "
            
        else:
            if dataframe['Symbol Type Name'].at[indexes[index]]== "SYMBOL_ENTRY":
                symbol_data+= dataframe['Symbol Datatype'].at[indexes[index]] + ", "
            else:
                input_data+= dataframe['Symbol Datatype'].at[indexes[index]] + ", "
                
            CU_symbol_dict[dataframe['Compilation Unit ID'].at[indexes[index]]]= symbol_data
            CU_input_dict[dataframe['Compilation Unit ID'].at[indexes[index]]]= input_data
            symbol_data=''
            input_data=''
    
    
    
    
    
    
    
    
    
    
    
    #Generating the flow information from Caller and Callee information
    print('Generating the flow information from Caller and Callee information')
    flow_list=[]
    flow_dict={}
    count=0
    i=0
    while i < len(pr):
        if pr[i]==sr[i]:
            #print(i)
            pr.pop(i)
            sr.pop(i)
            block.pop(i)
        i+=1
    
    
    
    for i in range(0,len(pr)):
        #print(i)
        flow_list.append(pr[i])
        flow_list.append(sr[i]) 
        p= pr[i]
        s= sr[i]
        
        while p in sr:
            
            index= pr.index(p)
            index2= sr.index(p)
            
            p = pr[sr.index(p)]
            flow_list.insert(0, p)
            
            if pr[index]==sr[index2]:
                break;
        
        
        while s in pr:
            
            
            index= sr.index(s)
            index2= pr.index(s)
            
            s = sr[pr.index(s)]
            flow_list.append(s)
            
            if sr[index]==pr[index2]:
                break;
            
        
        flow_dict[count]= flow_list
        flow_list=[]
        count+=1
          
        
    
    
    for x in pr:
        index=1
        for i in sr:
            if x==i:
                
                index+=1
        #print(x, "has ",index, " occurences")
    
    
    
    
    
    
    
    # Mapping all execution units to their corresponding blocks
    print('Mapping all execution units to their corresponding blocks')    
    
    import numpy as np
    df= pd.read_csv(filepath)
    df= df[['Execution Unit ID', 'Block ID']]
    df= df.dropna(subset=['Execution Unit ID'])
    df= df.replace(np.nan, "")
    df= df.sort_values(['Execution Unit ID','Block ID'], ascending= [True, True])
    df= df.reset_index(drop= True)
    unit_block_map= {}
    block_list=[]
    
    for i in range(0,len(df)-1):
        unit_block_map[df['Execution Unit ID'].at[i]]= []
        
    for i in range(0,len(df)-1):
        if df['Execution Unit ID'].at[i]== df['Execution Unit ID'].at[i+1]:
            if df['Block ID'].at[i]!= "":
                block_list.append(df['Block ID'].at[i])
        
        else:
            if df['Block ID'].at[i]!= "":
                block_list.append(df['Block ID'].at[i])
            block_list= list(dict.fromkeys(block_list).keys())
            unit_block_map[df['Execution Unit ID'].at[i]]= block_list
            block_list=[]
    
    
    
    
    
    
    
    
    
    
                        
                        
                        
                        
    #From the unit flow information fetching the block flow information                 
    print('From the unit flow information fetching the block flow information')
    final_block_list=[]
    new_block_list= {}
    count=0
    for unit_list in flow_dict.values():
        
        
        
        for index in range(1,len(unit_list)):
            
            
            
            i=0
            
            while i< len(pr):
                
                if pr[i]== unit_list[index-1] and sr[i]== unit_list[index]:
                    block_id= block[i]
                    #print(block_id)
                    break;
                i+=1
            
            
            if index+1 > len(unit_list):
                
                element= unit_block_map[index]
                
                
            
            elif index==1:
                
                 element= unit_block_map[unit_list[index-1]]
                 next_element= unit_block_map[unit_list[index]] #hey
                 
                 position= element.index(block_id) #hey
                 final_block_list= element[:position+1] + next_element + element[position+1:]
                 
            else:
                
                element= final_block_list
                next_element= unit_block_map[unit_list[index]] #no
                
                position= element.index(block_id) #no
                final_block_list= element[:position+1] + next_element + element[position+1:]
        
        new_block_list[count]= final_block_list
        count+=1
                
    
    
            
                
    #Mapping Block ID and Data Operations
    print('Mapping Block ID and Data Operations')
    df= pd.read_csv(filepath)
    df= df.replace(np.nan, '')
    
    block_data_map= {}
    data_source_name={}
    Data_Operation_Type=''
    Data_Source_Name= ''
    
    for index in range(0,len(df)-1):
        
        if df['Block ID'].at[index]== df['Block ID'].at[index+1]:
            if df['Data Operation Type'].at[index]== 'READ' or df['Data Operation Type'].at[index]== 'WRITE':
                Data_Operation_Type += str(df['Data Operation Type'].at[index]) + ", "
                Data_Source_Name += str(df['Data Source Name'].at[index]) + ", "
                
        
        else:
            if df['Data Operation Type'].at[index]== 'READ' or df['Data Operation Type'].at[index]== 'WRITE':
                Data_Operation_Type += str(df['Data Operation Type'].at[index]) + ", "
                Data_Source_Name += str(df['Data Source Name'].at[index]) + ", "
            block_data_map[df['Block ID'].at[index]]= Data_Operation_Type
            data_source_name[df['Block ID'].at[index]]= Data_Source_Name
            Data_Operation_Type=''
            Data_Source_Name=''
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #Getting all data operations sequentially
    print('Getting all data operations sequentially')
    df= pd.read_csv(filepath)
    df2= pd.DataFrame(index= range(0, len(new_block_list)), columns= ['Compilation Unit ID', 'Data Operation Type', 'Block Structure', 'Data Source Name', 'Symbol Entry'])
    structure=''
    
    item_list= list(df['Block ID'])
    count=0
    for item in new_block_list.values():
        
        data=''
        data_source=''
        if item[0] in item_list:
        
            index= item_list.index(item[0])
        
        df2['Compilation Unit ID'].at[count]= df['Compilation Unit ID'].at[index]
        
        
        for i in item:
            
            data+= block_data_map[i]
            structure+= structure_dict[i]
            data_source+= data_source_name[i]
        
        df2['Data Operation Type'].at[count]= data
        df2['Block Structure'].at[count]= structure
        df2['Data Source Name'].at[count]= data_source
        df2['Symbol Entry'].at[count]= CU_symbol_dict[df['Compilation Unit ID'].at[index]]
        count+=1
            
            
    df2['Data Operation Type']= df2['Data Operation Type'].replace('', np.nan)    
    df2= df2.dropna(subset=['Data Operation Type'])
    df2= df2.reset_index(drop= True)
    final_db= copy.deepcopy(df2)
    
    
    
    
    
    
    
    
            
                
    
    
    
    # Vectorizing data for KMeans Clustering
    print('Vecrotizing Data for KMeans Clustering')
    
    label_encoder = preprocessing.LabelEncoder()
    df2['Data Operation Type']= label_encoder.fit_transform((df2['Data Operation Type']))
    df2['Data Source Name']= label_encoder.fit_transform((df2['Data Source Name']))
    df2['Symbol Entry']= label_encoder.fit_transform((df2['Symbol Entry']))
    
    
    
    
    
    
    
    
    
    
    
    #Clustering the Data using SOM(Self Organizing Maps)
    print('Clustering the Data using SOM(Self Organizing Maps)')
    from minisom import MiniSom    
    df3= copy.deepcopy(df2)
    df3= df3[['Data Operation Type', 'Data Source Name', 'Symbol Entry']]
    data= df3.values
    
    #Standard Scaling
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    df3 = pd.DataFrame(data)
    data= df3.values
    
    
    som = MiniSom(10,10, 3, sigma=0.5, learning_rate=0.5)
    som.train_random(data, 1000)
    x= som.win_map(data)
    
    final_db['Cluster Center']= np.nan       
    cluster_centers= {}
    count=0
    for cnt,xx in enumerate(data):
        w = som.winner(xx)
        if w not in cluster_centers.keys():
            cluster_centers[w]=count
            count+=1
        final_db['Cluster Center'].at[cnt]= cluster_centers[w]
        #print(cnt)
       
    
    
    
    
    
    
    
    
    
    
    
    #Capability Mapping
    print('Capability Mapping')
    final_db['Capability']= np.nan
    final_db= final_db.fillna('')
    capability_dict= {'Purchase Order': 13, 'Slot Assignment': 11} #Input Dict for mapping capability
    CU_Cluster_map= {}
    
    
    
    for index in range(0,len(final_db)):
        
        CU_Cluster_map[final_db['Compilation Unit ID'].at[index]]= final_db['Cluster Center'].at[index]
        
    for item in capability_dict.keys():
        
        cluster_center= CU_Cluster_map[capability_dict[item]]
        
        for index in range(0, len(final_db)):
            
            if final_db['Cluster Center'].at[index]== cluster_center:
                
                final_db['Capability'].at[index]= item
    
    
    
    #Converting to dictionary
    final_db= final_db.fillna("")
    new_dict = {}
    for index in range(0, len(final_db)):
      new_dict[str(final_db['Compilation Unit ID'].at[index])] = final_db['Cluster Center'].at[index]
    
    
        
    return new_dict



cluster(r'C:\Users\u65988\cluster_latest.csv')











  
            
    












   