# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:18:13 2020

@author: U65988
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import copy
from difflib import SequenceMatcher
from sklearn.cluster import KMeans
import json

similarity_count = 0


def similarity(df_list, acc):
    copy_list = copy.deepcopy(df_list)
    added_index = []
    global similarity_count
    for i in range(0, len(df_list)):
        if i not in added_index:
            
            index_list = []
            index_list.append(i)
            for j in range(i + 1, len(df_list)):
                
                if j not in added_index:
                    ratio = SequenceMatcher(None, str(df_list[i]), str(df_list[j])).ratio()
                    ratio *= 100
                    
                    if ratio >= acc:
                        index_list.append(j)
                        added_index.append(j)
            
            for index in index_list:
                copy_list[index] = similarity_count
        
        similarity_count += 1
    
    return copy_list


def cluster(json_location, csv_location):
    df = pd.read_csv(csv_location)
    with open(json_location, 'r') as f:
        input_dict = json.load(f)
    column_list = []
    
    for i in input_dict.keys():
        column_list.extend(list(input_dict[i].keys()))
    
    column_list = list(set(column_list))
    
    # Data Preprocessing
    df2 = pd.DataFrame(index=range(0, len(df)), columns=['Module ID'] + column_list)
    df2 = df2.fillna('')
    df = df.fillna('')
    module_column_dict = {}
    
    # Combined all the columns from the input dict keys into a dataframe
    for i in column_list:
        module_column_dict[i] = []
    
    count = 0
    for i in range(0, len(df) - 1):
        if df['Module ID'].at[i] == df['Module ID'].at[i + 1]:
            for j in column_list:
                if df[j].at[i] != '':
                    module_column_dict[j].append(str(df[j].at[i]))
        
        else:
            df2['Module ID'].at[count] = df['Module ID'].at[i]
            for j in column_list:
                df2[j].at[count] = module_column_dict[j]
            for i in column_list:
                module_column_dict[i] = []
            count += 1
    
    df2 = df2.replace('', np.nan)
    df2 = df2.dropna(how='all')
    
    # Assigns corresponding version value if the attribute is present
    index_list = []
    for i in range(0, len(df2)):
        for j in input_dict.keys():
            columns = list(input_dict[j].keys())
            for z in columns:
                value_list = input_dict[j][z]
                actual_list = df2[z].at[i]
                for a in value_list:
                    for b in actual_list:
                        if a in b or a==b:
                            df2[z].at[i] = str(j)
                            index_list.append(i)
    
    
    index_list = list(set(index_list))
    df3 = pd.DataFrame(index=range(0, len(index_list)), columns=df2.columns)
    count = 0
    for i in index_list:
        df3.loc[count] = df2.loc[i]
        count += 1
    
    # Converting all lists into string
    for i in range(0, len(df3)):
        for j in column_list:
            if isinstance(df3[j].at[i], list):
                df3[j].at[i] = list(set(df3[j].at[i]))
                string = ''
                for k in df3[j].at[i]:
                    string += k + ","
                string = string.rstrip(',')
                df3[j].at[i] = string
    
    # Calculating similarity bw strings and filling null values
    for i in range(0, len(df3)):
        for j in column_list:
            df3[j] = similarity(list(df3[j]), 100)
    
    # Using KMeans Algorithm
    data = df3[column_list].values
    if len(column_list) == 1:
        data = data.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=len(df3) - 1, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    pred_y = kmeans.fit_predict(data)
    
    df3['Cluster Center'] = pred_y
    
    #Adding Version
    modules= list(df3['Module ID'])
    module_version_dict={}
    
    for i in range(0,len(df2)):
        if df2['Module ID'].at[i] in modules:
            module_version_dict[df2['Module ID'].at[i]]= df2[column_list[0]].at[i]
    
    df3['Version']= ''
    for i in range(0,len(df3)):
        df3['Version'].at[i]= module_version_dict[df3['Module ID'].at[i]]
    
    return df3


if __name__=='__main__':
    str1= r'D:\Deepminer\SourceCode\deepminer\dm-server\scripts\a.csv'
    file= r'C:\Users\u65988\distros.json'
    d= cluster(file,str1)
    print(d)
    


