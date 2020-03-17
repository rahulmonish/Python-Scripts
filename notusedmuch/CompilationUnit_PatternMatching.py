# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:44:44 2019

@author: u65988
"""
import pandas as pd
import math
import collections
from difflib import SequenceMatcher
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.cluster import KElbowVisualizer
from fuzzywuzzy import fuzz
import copy 


dataframe= pd.read_csv(r'C:\Users\u65988\Documents\cluster.csv')
dataframe= dataframe[['Compilation Unit ID', 'Execution Unit ID', 'Block Type', 'Data Operation Type', 'Data Source ID', 'Data Source Type', 'Data Source Infrastructure', 'Data Source Port', 'Data Source System', 'Data Source Name', 'Data Field ID', 'Data Field Name']]
dataframe.drop_duplicates(keep='first', inplace= True)
dataframe= dataframe.fillna('NIL')
dataframe = dataframe.dropna(axis=0, subset=['Compilation Unit ID'])
Execution_Units= {}
structure=''
structure_list=[]


for index in range(0,len(dataframe)-1):
    if dataframe['Compilation Unit ID'].iloc[index] == dataframe['Compilation Unit ID'].iloc[index+1]:
        if dataframe['Data Operation Type'].iloc[index]=='NIL':
            structure+= dataframe['Block Type'].iloc[index]
        else:
            structure+=  str(dataframe['Data Operation Type'].iloc[index]) + str(dataframe['Data Source ID'].iloc[index]) + str(dataframe['Data Source Type'].iloc[index]) + str(dataframe['Data Source Infrastructure'].iloc[index]) + str(dataframe['Data Source Port'].iloc[index]) + str(dataframe['Data Source System'].iloc[index]) + str(dataframe['Data Source Name'].iloc[index]) + str(dataframe['Data Field ID'].iloc[index]) + str(dataframe['Data Field Name'].iloc[index])
    else:
        if dataframe['Data Operation Type'].iloc[index]=='NIL':
            structure+= dataframe['Block Type'].iloc[index]
        else:
            structure+=  str(dataframe['Data Operation Type'].iloc[index]) + str(dataframe['Data Source ID'].iloc[index]) + str(dataframe['Data Source Type'].iloc[index]) + str(dataframe['Data Source Infrastructure'].iloc[index]) + str(dataframe['Data Source Port'].iloc[index]) + str(dataframe['Data Source System'].iloc[index]) + str(dataframe['Data Source Name'].iloc[index]) + str(dataframe['Data Field ID'].iloc[index]) + str(dataframe['Data Field Name'].iloc[index])
        structure_list.append(structure.replace('NIL',''))
        Execution_Units[dataframe['Compilation Unit ID'].iloc[index]]= structure.replace('NIL','')
        structure=''



#Using Fuzzy Ratio and finding similar compilation unit structures
structure_list2= copy.deepcopy(structure_list)
index=0
new_dict= {}
array_list=[]
while index < len(structure_list2):
    new_dict[structure_list2[index]]=[]
    index2= index + 1
    while index2 < len(structure_list2):
        ratio = fuzz.ratio(structure_list2[index],structure_list2[index2])
        if(ratio > 70):
            new_dict[structure_list2[index]].append(structure_list2[index2])
            structure_list2.pop(index2)
        index2+=1

    index+=1




#Plotting The Elbow Curve in order to find the optimal K value
x = TfidfVectorizer().fit_transform(structure_list)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(10,20), metric='distortion')
visualizer.fit(x.toarray())        # Fit the data to the visualizer
visualizer.poof() 
print(visualizer.elbow_value_)