# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:22:34 2019

@author: u65988
"""

import pandas as pd
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


dataframe= pd.read_csv(r'C:\Users\u65988\Desktop\whse.csv')
similarity_ratio= pd.DataFrame(columns=['Block1','Block2','Similarity'])
expression_list = []
expression = ""

for index in range(0,len(dataframe)-1):
    if dataframe['Execution_Block_ID'].iloc[index]==dataframe['Execution_Block_ID'].iloc[index+1]:
        expression += str(dataframe['Name_String'].iloc[index]) + str(dataframe['Name_String'].iloc[index+1])
    else:
        expression_list.append(expression)
        expression=""

count=0
expression_dict={}

for i in expression_list:
    for j in expression_list:
        count+=1
        ratio = SequenceMatcher(None, i, j).ratio()
        similarity_ratio.loc[count, 'Block1'] = i
        similarity_ratio.loc[count, 'Block2'] = j
        similarity_ratio.loc[count, 'Similarity'] = ratio
        print('similarities are',ratio)

