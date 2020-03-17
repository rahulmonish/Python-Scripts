import pandas as pd
import math
import collections

dataframe= pd.read_csv(r'C:\Users\u65988\Documents\expression_clustter.csv')
dataframe= dataframe[['Data_Operation','Execution_Block_Name','Execution_Unit_ID']]
dataframe= dataframe.dropna(how='all')
dataframe= dataframe.reset_index(drop=True)
dataframe['Data_Operation'] = dataframe['Data_Operation'].fillna('NIL')
structure_list=[]
structure=''
structure_dict=collections.OrderedDict()

for index in range(0,len(dataframe)-1):
    if math.isnan(dataframe['Execution_Unit_ID'].iloc[index]):
        dataframe['Execution_Unit_ID'].iloc[index]= dataframe['Execution_Unit_ID'].iloc[index-1]



for index in range(0,len(dataframe)-1):
    if dataframe['Execution_Unit_ID'].iloc[index]== dataframe['Execution_Unit_ID'].iloc[index+1]:
        if  dataframe['Data_Operation'].iloc[index]!='NIL':
            structure += " " + ' DO '
        else:
            structure += " " + dataframe['Execution_Block_Name'].iloc[index]
    else:
        if  dataframe['Data_Operation'].iloc[index]!='NIL':
            structure += " " + ' DO '
        else:
            structure += " " + dataframe['Execution_Block_Name'].iloc[index]
        
        structure_list.append(structure)
        structure_dict[dataframe['Execution_Unit_ID'].iloc[index]]= structure
        structure=''

    