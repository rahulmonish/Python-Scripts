import pandas as pd
import gensim
from sklearn.preprocessing import LabelEncoder
import collections



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