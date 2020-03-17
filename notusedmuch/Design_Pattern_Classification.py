# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:59:17 2019

@author: U65988
"""

import pandas as pd
import pickle
from grakn.client import GraknClient
caller = []
callee = []

def get_grakn_data():
    block=[]
    count=0
    with GraknClient(uri="172.16.253.250:48555") as client:
        with client.session(keyspace="dm_graph") as session:
            with session.transaction().read() as read_transaction:
                
                results = read_transaction.query("match (callee: $callee, caller: $caller) isa invoke; $callee has db_id $callee_id; $caller has db_id $caller_id; get;")

                for result in results:
                    x= result.map()['callee_id'].value()
                    y= result.map()['caller_id'].value()
                
                    callee.append(y)
                    caller.append(x)
                    print(x,y)
                    
                pickle_out = open("callee.pickle","wb")
                pickle.dump(callee, pickle_out)
                pickle_out.close()
                
                pickle_out = open("caller.pickle","wb")
                pickle.dump(caller, pickle_out)
                pickle_out.close()








get_grakn_data()


pickle_in = open(r"C:\Users\u65988\callee.pickle","rb")
Child_EU=  pickle.load(pickle_in)

pickle_in = open(r"C:\Users\u65988\caller.pickle","rb")
Child_Caller=  pickle.load(pickle_in)





inheritance={}
#For mapping Execution Units to Compilation Units
EU_CU_Map= {}
for i in range(0,len(df2)):
    EU_CU_Map[df2['Execution Unit ID'].at[i]]= df2['Compilation Unit ID'].at[i]




#Removing all imports that are not class names
for i in range(0,len(inheritance)):
    for j in range(0, len(inheritance[i])):
        if j not in list(df2['Compilation Unit Name']):
            inheritance[i].pop(j)





#Populating the main dataframe
df= pd.DataFrame(index=range(0,len(callee)), columns=['Caller','Callee','Callee_CU','Parent_CU'])
df['Caller']= caller
df['Callee']= callee

for i in range(0,len(df)):
    
    Compilation_Unit = EU_CU_Map[df['Callee'].at[i]]
    df['Callee_CU'].at[i]= Compilation_Unit
    
    if Compilation_Unit in inheritance:
        df['Parent_CU']= inheritance[Compilation_Unit]
    
    else:
        df['Parent_CU']=[]
        
        
        
        
    

df= df.sort_values(by=['Caller'])
factory_classes=[]
for i in range(0,len(df)-1):
    if df['Caller'].at[i]== df['Caller'].at[i+1]:
        if df['Parent_CU'].at[i]==df['Parent_CU'].at[i+1]:
            factory_classes.append(EU_CU_Map[df['Caller'].at[i]])

factory_classes= set(list(factory_classes))    

            





















#Builder Pattern
import numpy as np
df2= pd.read_csv(r'C:/Users/u65988/latest_dump.csv')

CU_list= list(set(list(df2['Compilation Unit Name'])))
EU_list= list(set(list(df2['Execution Unit Name'])))
CU_list_new=[]
for i in CU_list:
    split= i.split('.')
    CU_list_new.append(split[-1])

df2= df2.sort_values(by=['Expression Parent'])
df2= df2.reset_index(drop=True)
df2= df2.replace(np.nan, '')
statement_list={}
count=0

for i in range(0,len(df2)):
    statement_list[count]=[]
    count+=1


expression_list={}
expression_parent_list={}
expression_receiver_list={}
for i in range(0,len(df2)):
    if df2['Expression ID'].at[i]!= "":
        expression_list[int(df2['Expression ID'].at[i])]= df2['Expression Name'].at[i]
        expression_parent_list[int(df2['Expression ID'].at[i])]= df2['Expression Parent'].at[i]
        expression_receiver_list[expression_list[int(df2['Expression ID'].at[i])]]=  df2['Expression Index'].at[i]

    
df2= df2.rename(columns={"Expression Index": "Expression_Index"})
df2= df2[df2.Expression_Index == 'receiver']

        
df2= df2.reset_index(drop=True)
count=0



count=0
for i in expression_parent_list.keys():
    statement_list[count].append(i)
    parent= expression_parent_list[i]
    while parent in expression_parent_list:
        
        statement_list[count].append(parent)
        parent= expression_parent_list[parent]

    count+=1
    

    
    
    
    
    
pickle_in = open(r"C:\Users\u65988\callee.pickle","rb")
parent = pickle.load(pickle_in)   

pickle_in = open(r"C:\Users\u65988\caller.pickle","rb")
child = pickle.load(pickle_in)   
  

parent_child={}  
for i in range(0,len(parent)):
    parent_child[child[i]]= parent[i]


expression_unit_map={}
for i in range(0,len(df2)):
    expression_unit_map[df2['Expression ID'].at[i]]= df2['Execution Unit ID'].at[i]
   
for i in statement_list.keys():
    expression= statement_list[i]
    #print(expression)
    flag=0
    flag2=0
    flag3=0
    child_list=[]
    for j in expression:
        #print(j)
        if expression_list[j] in CU_list_new:
            flag+=1
        if expression_list[j] in EU_list and j in expression_unit_map:
            flag2+=1
            child_list.append(expression_unit_map[j]) 
    
    #print(child_list)
    if len(child_list)>2:
        print(child_list)
        
        
    if child_list!=[]:
        
        for i in range(0,len(child_list)-1):
            if child_list[i] not in parent_child:
                flag3=0
                
            elif parent_child[child_list[i]]== parent_child[child_list[i+1]]:
                flag3=1
            else:
                flag3=0
                break
                
        
            
    if flag3==1 :
        for i in expression:
            if expression_list[i]!=[]:
                print(expression_list[i])
    
        


if 21687 in list(df2['Expression ID']):
    print("h")    





        
        
        