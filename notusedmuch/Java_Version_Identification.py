# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:55:08 2020

@author: U65988
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing 

df= pd.read_csv(r'D:\Deepminer\SourceCode\deepminer\dm-server\scripts\a.csv')


# =============================================================================
# Symbol_Datatypes_Java8= ['java.time.LocalDate',
# 'java.time.LocalTime',
# 'java.time.LocalDateTime', 
# 'java.time.MonthDay', 
# 'java.time.OffsetTime', 
# 'java.time.OffsetDateTime',
# 'java.time.Clock' ,
# 'java.time.ZonedDateTime', 
# 'java.time.ZoneId' ,
# 'java.time.ZoneOffset', 
# 'java.time.Year' ,
# 'java.time.YearMonth', 
# 'java.time.Period' ,
# 'java.time.Duration', 
# 'java.time.Instant' ,
# 'java.time.DayOfWeek', 
# 'java.time.Month',
# 'java.util.Date' ,
# 'java.sql.Date' ,
# 'java.util.Calendar' ,
# 'java.util.GregorianCalendar' ,
# 'java.util.TimeZone' ,
# 'java.sql.Time' ,
# 'java.sql.Timestamp',
# 'java.text.DateFormat' ,
# 'java.text.SimpleDateFormat'  ]
# 
# q= [i for i in list(df['Symbol Datatype']) if i in Symbol_Datatypes_Java8]
# 
# Lambda_Functions=[]
# Stream= []
# for i in range(0, len(df)):
#     if df['Expression Type Name'].at[i] == 'EXTERNAL_INVOKE':
#         if  df['Expression Type Name'].at[i] in list(df['Execution Unit Name']):
#             if  df['Expression Type Name'].at[i]!=  np.nan:
#                 Lambda_Functions.append(i)
#         
#         if df['Expression Name'].at[i]=='stream':
#             Stream.append(i)
#             
# =============================================================================


# =============================================================================
# q= [i for i in list(df['Symbol Datatype']) if i== 'java.lang.instrument.Instrumentation']
# 
# 
# w= True if 'enum' in list(df['Compilation Unit Name']) else False
# =============================================================================


#Data Preprocessing
df2= pd.DataFrame(index= range(0,len(df)), columns=['Module ID','Imports'])
df2= df2.fillna('')
df= df.fillna('')
import_list=[]

count=0
for i in range(0, len(df)-1):
    if df['Module ID'].at[i]== df['Module ID'].at[i+1]:
        
        if df['Imports'].at[i]!= '':
            import_list.append(df['Imports'].at[i])
            

    else:
        df2['Module ID'].at[count]= df['Module ID'].at[i]
        df2['Imports'].at[count]= import_list
        print(import_list)
        count+=1
        import_list=[]
        
df2= df2.replace('', np.nan)
df2= df2.dropna(how='all')
df2= df2.reset_index(drop=True)

for i in range(0,len(df2)):
    
    import_list= df2['Imports'].at[i]
    import_list= list(set(import_list))
    feature_name=''
    for j in import_list:
        print(j)
        if 'java.sql.' in j:
            feature_name+= 'JDBC'
        elif 'javax.sql.Datasource' in j:
            feature_name+= ' + Datasource'
        elif 'net.sf.ehcache.' in j or 'net.spy.memcached.MemcachedClient' in j:
            feature_name+= '+ Cacheing'
    
    if feature_name!= '':
        df2['Imports'].at[i]= feature_name
    

        
df2= df2[['Module ID', 'Imports']]


#Label Encoding
label_encoder = preprocessing.LabelEncoder() 
df2['Imports']= label_encoder.fit_transform(df2['Imports'])
data= df2.values

#Minisom Neural Net Model
print('#Using Minisom for Clustering Purpose')
from minisom import MiniSom   
som = MiniSom(10,10, 2, sigma=1, learning_rate=0.5)
som.train_random(data, 1000)
x= som.win_map(data)

df2['Cluster Center']= np.nan
cluster_centers= {}
count=0
for cnt,xx in enumerate(data):
        w = som.winner(xx)
        if w not in cluster_centers:
            cluster_centers[w]=count
            count+=1
        df2['Cluster Center'].at[cnt]= cluster_centers[w]
        
    

