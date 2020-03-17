# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:05:50 2019

@author: U65988
"""

import pandas as pd
from sklearn.cluster import KMeans
import copy
import numpy as np
from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math,re
from difflib import SequenceMatcher
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from yellowbrick.cluster import KElbowVisualizer
import sys



capability_dict = {'Purchase Order': 3, 'Slot Assignment': 238}  # Input Dict for mapping capability



def capability_association(dataframe2):
    
    dataframe2['Capability'] = np.nan
    CU_Cluster_map = {}
    
    for index in range(0, len(dataframe2)):
      CU_Cluster_map[dataframe2['Block ID'].iloc[index]] = dataframe2['Cluster Center'].iloc[index]
    
    for item in capability_dict.keys():
    
      cluster_center = CU_Cluster_map[capability_dict[item]]
    
      for index in range(0, len(dataframe2)):
    
        if dataframe2['Cluster Center'].iloc[index] == cluster_center:
          dataframe2['Capability'].iloc[index] = item
    
    return dataframe2





def plot(df):
    pca = PCA(n_components=2)
    df2= pca.fit_transform(df)
    plt.scatter(df2[:,0], df2[:,1])




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








def calc_distance(x1, y1, a, b, c):
  d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
  return d



def Optimum_K_Value(df):
  data = df.values
  max_length = len(df) - 1
  #max_length = 50
  print(max_length)
  dist_points_from_cluster_center = []
  K = range(1, max_length)
  for no_of_clusters in K:
    k_model = KMeans(n_clusters=no_of_clusters)
    k_model.fit(data)
    dist_points_from_cluster_center.append(k_model.inertia_)

  plt.plot(K, dist_points_from_cluster_center)
  plt.plot([K[0], K[max_length - 2]],
           [dist_points_from_cluster_center[0], dist_points_from_cluster_center[max_length - 2]], 'ro-')

  a = dist_points_from_cluster_center[0] - dist_points_from_cluster_center[max_length - 2]
  b = K[max_length - 2] - K[0]
  c1 = K[0] * dist_points_from_cluster_center[max_length - 2]
  c2 = K[max_length - 2] * dist_points_from_cluster_center[0]
  c = c1 - c2

  distance_of_points_from_line = []
  for k in range(max_length - 1):
    distance_of_points_from_line.append(calc_distance(K[k], dist_points_from_cluster_center[k], a, b, c))

  plt.plot(K, distance_of_points_from_line)

  print("Optimum Value of K is ", int(distance_of_points_from_line.index(max(distance_of_points_from_line)) + 1))
  k = int(distance_of_points_from_line.index(max(distance_of_points_from_line)) + 1)





def cluster(path):
	# Getting all the execution Block Structures
	#path= sys.argv[1]
	#path=r'cluster_latest.csv'
	dataframe = pd.read_csv(path)
	dataframe = dataframe[['Block ID', 'Expression Type ID']]
	# dataframe.drop_duplicates(keep='first', inplace= True)
	dataframe = dataframe.dropna(axis=0, subset=['Block ID'])
	dataframe = dataframe.fillna('')
	structure_list = []
	structure = ''
	structure_dict = {}

	for index in range(0, len(dataframe)):
	  structure_dict[dataframe['Block ID'].iloc[index]] = ''

	for index in range(0, len(dataframe) - 1):
	  if dataframe['Block ID'].iloc[index] == dataframe['Block ID'].iloc[index + 1]:
		if str(dataframe['Expression Type ID'].iloc[index])!= '':
			structure += str(int((dataframe['Expression Type ID'].iloc[index]))) + " "
	  else:
		if str(dataframe['Expression Type ID'].iloc[index])!= '':
			structure += str(int((dataframe['Expression Type ID'].iloc[index]))) + " "
		structure_list.append(structure)
		structure_dict[dataframe['Block ID'].iloc[index]] = structure
		structure = ''

	print("There are ", len(structure_list), " no of Execution Blocks")




	# Mapping Blocks with Data Operations
	dataframe = pd.read_csv(path)
	dataframe = dataframe.sort_values(by=['Block ID'])
	dataframe = dataframe.fillna('')
	data_operation_list = []
	data_source_name_list = []
	Block_Data_Map = {}
	Block_Data_Name_Map = {}
	data_operation= ''
	data_source_name=''

	for index in range(0, len(dataframe) - 1):

	  if dataframe['Block ID'].iloc[index] == dataframe['Block ID'].iloc[index + 1]:

		if dataframe['Data Operation Type'].iloc[index] != '':
		  data_operation_list.append(dataframe['Data Operation Type'].iloc[index])
		  data_source_name_list.append(dataframe['Data Source Name'].iloc[index])

	  else:

		data_operation_list.append(dataframe['Data Operation Type'].iloc[index])
		data_source_name_list.append(dataframe['Data Source Name'].iloc[index])
		
		data_source_name_list.sort()
		data_operation_list.sort()
		
		for i in data_operation_list:
			if i== "":
				data_operation += i 
			else:
				data_operation += i + " "
				
			
		
		for j in data_source_name_list:
			
			if j== "":
				data_source_name += j 
			else:
				data_source_name += j + " "
		
		
		Block_Data_Map[dataframe['Block ID'].iloc[index]] = data_operation
		Block_Data_Name_Map[dataframe['Block ID'].iloc[index]] = data_source_name
		data_operation_list = []
		data_source_name_list = []
		data_operation= ''
		data_source_name=''





	#Mapping Blocks with External Invokes
	dataframe2 = pd.read_csv(path)
	dataframe2 = dataframe2.sort_values(by=['Block ID'])
	dataframe2 = dataframe2.fillna('')
	external_invoke = ''
	external_invoke_list=[]
	external_invoke_dict = {}

	for index in range(0, len(dataframe2) - 1):

	  if dataframe2['Block ID'].iloc[index] == dataframe2['Block ID'].iloc[index + 1]:
		if dataframe2['Expression Type Name'].iloc[index] == 'EXTERNAL_INVOKE':
		  external_invoke_list.append(str(dataframe2['Expression Name'].iloc[index]))

	  else:
		if dataframe2['Expression Type Name'].iloc[index] == 'EXTERNAL_INVOKE':
		  external_invoke_list.append(str(dataframe2['Expression Name'].iloc[index]))
		
		external_invoke_list.sort()
		
		for i in external_invoke_list:
			external_invoke+= i + " "
		
		external_invoke_dict[dataframe2['Block ID'].iloc[index]] = external_invoke
		external_invoke = ''
		external_invoke_list = []





	#Getting the final Dataframe
	dataframe = pd.DataFrame(index=range(0, len(Block_Data_Map)),
							 columns=['Block ID','External Invoke', 'Data Operation', 'Data Source Name', 'Block Structure'])
	Block_List = list(Block_Data_Map.keys())
	for index in range(0, len(Block_Data_Map)):
	  block_id = Block_List[index]
	  dataframe['Block ID'].iloc[index] = block_id
	  dataframe['Data Operation'].iloc[index] = Block_Data_Map[block_id]
	  dataframe['Block Structure'].iloc[index] = structure_dict[block_id]
	  dataframe['Data Source Name'].iloc[index] = Block_Data_Name_Map[block_id]
	  dataframe['External Invoke'].iloc[index] = external_invoke_dict[block_id]




	#Plotting the current Dataset
	#plot(dataframe)



	#Clustering DB items
	dataframe2 = copy.deepcopy(dataframe)
	dataframe2 = dataframe[['Block ID', 'Data Operation', 'Data Source Name']]
	df_db = copy.deepcopy(dataframe2)

	df_db= df_db.replace('', np.nan)
	df_db = df_db.dropna(subset=['Data Operation'])
	df_db= df_db.reset_index(drop=True)

	df_db= df_db[['Block ID','Data Operation', 'Data Source Name']]
	df_db= df_db.replace(np.nan, '')
	df_final_db= copy.deepcopy(df_db)

	df_db['Data Source Name'] = similarity(list(df_db['Data Source Name']), 50)
	df_db['Data Operation'] = similarity(list(df_db['Data Operation']), 50)

	data= df_db[['Data Operation', 'Data Source Name']].values
	 



	#using MiniSom for clustering 
	from minisom import MiniSom    
	som = MiniSom(10,10, 2, sigma=0.5, learning_rate=0.5)
	som.train_random(data, 1000)
	x= som.win_map(data)

	cluster_dict={}
		   
	df_final_db['Cluster Center']= np.nan
	cluster_centers= {}
	count=0
	for cnt,xx in enumerate(data):
		w = som.winner(xx)
		if w not in cluster_centers.keys():
			cluster_centers[w]=count
			count+=1
		df_final_db['Cluster Center'].iloc[cnt]= cluster_centers[w]
		print(cnt)

		
		
		







	#Clustering non DB items
	dataframe2 = copy.deepcopy(dataframe)
	dataframe2 = dataframe[['Block ID','External Invoke', 'Data Operation', 'Data Source Name', 'Block Structure']]
	df_nondb = copy.deepcopy(dataframe2)
	#df_nondb= df_nondb.replace(', ',np.nan)
	df_nondb_final= pd.DataFrame(index= range(0,len(df_nondb)), columns=['Block ID','External Invoke', 'Block Structure'])
	count=0
	for index in range(0,len(df_nondb)):
		
		if df_nondb['Data Operation'].iloc[index]== "":
			print(df_nondb['Block ID'].iloc[index])
			df_nondb_final['Block ID'].iloc[count]=  df_nondb['Block ID'].iloc[index]
			df_nondb_final['Block Structure'].iloc[count]=  df_nondb['Block Structure'].iloc[index]
			df_nondb_final['External Invoke'].iloc[count]=  df_nondb['External Invoke'].iloc[index]
			count+=1

	df_nondb_final= df_nondb_final.dropna(how="all")
	df_nondb_final= df_nondb_final.fillna("")
	df_nondb_final2 = copy.deepcopy(df_nondb_final)

	df_nondb_final['Block Structure']= similarity(list(df_nondb_final['Block Structure']), 90)
	df_nondb_final['External Invoke']= similarity(list(df_nondb_final['External Invoke']), 90)

	data= df_nondb_final[['External Invoke', 'Block Structure']].values
	df_nondb_final['Cluster Center']= np.nan



	#Using Minisom for Clustering Purpose
	from minisom import MiniSom    
	som = MiniSom(10,10, 2, sigma=0.5, learning_rate=0.5)
	som.train_random(data, 1000)
	x= som.win_map(data)

	cluster_dict={}
		   
	cluster_centers= {}
	count=0
	for cnt,xx in enumerate(data):
		w = som.winner(xx)
		if w not in cluster_centers.keys():
			cluster_centers[w]=count
			count+=1
		df_nondb_final['Cluster Center'].iloc[cnt]= cluster_centers[w]
		print(cnt)

	df_nondb_final2['Cluster Center']= df_nondb_final['Cluster Center']




	# =============================================================================
	# df['Block Structure'] = dataframe['Block Structure']
	# 
	# 
	# df['Data Operation'] = similarity(list(df['Data Operation']), 90)
	# print("1")
	# df['Block Structure'] = similarity(list(df['Block Structure']), 90)
	# print("2")
	# 
	# df['Data Source Name'] = similarity(list(df['Data Source Name']), 90)
	# print("3")
	# 
	# df['External Invoke'] = similarity(list(df['External Invoke']), 90)
	# print("4")
	# 
	# 
	# 
	# #Grouping all similar records using Similarity Function
	# df2= copy.deepcopy(df)
	# df2= df2[['Data Operation', 'Block Structure', 'Data Source Name', 'External Invoke']]
	# data = df2.values
	# 
	# df3= pd.DataFrame(index= range(0,len(df2)), columns=['Data','Cluster'])
	# 
	# for index in range(0,len(df3)):
	#     df3['Data'].iloc[index]= str(df2['Data Operation'].iloc[index]) + "-" + str(df2['Block Structure'].iloc[index]) + "-" + str(df2['Data Source Name'].iloc[index]) + "-" + str(df2['External Invoke'].iloc[index])
	# 
	# data= df3.values
	# new_list= similarity(list(df3['Data']), 95)
	# 
	# df3['cluster']= new_list
	# dataframe2['Cluster Center']= new_list
	# df['Cluster Center']= new_list
	# 
	# plot(df)
	# =============================================================================













	#Converting to dictionary
	dataframe2= dataframe2.fillna("")
	dataframe2['Cluster Center']= np.nan
	new_dict = {}
	for index in range(0, len(dataframe2)):
	  new_dict[str(dataframe2['Block ID'].iloc[index])] = dataframe2['Cluster Center'].iloc[index]

	new_dict= new_dict.values() 

return df_finaldb,df_nondb_final2


x,y= cluster(r'C:\Users\u65988\cluster_final.csv')




