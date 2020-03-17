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

WORD = re.compile(r'\w+')
capability_dict = {'Purchase Order': 13, 'Slot Assignment': 11}  # Input Dict for mapping capability


def similarity(new_list):
  copy_list = copy.deepcopy(new_list)
  added_index = []
  count = 0
  for i in range(0, len(new_list) - 1):

    if i not in added_index:
      index_list = []
      index_list.append(i)
      for j in range(i + 1, len(new_list)):

        vector1 = text_to_vector(new_list[i])
        vector2 = text_to_vector(new_list[j])

        cosine = get_cosine(vector1, vector2)
        cosine = cosine * 100

        if cosine > 80:
          index_list.append(j)
          added_index.append(j)

      print(index_list)

      for index in index_list:
        copy_list[index] = count

    count += 1

  return copy_list


def get_cosine(vec1, vec2):
  intersection = set(vec1.keys()) & set(vec2.keys())
  numerator = sum([vec1[x] * vec2[x] for x in intersection])

  sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
  sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
  denominator = math.sqrt(sum1) * math.sqrt(sum2)

  if not denominator:
    return 0.0
  else:
    return float(numerator) / denominator


def text_to_vector(text):
  words = WORD.findall(text)
  return Counter(words)



def calc_distance(x1, y1, a, b, c):
  d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
  return d



def Optimum_K_Value(df):
  data = df.values
  max_length = len(df) - 1
  max_length = 50
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

def b_cluster():
    # Getting all the execution Block Structures
    dataframe = pd.read_csv(r'cluster_latest.csv')
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
        structure += " " + str((dataframe['Expression Type ID'].iloc[index]))
      else:
        structure += " " + str((dataframe['Expression Type ID'].iloc[index]))
        structure_list.append(structure)
        structure_dict[dataframe['Block ID'].iloc[index]] = structure
        structure = ''

    print("There are ", len(structure_list), " no of Execution Blocks")

    # Mapping Blocks with Data Operations
    dataframe = pd.read_csv(r'cluster_latest.csv')
    dataframe = dataframe.sort_values(by=['Block ID'])
    dataframe = dataframe.fillna('')
    data_operation = ''
    data_source_name = ''
    Block_Data_Map = {}
    Block_Data_Name_Map = {}

    for index in range(0, len(dataframe) - 1):

      if dataframe['Block ID'].iloc[index] == dataframe['Block ID'].iloc[index + 1]:

        if dataframe['Data Operation Type'].iloc[index] != '':
          data_operation += dataframe['Data Operation Type'].iloc[index] + ", "
          data_source_name += dataframe['Data Source Name'].iloc[index] + ", "

      else:

        data_operation += dataframe['Data Operation Type'].iloc[index]
        data_source_name += dataframe['Data Source Name'].iloc[index]
        Block_Data_Map[dataframe['Block ID'].iloc[index]] = data_operation
        Block_Data_Name_Map[dataframe['Block ID'].iloc[index]] = data_source_name
        data_operation = ''
        data_source_name = ''

    # Mapping Blocks with External Invokes
    dataframe2 = pd.read_csv(r'cluster_latest.csv')
    dataframe2 = dataframe2.sort_values(by=['Block ID'])
    dataframe2 = dataframe2.fillna('')
    external_invoke = ''
    external_invoke_dict = {}

    for index in range(0, len(dataframe2) - 1):

      if dataframe2['Block ID'].iloc[index] == dataframe2['Block ID'].iloc[index + 1]:
        if dataframe2['Expression Type Name'].iloc[index] == 'EXTERNAL_INVOKE':
          external_invoke += str(dataframe2['Expression Name'].iloc[index]) + ", "

      else:
        if dataframe2['Expression Type Name'].iloc[index] == 'EXTERNAL_INVOKE':
          external_invoke += str(dataframe2['Expression Name'].iloc[index]) + ", "
        external_invoke_dict[dataframe2['Block ID'].iloc[index]] = external_invoke
        external_invoke = ''



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





    # Label Encoding
    dataframe2 = copy.deepcopy(dataframe)
    dataframe = dataframe[['Block ID','External Invoke', 'Data Operation', 'Data Source Name', 'Block Structure']]
    #dataframe = dataframe.drop_duplicates()
    #dataframe = dataframe.reset_index(drop=True)
    df = copy.deepcopy(dataframe)
    label_encoder = preprocessing.LabelEncoder()
    df['Data Operation'] = label_encoder.fit_transform(df['Data Operation'])
    df['Block Structure'] = label_encoder.fit_transform(df['Block Structure'])
    df['Data Source Name'] = label_encoder.fit_transform(df['Data Source Name'])
    df['External Invoke'] = label_encoder.fit_transform(df['External Invoke'])
    data = df.values





    # Finding Optimum K value
    #Optimum_K_Value(df)


    #Kmeans Clustering
    dataframe['Cluster Center'] = np.nan
    dataframe2['Cluster Center'] = np.nan

    model = KMeans(n_clusters=int(len(df)/3), random_state=0).fit(data)
    for index in range(0, len(df)):
      cluster_center = model.predict(data[index].reshape(1, -1))
      dataframe['Cluster Center'].loc[index] = int(cluster_center)
      dataframe2['Cluster Center'].loc[index] = int(cluster_center)






    # Capability Association
    dataframe2['Capability'] = np.nan
    CU_Cluster_map = {}

    for index in range(0, len(dataframe2)):
      CU_Cluster_map[dataframe2['Block ID'].iloc[index]] = dataframe2['Cluster Center'].iloc[index]

    for item in capability_dict.keys():

      cluster_center = CU_Cluster_map[capability_dict[item]]

      for index in range(0, len(dataframe2)):

        if dataframe2['Cluster Center'].iloc[index] == cluster_center:
          dataframe2['Capability'].iloc[index] = item






    #Converting to dictionary
    dataframe2= dataframe2.fillna("")
    new_dict = {}
    for index in range(0, len(dataframe2)):
      new_dict[str(dataframe2['Block ID'].iloc[index])] = dataframe2['Cluster Center'].iloc[index]

    new_dict= new_dict.values()


    return new_dict



if __name__== "main":
  b_cluster()
