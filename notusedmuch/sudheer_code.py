# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:11:08 2020

@author: U65988
"""


import os 
import csv
import pandas as pd
import re
import numpy as np
import sys
cwd = os.getcwd()
path= str(sys.argv[0])
output_path= str(sys.argv[1])


def color(cell):
    string=''
    if 'Fail' in cell:
        string="<td><b><font color=\"red\">" + cell + "</td></b></font>"
    elif 'Pass' in cell:
        string= "<td><b><font color=\"green\">" + cell + "</td></b></font>"
    else:
        string= "<td>" + cell + "</td>"
    return string




def convert_csvtohtml(content, header=None, delimiter=","):
    rows = [x.strip() for x in content]
    table = "<table border=2>"
    if header is not None:
        table += "<tr style=\"background-color:#4DA0F3;color:white;\" >"
        table += "".join(["<th>" + cell + "</th>" for cell in header.split(delimiter)])
        table += "</tr>"

    else:
        table += "".join(["<th>" + cell + "</th>" for cell in rows[0].split(delimiter)])
        rows = rows[1:]
    expr=''
    for row in rows:
        #print(row)
        
        table += "<tr>" + "".join([ color(cell) for cell in row.split(delimiter)]) + "</tr>" + "\n"
        
    table += "</table><br>"
    return table







#Converting text file to CSV
with open(path, 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split("\t") for line in stripped if line)
    with open(str(cwd) + "\result_summary.csv", 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('ModuleName', 'ExecutionResult', 'Remarks','dump'))
        writer.writerows(lines)



#Creating the dataframe
#r"C:\Users\u65988\Documents\result_summary.csv"
df= pd.read_csv(str(cwd) + "\result_summary.csv")
df= df.dropna(how="all")
df= df.replace(',','-')



df= df[['ModuleName','ExecutionResult','Remarks']]
i=0
while i< len(df):
    if df['ModuleName'].at[i]=='ModuleName':
        df= df.drop(i)
        
    df=df.reset_index(drop=True)
    i+=1
i=0
while i< len(df):   
    if df['Remarks'].at[i]==np.nan:
        #print("help")
        df= df.drop(i)
    df=df.reset_index(drop=True)
    i+=1
    
    
   
header_list=[]
row_list=[]
header=''
for i in list(df.columns):
    header+=i+ ","
header_list.append(header)
df= df.replace(np.nan, "")

for i in range(0,len(df)):
    df['Remarks'].at[i]= str(df['Remarks'].at[i]).replace(',','-')
    

for i in range(0, len(df)):
    if str(df['ExecutionResult'].at[i])!= np.nan:
        string= str(df['ModuleName'].at[i]) + "," + str(df['ExecutionResult'].at[i]) + "," + str(df['Remarks'].at[i])
        row_list.append(string)






#Creating the HTML Page 
html=''
html+= "<h2><b><i>" + str(df['ModuleName'].at[0]) + str(df['ModuleName'].at[1]) + "</i></b></h2><br><br>"
row_list=[]
string=''
for i in range(0, len(df)):
    if df['ExecutionResult'].at[i]=="":
        if string!='':
            html+= convert_csvtohtml(row_list, header)+"<br><br>"
            row_list=[]
        df['ModuleName'].at[i]=df['ModuleName'].at[i].replace('<','').replace('>','') 
        if 'Feature' in str(df['ModuleName'].at[i]):
            html+= "<h3> " + df['ModuleName'].at[i] + " </h3>"
        elif 'Exception' in str(df['ModuleName'].at[i]):
            html+= "<h3> <font color=\"red\">" + df['ModuleName'].at[i] + " </font></h3>"
    else:
        string= str(df['ModuleName'].at[i]) + "," + str(df['ExecutionResult'].at[i]) + "," + str(df['Remarks'].at[i])
        row_list.append(string)
    
    if i==len(df)-1:
        html+= convert_csvtohtml(row_list, header)




#Writing to the output
text_file = open(output_path, "w")
n = text_file.write(html)
text_file.close()       
        