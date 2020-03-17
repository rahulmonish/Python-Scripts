# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:46:56 2020

@author: U65988
"""

f = open(r"D:\Python Scripts\newfile.txt", "r")
string= f.read()
import re

x= re.findall('\(.*?\)',string)
y=[]
for i in x:
    i= i[1:len(i)-1]
    i= i.lower()
    print(i)
    j= i.replace(' ', '-')
    y.append(j)



import os
a= os.listdir("D:\Deepminer_Wiki")
b=[]
for i in a:
    i= i[0:len(i)-3]
    i=i.lower()
    b.append(i)
    
c= list(set(b) - set(y))
d= list(set(b) - set(y))

notinb=[]
for i in y:
    if i not in b:
        notinb.append(i)

notiny=[]
for i in b:
    if i not in y:
        notiny.append(i)
