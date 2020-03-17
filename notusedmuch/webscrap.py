# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:25:25 2020

@author: U65988
"""

import time
import os
import requests


def convert_to_html():
    
    for year in range(2006,2020):
        
        for month in range(1,13):
            
            if month < 10:
                url = 'https://www.tutiempo.net/clima/0%x-%y/ws-433530.html'.format(month, year)
            else:
                url = 'https://www.tutiempo.net/clima/%x-%y/ws-433530.html'.format(month, year)
    
    
            texts= requests.get(url)
            text_utf= texts.text.encode('utf=8')
    
            if not os.path.exists('Data/HTML_Data/{}'.format(year)):
                os.makedirs('Data/HTML_Data/{}'.format(year))
            with open('Data/HTML_Data/{}/{}.html'.format(year,month), "wb") as output:
                output.write(text_utf)
            
    
    sys.stdout.flush()


if __name__== '__main__':
    convert_to_html()