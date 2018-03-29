# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:23:28 2018

@author: Dmob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import turtle
#import time
import random
import dbm
import pickle
import copy
from collections import namedtuple
import re
import gensim
import os
#os.chdir('C:\Users\me\Documents')

#wd=os.getcwd()
#a = os.chdir(r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science')
#wd2=os.getcwd()
print(wd2)


df=pd.read_csv("TRAIN.csv", sep=',',header=None)
df.values
