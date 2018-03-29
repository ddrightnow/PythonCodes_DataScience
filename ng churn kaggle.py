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
import csv

#train_inputs = train_data.ix[:,0]
#train_labels = train_data.drop(0, axis=1)

DIR=r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science'
train_data = pd.read_csv(DIR+'/train.csv', delimiter=',')
test_data = pd.read_csv(DIR+'/test.csv', delimiter=',')

print(train_data.head(4))













