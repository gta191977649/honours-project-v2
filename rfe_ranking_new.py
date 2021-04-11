from math import sqrt

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from utils import printHeader
from os import system
import os

PATH_DATA_SET = './dataset/dataset_nor_maxmean.csv'
database = pd.read_csv(PATH_DATA_SET)
PATH_RANK_A = './rank_rfe_arousal.csv'
PATH_RANK_V = './rank_rfe_valance.csv'

