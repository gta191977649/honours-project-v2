import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rfpimp import *
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold
import json
from utils import printHeader

#LIMIT
#70% Traning, 30 Testing
LIMIT = 1260
PATH_DATA_SET = 'D:/dev/honours-project-v2/dataset/dataset_nor_zsocre.csv'
dataset = pd.read_csv(PATH_DATA_SET)[:1260]
data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']

#train_x, test_x, train_y, test_y = train_test_split(data_x, data_valance_y, test_size=0.20, shuffle=True)

def rfe_rank(label_set,filename):
    # SVR Regressor
    print("Start rfe feature selection for {}".format(filename))
    svr = SVR(kernel="linear")

    rfe = RFE(svr, n_features_to_select=1, verbose=3)
    rfe.fit(data_x, label_set)

    print("Optimal number of features : %d" % rfe.n_features_)
    print(rfe.ranking_)
    # print feature importances
    feature_cols = data_x.columns.values
    rfe_rank = []

    for idx, _ in enumerate(rfe.ranking_):
        # rfe_rank[feature_cols[idx]] = rfe.ranking_[idx]
        rfe_rank.append({'feature': feature_cols[idx], 'rank': rfe.ranking_[idx]})
    rfe_csv = pd.DataFrame(rfe_rank)
    rfe_csv = rfe_csv.sort_values(by='rank')
    rfe_csv.to_csv('./{}.csv'.format(filename))



printHeader()

rfe_rank(data_valance_y,"rank_rfe_valance")
rfe_rank(data_arousal_y,"rank_rfe_arousal")
