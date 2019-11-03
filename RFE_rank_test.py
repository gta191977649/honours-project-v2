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

#根据给定的feature建立特殊的特征集
def constructFeatureSet(selectedFeature):
    featureSet = {}
    for col in selectedFeature:
        featureSet[col] = database[col]

    #插入annotation
    featureSet['v'] = database['v']
    featureSet['a'] = database['a']
    featureSet['song_id'] = database['song_id']
    return pd.DataFrame(featureSet)

def runSVMRegressor(train_x,train_y,test_x,test_y):
    print("Run SVR Regressor")
    svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
    kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
    svr.fit(train_x,train_y)
    score = svr.score(test_x, test_y)
    return score
def runRFRegressor(train_x,train_y,test_x,test_y):
    print("Run RandomForest Regressor")
    RFR = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                                max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=100,
                                oob_score=False, random_state=0, verbose=0, warm_start=False, )
    RFR.fit(train_x, train_y)
    score = RFR.score(test_x, test_y)
    return score
def prepareDatasetForTraining(feature,annotation):
    col_len = len(feature.columns)
    fist_col = feature.columns[0]
    last_col = feature.columns[col_len - 4]

    data_x = feature.loc[:, fist_col:last_col]
    data_y = feature.loc[:, annotation]
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    # 拆分测试和训练
    return train_test_split(data_x, data_y, test_size=0.20, shuffle=True)

def rfeTest(rankPath,testingfor):
    data = pd.read_csv(rankPath)
    result = []

    selected_feature = []
    n_of_feature = 0
    for col in data["feature"]:
        selected_feature.append(col)
        n_of_feature+=1
        #1.构造测试数据集合
        feature = constructFeatureSet(selected_feature)

        #2.对每一个数据集合测试Regressor
        train_x, test_x, train_y, test_y = prepareDatasetForTraining(feature,testingfor)
        #score = runSVMRegressor(train_x,train_y,test_x,test_y)
        score = runRFRegressor(train_x,train_y,test_x,test_y)
        #3.获取regressor结果
        print("\n")
        system('cls')
        printHeader()
        print("Benchmark 第:{}个特征 for: {}".format(n_of_feature,testingfor))
        print("------------------- SVR Result ---------------------")
        print("SCORE:{}".format(score))
        print("----------------------------------------------------")
        result.append({"n_of_feature":n_of_feature,"score":score})

    return result

def startRFEFeatureTest():
    #需要测试2类，valance 和 arousal
    #测试 valance
    valanceScore = rfeTest(PATH_RANK_V,'v')
    #保存文件
    csv = pd.DataFrame(valanceScore)
    csv.to_csv('./rfe_benchmark_v.csv')
    #测试 arousal
    arousalScore = rfeTest(PATH_RANK_A,'a')
    # 保存文件
    csv = pd.DataFrame(arousalScore)
    csv.to_csv('./rfe_benchmark_a.csv')

    print("Benchmark Finish!")



startRFEFeatureTest()