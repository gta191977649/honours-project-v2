import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from math import *
# 准备数据集
PATH_DATA_SET = './dataset/dataset_nor_zsocre.csv'

dataset = pd.read_csv(PATH_DATA_SET)

data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
train_csv = data_x

data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']

data_x = np.array(data_x)

data_valance_y = np.array(data_valance_y)
data_arousal_y = np.array(data_arousal_y)

#配置Random forest Regressor (Valance)
def runRFRRegressor(train_x,train_y,test_x,test_y,rankingFileName):
    RFR = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=100,
               oob_score=False, random_state=0, verbose=0, warm_start=False,)
    RFR.fit(train_x,train_y)

    # 测试模型
    v_predict = RFR.predict(test_x)

    mse = mean_squared_error(test_y, v_predict)
    rmse = sqrt(mse)

    evaluateCsv = pd.DataFrame(columns=['predict', 'truth'])
    evaluateCsv['predict'] = v_predict
    evaluateCsv['truth'] = test_y
    evaluateCsv.to_csv('./e.csv')

    score = RFR.score(test_x, test_y)
    print("MSE:{},RMSE:{},SCORE:{}".format(mse, rmse, score))

    # 列出重要Features
    featureImportance = pd.DataFrame(
        {'feature': list(train_csv.columns), 'importance': RFR.feature_importances_}).sort_values('importance',
                                                                                                  ascending=False)
    featureImportance.to_csv("./{}.csv".format(rankingFileName))

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def gridSearchCVRFR(train_x,train_y):
    param_grid = {
        "n_estimators":[100,200,300,400,500,600,700,800,900,1000],
        "max_features":["auto","sqrt","log2"],
        "criterion":["mse","mae"]
    }
    clf = RandomForestRegressor(n_estimators=20,n_jobs=60)
    grid_search = GridSearchCV(clf,param_grid=param_grid,cv=5,iid=False,verbose=3)
    grid_search.fit(train_x,train_y)
    report(grid_search.cv_results_)

#训练Valance
def doGridSearch():
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_valance_y, test_size=0.20, shuffle=True)
    gridSearchCVRFR(train_x, train_y)
def doNormalRegressor():
    print("Estamateing Valance:")
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_valance_y, test_size=0.20, shuffle=True)
    runRFRRegressor(train_x,train_y,test_x,test_y,"rank_valance")
    print("Estamateing Arousal:")
    train_x,test_x,train_y,test_y = train_test_split(data_x, data_arousal_y, test_size=0.20,shuffle=True)
    runRFRRegressor(train_x,train_y,test_x,test_y,"rank_arousal")

doNormalRegressor()
#doGridSearch()