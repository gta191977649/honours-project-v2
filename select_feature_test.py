''''
This file use the seleted feature generate from the rfcv to test the model.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

PATH_DATA_SET = './dataset/dataset_nor_zsocre.csv'
PATH_SELECTED_FEATURES_SET = './selected_feature/serlected_fetures.csv'
dataset = pd.read_csv(PATH_DATA_SET)[:1260]
selected_feature = pd.read_csv(PATH_SELECTED_FEATURES_SET)

data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']

def svrModelTestArousal():
    feature_arosal = []
    for feature in selected_feature["SVR_A_74"].tolist():
        if not pd.isnull(feature) :
            feature_arosal.append(feature)

    data_x = dataset.loc[:,feature_arosal]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
              kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
    svr.fit(train_x, train_y)
    score = svr.score(test_x, test_y)
    return score

def svrModelTestValance():
    features = []
    for feature in selected_feature["SVR_V_115"].tolist():
        if not pd.isnull(feature) :
            features.append(feature)

    data_x = dataset.loc[:,features]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
              kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
    svr.fit(train_x, train_y)
    score = svr.score(test_x, test_y)
    return score

def rfModelTestArousal():
    feature_arosal = []
    for feature in selected_feature["RF_A_38"].tolist():
        if not pd.isnull(feature) :
            feature_arosal.append(feature)

    data_x = dataset.loc[:,feature_arosal]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    RFR = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                                max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=100,
                                oob_score=False, random_state=0, verbose=0, warm_start=False, )
    RFR.fit(train_x, train_y)
    score = RFR.score(test_x, test_y)
    return score

def rfModelTestValance():
    features = []
    for feature in selected_feature["RF_V_203"].tolist():
        if not pd.isnull(feature) :
            features.append(feature)

    data_x = dataset.loc[:,features]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    RFR = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                                max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=100,
                                oob_score=False, random_state=0, verbose=0, warm_start=False, )
    RFR.fit(train_x, train_y)
    score = RFR.score(test_x, test_y)
    return score

def rfModelTestNoArousal():
    data_x = dataset.loc[:, 'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    RFR = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                                max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=100,
                                oob_score=False, random_state=0, verbose=0, warm_start=False, )
    RFR.fit(train_x, train_y)
    score = RFR.score(test_x, test_y)
    return score

def rfModelTestNoValance():
    data_x = dataset.loc[:, 'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    RFR = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                                max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=100,
                                oob_score=False, random_state=0, verbose=0, warm_start=False, )
    RFR.fit(train_x, train_y)
    score = RFR.score(test_x, test_y)
    return score



def svrModelTestNoSelectArousal():
    data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
              kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
    svr.fit(train_x, train_y)
    score = svr.score(test_x, test_y)
    return score

def svrModelTestNoSelectValance():
    data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_arousal_y, test_size=0.20, shuffle=True)
    svr = SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
              kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
    svr.fit(train_x, train_y)
    score = svr.score(test_x, test_y)
    return score

if __name__ == '__main__':
    testCycle = 10

    score_svr_a = []
    score_svr_v = []
    score_svr_a_no = []
    score_svr_v_no = []

    score_rf_a = []
    score_rf_v = []
    score_rf_a_no = []
    score_rf_v_no = []


    for i in range(0,testCycle):
        # SVR
        score_svr_a.append(svrModelTestArousal())
        score_svr_a_no.append(svrModelTestNoSelectArousal())
        score_svr_v.append(svrModelTestValance())
        score_svr_v_no.append(svrModelTestNoSelectValance())
        # RF
        score_rf_a.append(rfModelTestArousal())
        score_rf_v.append(rfModelTestValance())
        score_rf_a_no.append(rfModelTestNoArousal())
        score_rf_v_no.append(rfModelTestNoValance())




    # SVR
    svr_a_average = np.average(np.array(score_svr_a))
    print("SVR Arousal Average: {}".format(svr_a_average))
    svr_a_average_no = np.average(np.array(score_svr_a_no))
    print("SVR Arousal No Select Average: {}".format(svr_a_average_no))
    svr_v_average = np.average(np.array(score_svr_v))
    print("SVR Valance Average: {}".format(svr_v_average))
    svr_v_average_no = np.average(np.array(score_svr_v))
    print("SVR Valance No Select Average: {}".format(svr_v_average_no))

    # RF
    rf_a_average = np.average(np.array(score_rf_a))
    print("SVR Arousal Average: {}".format(rf_a_average))
    rf_v_average_no = np.average(np.array(score_rf_v_no))
    print("SVR Arousal No Select Average: {}".format(rf_v_average_no))
    rf_v_average = np.average(np.array(score_rf_v))
    print("SVR Valance Average: {}".format(rf_v_average))
    rf_v_average_no = np.average(np.array(score_rf_v_no))
    print("SVR Valance No Select Average: {}".format(rf_v_average_no))


    # chart
    """
    plt.plot(range(0, 10), score_svr_a, linewidth=2.0)
    plt.ylim([0, 1])
    plt.show()
    """

