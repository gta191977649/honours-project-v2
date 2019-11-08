import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import numpy as np
from utils import printHeader
from os import system

PATH_DATA_SET = 'D:/dev/honours-project-v2/dataset/dataset_nor_zsocre.csv'

PATH_RANK_A = './rank_rfe_arousal.csv'
PATH_RANK_V = './rank_rfe_valance.csv'


dataset = pd.read_csv(PATH_DATA_SET)
data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']

# DEAM SET
data_x = np.array(data_x)
data_valance_y = np.array(data_valance_y)
data_arousal_y = np.array(data_arousal_y)

# OUR SET

def generateBestArousalTrainX(best_n_of_feature):
    a = pd.read_csv(PATH_RANK_A)
    bestFeatures = a["feature"][:best_n_of_feature]
    train_x = pd.DataFrame()
    for feature in bestFeatures:
        train_x[feature] = dataset[feature]
    #print(train_x)
    return train_x
def generateBestValanceTrainX(best_n_of_feature):
    a = pd.read_csv(PATH_RANK_V)
    bestFeatures = a["feature"][:best_n_of_feature]
    train_x = pd.DataFrame()
    for feature in bestFeatures:
        train_x[feature] = dataset[feature]
    #print(train_x)
    return train_x

def cv_dataset(data_x,data_y,filename):
    print("Cross vaildation for {}".format(filename))
    cv_scores = []
    #regressor = SVR(kernel='rbf',verbose=1)
    regressor = RandomForestRegressor(criterion='mse',n_jobs=100)

    cv = KFold(n_splits=10, random_state=None, shuffle=False)
    n_of_split = 0
    for train_index, test_index in cv.split(data_x):
        n_of_split+=1
        system('cls')
        printHeader()
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)

        X_train, X_test, y_train, y_test = data_x[train_index], data_x[test_index], data_y[train_index], data_y[test_index]
        regressor.fit(X_train, y_train)

        cv_scores.append({"traning_size":n_of_split,"score":regressor.score(X_test, y_test)})


    rfe_csv = pd.DataFrame(cv_scores)
    #rfe_csv = rfe_csv.sort_values(by='rank')
    rfe_csv.to_csv('./{}.csv'.format(filename))
    print(cv_scores)

#
cv_dataset(data_x,data_arousal_y,"cv_deam_a")
cv_dataset(data_x,data_valance_y,"cv_deam_v")

sf_train_a_x = generateBestArousalTrainX(133)
sf_train_v_x = generateBestValanceTrainX(139)
sf_train_a_x = np.array(sf_train_a_x)
sf_train_v_x = np.array(sf_train_v_x)

cv_dataset(sf_train_a_x,data_arousal_y,"cv_sf_a")
cv_dataset(sf_train_v_x,data_valance_y,"cv_sf_v")
