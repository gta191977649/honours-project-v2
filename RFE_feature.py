import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rfpimp import *
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold

PATH_DATA_SET = 'D:/dev/honours-project-v2/dataset/dataset_nor_zsocre.csv'
dataset = pd.read_csv(PATH_DATA_SET)
data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']

#train_x, test_x, train_y, test_y = train_test_split(data_x, data_valance_y, test_size=0.20, shuffle=True)

#SVR Regressor
svr = SVR(kernel="linear")
#svr.fit(data_x,data_valance_y)

rfe = RFE(svr, 5, step=1)
rfe.fit(data_x, data_valance_y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()