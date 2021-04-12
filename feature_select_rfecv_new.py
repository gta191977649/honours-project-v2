from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 准备数据集
#PATH_DATA_SET = './dataset/selected_v_dataset.csv'
PATH_DATA_SET = './dataset/dataset_nor_zsocre.csv'

dataset = pd.read_csv(PATH_DATA_SET)
#data_x = dataset.loc[:,"audspec_lengthL1norm_sma_amean":"pcm_fftMag_psySharpness_sma_de_stddev"]
data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

train_csv = data_x
data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']
data_x = np.array(data_x)
data_valance_y = np.array(data_valance_y)
data_arousal_y = np.array(data_arousal_y)



min_features_to_select = 1
X, y = data_x,data_valance_y
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=10,verbose=3,n_jobs=-1)
selector = selector.fit(X, y)
feature_names = data_x.columns

mask = selector.get_support()
new_features = []
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)


print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(selector.grid_scores_) + min_features_to_select),
         selector.grid_scores_)
plt.show()