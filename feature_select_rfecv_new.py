from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# 准备数据集
#PATH_DATA_SET = './dataset/selected_v_dataset.csv'
PATH_DATA_SET = './dataset/dataset_nor_zsocre.csv'

dataset = pd.read_csv(PATH_DATA_SET)
#data_x = dataset.loc[:,"audspec_lengthL1norm_sma_amean":"pcm_fftMag_psySharpness_sma_de_stddev"]
data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']
feature_names = data_x.columns

train_csv = data_x
data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']
data_x = np.array(data_x)
data_valance_y = np.array(data_valance_y)
data_arousal_y = np.array(data_arousal_y)



#min_features_to_select = 3

#Chouse label --->
X, y = data_x,data_arousal_y

#estimator = SVR(cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto', kernel='linear', max_iter=-1, shrinking=True, tol=0.001)
estimator = RandomForestRegressor()
selector = RFECV(estimator, step=1, cv=10,n_jobs=-1,verbose=3, scoring="r2")
selector = selector.fit(X, y)


mask = selector.get_support()
new_features = []
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)


print("Optimal number of features : % d" % selector.n_features_)
print(new_features)
f = open("selected_feature_rf_a.txt","w")
for feature in new_features:
    f.write(feature+"\n")

data = {}
data["n_of_feature"] = range(0,len(selector.grid_scores_))
data["score"] = selector.grid_scores_
csv = pd.DataFrame(data)
csv.to_csv("selected_feature_rf_a_chart.csv")
print(csv)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(0,
               len(selector.grid_scores_) ),
         selector.grid_scores_)
plt.show()

