from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

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

# 配置Baseline
def network_model():
    model = Sequential()

    model.add(Dense(108,input_dim=260, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #adam = Adam(0.00001,0.99,0.999)

    model.compile(loss='mse', optimizer='sgd',metrics=['accuracy'])
    return model

#evaluate
# estimator = KerasRegressor(build_fn=network_model, epochs=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=2)
# results = cross_val_score(estimator, data_x, data_valance_y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
train_x,test_x,train_y,test_y = train_test_split(data_x, data_valance_y, test_size=0.20,shuffle=True)

model = network_model()
model.fit(train_x,train_y,epochs=100,batch_size=40)

val_loss, val_acc = model.evaluate(test_x,test_y)
print("loss:{}, acc:{}".format(val_loss,val_acc))
