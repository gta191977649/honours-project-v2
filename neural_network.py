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
PATH_DATA_SET = './dataset/selected_v_dataset.csv'

dataset = pd.read_csv(PATH_DATA_SET)
data_x = dataset.loc[:,"audspec_lengthL1norm_sma_amean":"pcm_fftMag_psySharpness_sma_de_stddev"]
train_csv = data_x

data_valance_y = dataset.loc[:,'v']
data_arousal_y = dataset.loc[:,'a']

data_x = np.array(data_x)

data_valance_y = np.array(data_valance_y)
data_arousal_y = np.array(data_arousal_y)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# 配置Baseline
def network_model():
    model = Sequential()

    model.add(Dense(260,input_dim=10, activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#evaluate
# estimator = KerasRegressor(build_fn=network_model, epochs=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=2)
# results = cross_val_score(estimator, data_x, data_valance_y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
train_x,test_x,train_y,test_y = train_test_split(data_x, data_valance_y, test_size=0.20,shuffle=True)

model = network_model()
model.fit(train_x,train_y,epochs=100,batch_size=40)
model.evaluate(test_x,test_y)
model.summary()
print("loss:{}, acc:{}".format(val_loss,val_acc))
