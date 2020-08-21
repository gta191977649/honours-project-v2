from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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

"""
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
"""
# 配置Baseline
def network_model():
    model = Sequential()

    model.add(Dense(260,input_dim=train_x.shape[1], activation='tanh'))
    model.add(Dense(256, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(256, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(256, kernel_initializer='normal', activation='tanh'))
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
history = model.fit(train_x,train_y,epochs=100,batch_size=40)
"""
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
"""
model.summary()
test_loss, test_acc = model.evaluate(test_x,test_y)
#mse = tf.keras.losses.MeanSquaredError()

print(test_acc)

#print("loss:{}, acc:{}".format(val_loss,val_acc))
