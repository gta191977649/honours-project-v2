import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

path_dataset = './dataset/dataset.csv'

dataset = pd.read_csv(path_dataset)
#切分出特征集部分
#dataset = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

normalizedCsv = pd.DataFrame()

for col in dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']:
    data = np.array(dataset[col])
    #归一化数据
    s = MinMaxScaler()
    #data = s.fit(data).transform(data)
    data = stats.zscore(data)
    dataset[col] = data

dataset.to_csv('./dataset/dataset_nor_zsocre.csv')

#验证zscore
for col in dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']:
    data = np.array(dataset[col])
    mean = np.array(data).mean()
    std = np.array(data).std()
    print("col:{}\t\tmean:{}\t\tstd:{}".format(col,mean,std))