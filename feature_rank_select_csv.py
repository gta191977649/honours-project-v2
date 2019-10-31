import pandas as pd

path_rank_valance = "./rank_valance.csv"
path_dataset = "./dataset/dataset_nor_zsocre.csv"
def generateSelectedFeatureSet(rankset,anotation):
    v = pd.read_csv(rankset)
    dataset = pd.read_csv(path_dataset)
    FIRST_NTH_FEATURE = 10
    # 取得前10的feature
    selectedFeatures = v["feature"][:FIRST_NTH_FEATURE].tolist()
    selectedFeatures.append('v')
    selectedFeatures.append('a')
    selectedFeatures.append('song_id')
    # 构造新的数据集
    newDataset = pd.DataFrame()
    for col in dataset:
        if col in selectedFeatures:
            newDataset[col] = dataset[col]
            #print(col)
    newDataset.to_csv("./dataset/selected_{}_dataset.csv".format(anotation))

generateSelectedFeatureSet("./rank_valance.csv","v")
generateSelectedFeatureSet("./rank_arousal.csv","a")