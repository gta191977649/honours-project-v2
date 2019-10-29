from os import system
import os
import pandas as pd
import json
import numpy as np
from os.path import splitext
import pyfiglet

path_song_feature = os.path.join("./dataset/song_features")
path_song_arousal = "./dataset/arousal.csv"
path_song_valence = "./dataset/valence.csv"

#输出文件
label = []
# 一首个的feature求平均
def meanSongFeature(filePath):
    csv = pd.read_csv(path,delimiter=';')
    csv = csv[30:90]
    csv = csv.drop(columns=['frameTime'])
    mean_csv = csv.mean()
    return mean_csv
    #print(mean_csv)

    # s = mean_csv.to_json()
    # s = json.loads(s)
    # s['a'] = 0.22
    # s['v'] = 0.54
    # df = pd.DataFrame([s])
    # df.to_csv('./labeled.csv', encoding='utf-8', index=False)

# 根据songid找到该歌曲的V&A的值
def processMeanAnotations(annotation_csv_path,songid):
    label = pd.read_csv(annotation_csv_path)
    # 根据所给的songid找到改数据的行
    label = label.loc[label['song_id'] == songid]
    label = label.drop(columns=['song_id'])
    # 切分（只提取15-44.5范围的数据）
    label = label.loc[:, 'sample_15000ms':'sample_44500ms'].values.tolist()
    label = np.array(label)
    label = label.reshape(-1)
    label = np.mean(label)
    return label

def getVAFromSongId(songid):
    # Process valance
    v = processMeanAnotations(path_song_valence,songid)
    a = processMeanAnotations(path_song_arousal,songid)
    #print("v:{} a:{}".format(v,a))
    return v,a
# LIMIT = 10
counter = 0
#处理label
for root,dirs,files in os.walk(path_song_feature):
    for file in files:
        if file.endswith(".csv"):
            # if(counter >= LIMIT): break

            path = path_song_feature + "/" + file
            #取得song id
            song_id = splitext(file)[0]

            #获取平均特征
            mean_feature = meanSongFeature(path)
            #获取平均 v a 标签
            v, a = getVAFromSongId(int(song_id))

            #构造buffer json
            song_item = json.loads(mean_feature.to_json())
            song_item['v'] = v
            song_item['a'] = a
            song_item['song_id'] = song_id

            label.append(song_item)

            system('cls')
            # Print logo
            print(pyfiglet.figlet_format("Sparrow Research"))
            print("(C) Project Sparrow 2019 Dev: Nurupo27\n")
            print("Processsing songid {}, total:{}, remains:{}, finish:{}%".format(song_id,len(files),len(files)-counter,round(counter/len(files) *100,2) ),flush=True)
            counter+=1

# 保存CSV
df = pd.DataFrame(label)
df.to_csv('./labeled.csv', encoding='utf-8', index=False)



