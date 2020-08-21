import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

model = models.Sequential()
model.add(layers.Conv2D(260, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

PATH_DATA_SET = './dataset/dataset_nor_zsocre.csv'
dataset = pd.read_csv(PATH_DATA_SET)

label_valance = dataset.loc[:,'v']
label_arousal = dataset.loc[:,'a']

data_x = dataset.loc[:,'F0final_sma_stddev':'pcm_fftMag_mfcc_sma_de[14]_amean']

train_x, test_x, train_y, test_y = train_test_split(data_x, label_valance, test_size=0.20, shuffle=True)


history = model.fit(train_x, train_y, epochs=10, 
                    validation_data=(test_x, test_y))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
