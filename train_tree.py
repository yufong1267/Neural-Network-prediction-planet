from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import Adam


from find_path import GetFileList
import cv2


from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array

from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#建立模型
vgg19_model = keras.applications.vgg16.VGG16()  #預訓練模型->1000種類
model = Sequential()
for layer in vgg19_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(4 , activation='softmax'))
model.summary()

#導入訓練資料
from keras.preprocessing.image import load_img
# x ->照片轉乘陣列 , y->是哪種植物的機率 [1,0,0,0]
# 
data = []
label = []
train_path = 'work\\longan'
train_path2 = 'work\\longan\\'
## 取得龍眼的照片
DirnameList , FilenameList = GetFileList().FileList(path = train_path)


for filepath in FilenameList:
    #image = cv2.imread(train_path2 + filepath)
    #image = cv2.resize(image , (224, 224))
    image = load_img(train_path2 + str(filepath), target_size=(224, 224))
    image = np.asarray(image)
    data.append(image)

    #label[龍眼,芒果、馬路、旱地]
    label_image = [1 , 0 , 0 , 0]
    label.append(label_image)
    
print("龍眼的照片已經處理完")

train_path = 'work\\mango'
train_path2 = 'work\\mango\\'
## 取得龍眼的照片
DirnameList , FilenameList = GetFileList().FileList(path = train_path)


for filepath in FilenameList:
    #image = cv2.imread(train_path2 + filepath)
    #image = cv2.resize(image , (224, 224))
    image = load_img(train_path2 + str(filepath), target_size=(224, 224))
    image = np.asarray(image)
    data.append(image)

    #label [龍眼,芒果、馬路、旱地]
    label_image = [0 , 1 , 0 , 0]
    label.append(label_image)

    
print("芒果的照片已經處理完")

train_path = 'E:\\ipynb\\work\\road'
train_path2 = 'E:\\ipynb\\work\\road\\'
## 取得龍眼的照片
DirnameList , FilenameList = GetFileList().FileList(path = train_path)


for filepath in FilenameList:
    #image = cv2.imread(train_path2 + filepath)
    #image = cv2.resize(image , (224, 224))
    image = load_img(train_path2 + str(filepath), target_size=(224, 224))
    image = np.asarray(image)
    data.append(image)

    #label [龍眼,芒果、馬路、旱地]
    label_image = [0 , 0 , 1 , 0]
    label.append(label_image)

    
print("馬路的照片已經處理完")

train_path = 'E:\\ipynb\\work\\wasteland'
train_path2 = 'E:\\ipynb\\work\\wasteland\\'
## 取得龍眼的照片

DirnameList , FilenameList = GetFileList().FileList(path = train_path)
for filepath in FilenameList:
    #image = cv2.imread(train_path2 + filepath)
    #image = cv2.resize(image , (224, 224))
    image = load_img(train_path2 + str(filepath), target_size=(224, 224))
    image = np.asarray(image)
    data.append(image)

    #label [龍眼,芒果、馬路、旱地]
    label_image = [0 , 0 , 0 , 1]
    label.append(label_image)

    
print("旱地的照片已經處理完")

from sklearn.preprocessing import LabelBinarizer
# 圖片正規化
data = np.array(data, dtype='float') / 255.0
label = np.array(label)
from sklearn.model_selection import train_test_split
# 將資料分為 80% 訓練, 20% 測試
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.2, random_state=8787)
# 資料增強
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

EPOCHS = 100
LR = 1e-4
BS = 8

opt = Adam(lr=LR)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#計算訓練時間
import time
start = time.clock()

# 開始訓練
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

#print(model.evaluate(testX, testY))
model.save('tree_classify_vgg16.h5')
print('模型存檔完成')
print('共花費：',time.clock() - start)


import matplotlib.pyplot as plt
# 繪製 Acc 及 Loss
plt.style.use('ggplot')
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='upper left')
plt.savefig('tree_vgg16_loss.png')

print('收斂圖片完成')
