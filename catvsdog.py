import tensorflow
from tensorflow import keras
from matplotlib import pyplot
import os
import cv2
import random
import pickle
import numpy
categories=['Cat','Dog']
traindata=[]
def create_train_data(Data):
    Dir="E:\PetImages"
    categories=['Cat','Dog']
    for a in categories:
        Path=os.path.join(Dir,a)
        classnum=categories.index(a)
        for b in os.listdir(Path):
            try:
                img=cv2.imread(os.path.join(Path,b),cv2.IMREAD_GRAYSCALE)
                img=cv2.resize(img,(200,200))
                Data.append([img,classnum])
            except:
                pass
create_train_data(traindata)
random.shuffle(traindata)
xs=[]
y=[]
for a,b in traindata:
    xs.append(a)
    y.append(b)
y=numpy.array(y)
x=numpy.array(xs).reshape(-1,200,200,1)/255
testx=x[:100]
testy=y[:100]
x=x[100:]
y=y[100:]
modelname='catvsdog'
a=0
model=keras.Sequential([

    
    
    keras.layers.Conv2D(16,(3,3),use_bias=False,input_shape=(x.shape[1:])),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.Conv2D(16,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.MaxPooling2D(2,2),
                                    
    keras.layers.Conv2D(32,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
                                    
    keras.layers.Conv2D(32,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.MaxPooling2D(2,2),
                                    
    keras.layers.Conv2D(64,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
                                    
    keras.layers.Conv2D(64,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(128,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
                                    
    keras.layers.Conv2D(128,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.Conv2D(128,(3,3),use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.MaxPooling2D(2,2),                       
                                    
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2048,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2048,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation='sigmoid')
    ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=5)
z=modelname+str(a)
model.save(f'{z}.h5')
a+=1
