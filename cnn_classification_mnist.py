# -*- coding: utf-8 -*-

"""
eng : Load libraries
tr : Kütüphanelerimizi yüklüyoruz.
"""
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

"""
eng : load and preprocess
tr : resimlerimizi yükleyip eğitilecek hale getiriyoruz.
"""
def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.as_matrix()
    np.random.shuffle(data)
    x = data[:,1:].reshape(-1,28,28,1)/255.0
    y = data[:,0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))

    return x,y

"""
eng : The path to the file where the training and testing data is.
tr :  train ve test datalarımızın olduğu dosyanın yolunu veriyoruz.
"""   
train_data_path = "mnist-in-csv\mnist_train.csv"
test_data_path = "mnist-in-csv\mnist_test.csv"

x_train,y_train = load_and_preprocess(train_data_path)
x_test, y_test = load_and_preprocess(test_data_path)

"""
eng : visualize
tr : görselleştirerek verimize bakıyoruz.
"""
index = 55
vis = x_train.reshape(60000,28,28)
plt.imshow(vis[index,:,:]) 
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index]))

"""
eng : We create our CNN model.
tr : CNN modelimizi oluşturuyoruz.
"""
numberOfClass = y_train.shape[1]

model = Sequential()

model.add(Conv2D(input_shape = (28, 28, 1), filters = 16, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 128, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(units = numberOfClass))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

"""
eng : We train our model that we have created
tr : oluşturduğumuz modelimizi eğitiyoruz
"""
hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs= 100, batch_size= 4000)

"""
eng : We keep the weights for later use.
tr : Ağırlıkları daha sonra kullanmak üzere saklıyoruz. Tekrar tekrar 100 epochs, zaman harcamamak için.
"""
model.save_weights('cnn_mnist_model.h5')

"""
eng : The loss and acc values ​​are checked by visualizing the model we have created.
tr : oluşturduğumuz modeli görselleştirerek loss ve acc değerlerine bakıyoruz.
""" 
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "Train Loss")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["acc"],label = "Train Accuracy")
plt.plot(hist.history["val_acc"],label = "Validation Accuracy")
plt.legend()
plt.show()

"""
eng : save history
tr : oluşturduğumuz modelimizi kayıt ediyoruz.
"""
import json
with open('cnn_mnist_hist.json', 'w') as f:
    json.dump(hist.history, f)
    
"""
eng : We read it to visualize the model we saved.
tr : Kaydettiğimiz modeli görselleştirmek için okuyoruz.
"""
import codecs
with codecs.open("cnn_mnist_hist.json", 'r', encoding='utf-8') as f:
    h = json.loads(f.read())

plt.figure()
plt.plot(h["loss"],label = "Train Loss")
plt.plot(h["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["acc"],label = "Train Accuracy")
plt.plot(h["val_acc"],label = "Validation Accuracy")
plt.legend()
plt.show()









































    