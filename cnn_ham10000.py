import numpy as np
import pandas as pd
import matplotlib
from sklearn.utils import shuffle
matplotlib.use('agg')
import pylab as plt

## Purpose of the code: classification of skin cancer images using
## a deep convolutional neural network

data_frame = pd.read_csv('HAM10000_metadata.csv')

## We have to create a balanced dataset due to the huge size of one
## sample (we will do copies of the other samples so as to have 
## all categories with the same amount of images)

skin_df = pd.read_csv('/home/fran/Manuel/hmnist_28_28_RGB.csv')
data_frame = pd.read_csv('HAM10000_metadata.csv')

df0 = np.asarray(skin_df[skin_df['label'] == 0].drop(['label'], axis = 1))
df1 = np.asarray(skin_df[skin_df['label'] == 1].drop(['label'], axis = 1))
df2 = np.asarray(skin_df[skin_df['label'] == 2].drop(['label'], axis = 1))
df3 = np.asarray(skin_df[skin_df['label'] == 3].drop(['label'], axis = 1))
df4 = np.asarray(skin_df[skin_df['label'] == 4].drop(['label'], axis = 1))
df5 = np.asarray(skin_df[skin_df['label'] == 5].drop(['label'], axis = 1))
df6 = np.asarray(skin_df[skin_df['label'] == 6].drop(['label'], axis = 1))

## Let us create the balanced dataset and then use the concatenate function
## to have a good working array as input data

df0_bal = np.repeat(df0, 20, axis = 0)
df1_bal = np.repeat(df1, 13, axis = 0)
df2_bal = np.repeat(df2, 6, axis = 0)
df3_bal = np.repeat(df3, 55, axis = 0)
df4_bal = np.repeat(df4, 1, axis = 0)
df5_bal = np.repeat(df5, 45, axis = 0)
df6_bal = np.repeat(df6, 6, axis = 0)

df = np.concatenate((df0_bal, df1_bal, df2_bal, df3_bal, 
                    df4_bal, df5_bal, df6_bal), axis = 0)

## We repeat exactly the same with the y vector (labels for each disease)

y0 = np.repeat(0, len(df0_bal))
y1 = np.repeat(1, len(df1_bal))
y2 = np.repeat(2, len(df2_bal))
y3 = np.repeat(3, len(df3_bal))
y4 = np.repeat(4, len(df4_bal))
y5 = np.repeat(5, len(df5_bal))
y6 = np.repeat(6, len(df6_bal))

y = np.concatenate((y0, y1, y2, y3, 
                    y4, y5, y6), axis = 0)


from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Activation , Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy 
import os
import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

## We split the data into training, testing and validation

X_train, X_val_test, y_train, y_val_test = train_test_split(df, y,
                                                  test_size= .5, shuffle = True, random_state=1)

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test,
                                                  test_size= .5, shuffle = True ,random_state=1)


## Normalization of data (they have already been randomized and split into
## two categories)

X_train = X_train/255.
X_val = X_val/255.
X_test = X_test/255.
num_classes = 7

## We define the convolutional model

model = Sequential()

model.add(Conv2D(32, (3, 3),padding='same', input_shape=(28, 28, 3), 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(5, (3, 3),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer= 'adam', metrics=['accuracy'])

model.summary()

## Reshaping of data to introduce them as an input to the network

X_train_cnn = np.array(X_train)
n_samples_train = len(y_train)
images_train = X_train_cnn.reshape(n_samples_train, 28, 28, 3)

batch_size = 1
epochs = 50

## Store the model in an .h5 file to be later used 

for i in range(1,epochs+1):
    print('Epoch',i)
    history = model.fit(x = images_train , y = y_train , batch_size = batch_size ,
       shuffle = "batch", epochs = 1, verbose = 1) 
    model.save('/home/fran/Manuel/cnn_ham10000.h5')

print('Model saved successfully')

## The model has been successfully saved. We have to evaluate
## its performance over the test set

from tensorflow.keras.models import load_model

model_cnn = load_model('/home/fran/Manuel/cnn_ham10000.h5')

X_test_cnn = np.array(X_test)
n_samples_test = len(X_test_cnn)
images_test = X_test_cnn.reshape(n_samples_test, 28, 28, 3)

predicciones_cnn = model_cnn.predict(images_test)
predicciones_label = np.zeros(len(predicciones_cnn))

for i in range(len(predicciones_cnn)):
    predicciones_label[i] = np.argmax(predicciones_cnn[i])

## We define the confussion matrix and we plot it

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

target_names = ['nv' , 'mel' , 'bkl' , 'bcc' , 'akiec' , 'vasc' , 'df' ]

conf_matrix = confusion_matrix(predicciones_label, y_test)
cm = conf_matrix
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('/home/fran/Manuel/confmatrixcnn.png')
