#!/usr/bin/env python
# coding: utf-8

# Import statements


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os  
from sklearn.model_selection import train_test_split   # Test_Train split
import json
import glob
import random
import collections
from glob import glob
import numpy as np
import pandas as pd
#import cv2                       # Replaced by pillow. You can uncomment and use OpenCV for faster read. 
import matplotlib.pyplot as plt   # Visualization
import shutil
import keras                       
from PIL import Image             # Pillow for Image IO // slightly slower than OpenCV


#  Data Read using glob  


#  Adding the filenames to an array

from glob import glob
trn1='/src/invasive-aquatic-species-data/invasive/*/'
trn2='/src/invasive-aquatic-species-data/noninvasive/*/'
tr1= glob(trn1)
tr2= glob(trn2)

print("\n\n Imports completed.  Reading Data ----\n ")

# In[ ]:     Adding all the filenames to array 'data' and the labels to array 'label'.  0 for invasisve and 1 for non-invasisve

data = []
label = []
for i in tr1:
    for j in glob(i+'/*'):
        data.append(j)
        label.append(0)


for i in tr2:
    for j in glob(i+'/*'):
        data.append(j)
        label.append(1)
        
        
# Adding all the images to array 'imgdata', reshaped to (40,40,3) and normalization.

imgdata=[]
for i in range(len(data)):
    a = Image.open(data[i])
    b = a.resize((40, 40))
    c = np.array(b)
    imgdata.append(c.reshape(40,40,3))
    
    
#creating categorical labels and normalizing data
    
from tensorflow.keras.utils import to_categorical
idata = np.array(imgdata)
x_train = idata
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train),40,40,3))
# One hot vector representation of labels
yy_labels = to_categorical(label)


# Test Train split 

X_train, X_test, y_train, y_test = train_test_split( x_train, yy_labels, shuffle=True, test_size=0.25, random_state=42)



from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K

# Function for F1 Score

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
    

# Model imports    
    
import numpy as np
import pandas as pd


from keras import layers
from keras import models
import keras
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Model Defination. Uncomment model.summary for visualization

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(40, 40, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),))
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))
#model.summary()


# Model compilation . 30% of the data used for validation


print("\n\n Model training : with F1 score ----\n ")


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=[get_f1])

model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split = .3, verbose = 1)


# Making prediction on the test data. Round off to test the results on sklearn

pred = model.predict(X_test)
p = np.round(pred)

f1 = get_f1(y_test, pred)

print ("\n\n  This is the F1 Score for the test data : \n")
print(f1)



from sklearn.metrics import f1_score
f1 = f1_score(y_test, p, average='weighted')

print ("\n\n  This is the SKLearn weighted F1 score : \n")
print(f1,"\n")

y_p = []
for i in range(len(p)):
    if ( p[i][0] == 1 ):
        y_p.append(0)
    else :
        y_p.append(1)
y_p = np.array(y_p)
y_t = []
for i in range(len(y_test)):
    if ( y_test[i][0] == 1 ):
        y_t.append(0)
    else :
        y_t.append(1)
y_t = np.array(y_t)


from sklearn.metrics import confusion_matrix
a=np.flip(confusion_matrix(y_t, y_p , labels=[0,1]))

print ("\n\n  This is the SKLearn confusion matrix : \n")
print(a,"\n")


