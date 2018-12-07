import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tof
np.random.seed(666)
#Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

N=pd.read_hdf('N5dt.h5')

dim = len(N.samples[0])

Train0 = pd.read_hdf('N5dt.h5').query('35<dt<42')
Train0['species']=np.array([0]*len(Train0))
Train1 = pd.read_hdf('N5dt.h5').query('65<dt<90')
Train1['species']=np.array([1]*len(Train1))
minLen = min( [len(Train0), len(Train1) ] )
Train0 = Train0.head(n=minLen)
Train1 = Train1.head(n=minLen)
Train = pd.concat([Train0, Train1]).sample(frac=1).reset_index(drop=True)
X_Train = Train.samples
Y_Train = Train.species #the goals


#defining the model
model = Sequential()
model.add(Conv1D(nb_filter=512, filter_length=1, activation='relu', input_shape=(dim, 272)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_Train, Y_Train, batch_size=10, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Score = ', score)
