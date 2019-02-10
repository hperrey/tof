# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pandas as pd
import numpy as np
import dask.dataframe as dd
np.random.seed(666)
#Keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D



#training set
Training_set = dd.read_parquet('data/NandYseperated/T.pq', engine='pyarrow').reset_index()
neutrons = Training_set.query('62000<tof<92000 and 200000<cfd_trig_rise<1000000')
neutrons = neutrons.compute()
neutrons = neutrons.drop('index', axis=1)
neutrons = neutrons.reset_index()
gammas = Training_set.query('30000<tof<39000 and 200000<cfd_trig_rise<1000000')
gammas = gammas.compute()
gammas = gammas.drop('index', axis=1)
gammas = gammas.reset_index()

#testset
df_test = dd.read_parquet('data/NandYseperated/val.pq', engine='pyarrow').reset_index()
df_test = df_test.query(' 200000<cfd_trig_rise<1000000')
df_test = df_test.compute()
df_test = df_test.drop('index', axis=1)
df_test = df_test.reset_index()


def get_samples(df):
    S = np.array([None]*df.shape[0])
    #S = [0]*len(df)
    for i in range(0, len(df)):
        S[i] = df.samples[i][int(0.5 + df.cfd_trig_rise[i]/1000)-20: int(0.5 + df.cfd_trig_rise[i]/1000)+180]
    return S

Sn = get_samples(neutrons)
Sy = get_samples(gammas)
St = get_samples(df_test)


L=min([len(neutrons), len(gammas)])
print(L, ' samples of each species will be used')
window_width = len(Sn[0])#n_train.window_width[0]
r = 0.85
X1=np.stack(Sn[0:int(r*L)])#n_train.samples)
X2=np.stack(Sy[0:L])#y_train.samples)
x_train = np.concatenate([X1, X2]).reshape(L+int(r*L),window_width,1)
y_train = np.array([1]*int(r*L) + [0]*L)

x_test=np.stack(St)#df_test.samples)
x_test=x_test.reshape(len(x_test), window_width, 1)

#model definition
model = Sequential()
model.add(Conv1D(16, 7, strides=3, activation='relu', input_shape=(window_width, 1)))
model.add(Dropout(0.1))
model.add(MaxPooling1D(3, strides=3))
model.add(Conv1D(16, 7, strides=1, activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(3, stride=3))
model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=24, epochs=3)
predictions = model.predict(x_test)
df_test['pred']=predictions
df_test0 = df_test.query('pred<0.5')
df_test1 = df_test.query('pred>=0.5')

#ToF spectrum
plt.hist(df_test.tof/1000, bins=250, range=(0,500), alpha=0.25, label='Sum')
plt.hist(df_test0.tof/1000, bins=250, range=(0,500), histtype='step', lw=1.5, label='Gamma')
plt.hist(df_test1.tof/1000, bins=250, range=(0,500), histtype='step', lw=1.5, label='Neutron')
plt.legend()
plt.title('ToF spectrum \nfiltered by convolutional neural network\nTrained on 30 minute dataset, here tested on 10 minute dataset')
plt.ylabel('Counts')
plt.xlabel('t(ns)')
plt.show()

#Prediction space
plt.hist(df_test.pred, bins=50, label='Neutron region = 0.5-1, gamma region=0-0.5')
plt.legend()
plt.title('CNN prediction space\n the final layers output is the logistic function, so it is bounded between 0 and 1')
plt.ylabel('Counts')
plt.xlabel('CNN prediction')
plt.show()

#prediction vs QDC
plt.scatter(df_test0.qdc_lg/100, df_test0.pred, alpha=0.45, label='Gamma')
plt.scatter(df_test1.qdc_lg/100, df_test1.pred, alpha=0.45, label='Neutron')
plt.xlim(0,12500)
plt.legend()
plt.title('CNN predictions versus longgate QDC values')
plt.xlabel('qdc channel')
plt.ylabel('CNN prediction')
plt.show()
T=Training_set.query('0<tof<100000 and 0<cfd_trig_rise<650000')

#hexbin
plt.hexbin(df_test.qdc_lg/100, df_test.pred, gridsize=30, label=('10 minute dataset'),cmap='inferno')
plt.xlim(0,12500)
#plt.legend()
plt.title('CNN predictions versus longgate QDC values')
plt.xlabel('qdc channel')
plt.ylabel('CNN prediction')
plt.show()
# T=Training_set.query('0<tof<100000 and 0<cfd_trig_rise<650000')
# plt.hexbin(T.cfd_trig_rise.compute(), T.tof.compute(), gridsize=100)
# plt.xlabel('cfd_trigger time within window (ps)')
# plt.ylabel('tof (ps)')
# plt.show()
