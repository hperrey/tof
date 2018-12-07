import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras import optimizers
import numpy as np

# fix random seed for reproducibility
numpy.random.seed(666)


MaxQDC=6000
N=pd.read_hdf('data/2018-12-04/N_cooked.h5').query('1500<longgate<=%d'%MaxQDC)
N['species']=np.array([0]*len(N))
#N=pd.read_hdf('1160_species.h5')
#M0=N.query('35<dt<42').reset_index()
M0=N.query('31<=dt<=33').reset_index()
M0['species']=np.array([0]*len(M0))
M0=M0.head(n=130)
M1=N.query('60<dt<85').reset_index()
M1['species']=np.array([1]*len(M1))
M1=M1.head(n=130)
M=pd.concat([M0,M1])
M=M.sample(frac=1).reset_index(drop=True)
#M['ps']=(M.qdc_det0-M.qdc_sg_det0)/M.qdc_det0
M['longgate']=M.longgate/MaxQDC
M['shortgate']=M.shortgate/MaxQDC
M['height']=M['height']/250


K=len(M)

M_train = M.head(n=K)
X_train=M_train[['longgate','ps', 'height', 'shortgate']]
Y_train=M_train.species#the goals

M_test = N#.head(n=50000)
#M_test['ps']=(M_test.qdc_det0-M_test.qdc_sg_det0)/M_test.qdc_det0
M_test['longgate']=M_test.longgate/MaxQDC
M_test['shortgate']=M_test.shortgate/MaxQDC
M_test['height']=M_test['height']/250

X_test=M_test[['longgate','ps', 'height', 'shortgate']]
Y_test=M_test.species#the goals

# create model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
optimizers.Adam(lr=0.5, beta_1=0.7, beta_2=0.9, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=100, batch_size=11)
scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)

M_test['predict'] = predictions
plt.scatter(M_test.query('predict>0.50').longgate*MaxQDC, M_test.query('predict>0.5').ps, s=10, alpha=0.35, label='test data classified as neutrons')
plt.scatter(M_test.query('predict<=0.5').longgate*MaxQDC, M_test.query('predict<=0.5').ps,s=10, alpha=0.35, label='test data classified as gammas')
plt.scatter(M1.longgate, M1.ps, label='training data (neutron peak)')
plt.scatter(M0.longgate, M0.ps, label='training data (gamma peak)')
plt.legend()
plt.show()

K = pd.read_hdf('data/2018-10-23/N_cooked.h5')
plt.hist(M_test.dt, range=(20,120), bins=50, histtype='step', lw=2, label='Unfiltered (With QDC cut at 1500)')
#plt.hist(K.dt, range=(20,120), bins=50, histtype='step', lw=2, label='Unfiltered (Without QDC cut)')
plt.hist(M_test.query('predict<=0.5').dt, range=(20,120), bins=50, histtype='step', lw=2, label='Gamma prediction')
plt.hist(M_test.query('predict>0.5').dt, range=(20,120), bins=50, histtype='step', lw=2, label='Neutron prediction')
plt.legend()
plt.show()
