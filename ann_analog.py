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

#N=pd.read_hdf('data/2018-10-23/N_cooked.h5')#.head(n=10000)
N=pd.read_hdf('1160_species.h5').query('0<(qdc_det0-qdc_sg_det0)/qdc_det0<1')
M0=N.query('695<tdc_det0_yap0<715').reset_index()
M0['species']=np.array([0]*len(M0))
M0['ps']=(M0.qdc_det0-M0.qdc_sg_det0)/M0.qdc_det0
#M0=M0.head(n=5859)
M1=N.query('550<tdc_det0_yap0<595').reset_index()
M1['species']=np.array([1]*len(M1))
M1['ps']=(M1.qdc_det0-M1.qdc_sg_det0)/M1.qdc_det0
#M1=M1.head(n=5859)
M=pd.concat([M0,M1])
M=M.sample(frac=1).reset_index(drop=True)
#M['ps']=(M.qdc_det0-M.qdc_sg_det0)/M.qdc_det0
M['qdc_det0']=M.qdc_det0/4000


K=len(M)

M_train = M.head(n=K)
X_train=M_train[['qdc_det0','ps']]
Y_train=M_train.species#the goals

M_test = N.query('0<qdc_det0<4000 and 0<(qdc_det0-qdc_sg_det0)/qdc_det0<1').head(n=500000)
M_test['ps']=(M_test.qdc_det0-M_test.qdc_sg_det0)/M_test.qdc_det0
M_test['qdc_det0']=M_test.qdc_det0/4000

X_test=M_test[['qdc_det0','ps']]
#Y_test=M_test.species#the goals

# create model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
optimizers.Adam(lr=0.2, beta_1=0.7, beta_2=0.9, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=8, batch_size=50)
scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)
M_test['predict']=predictions

M_test['predict'] = rounded

plt.scatter(M_test.query('predict==1').qdc_det0, M_test.query('predict==1').ps, s=2, alpha=0.35)
plt.scatter(M_test.query('predict==0').qdc_det0, M_test.query('predict==0').ps,s=2, alpha=0.35)
#plt.scatter(M1.qdc_det0/4000, M1.ps, label='training data (neutron peak)')
#plt.scatter(M0.qdc_det0/4000, M0.ps, label='training data (gamma peak)')
plt.legend()
plt.show()

K=pd.read_hdf('1160_species.h5')
plt.hist(M_test.tdc_det0_yap0, range=(400,1000), bins=600, histtype='step', lw=2, label='Unfiltered (No QDC cut)')
#plt.hist(K.tdc_det0_yap0, range=(400,1000), bins=600, histtype='step', lw=2, label='Unfiltered (QDC cut at 1500)')
plt.hist(M_test.query('predict==0').tdc_det0_yap0, range=(400,1000), bins=600, histtype='step', lw=2, label='Gamma prediction')
plt.hist(M_test.query('predict==1').tdc_det0_yap0, range=(400,1000), bins=600, histtype='step', lw=2, label='Neutron prediction')
plt.legend()
plt.show()

