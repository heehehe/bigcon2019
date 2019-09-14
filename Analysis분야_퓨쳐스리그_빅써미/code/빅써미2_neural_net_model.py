#!/usr/bin/env python
# coding: utf-8

# # 패키지 로드

# In[402]:


import pandas as pd
import numpy as np
import keras as keras

from sklearn.ensemble import *
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.datasets import make_classification


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop, SGD, Adamax
from keras.models import load_model

import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
from sklearn.metrics import confusion_matrix


# # 데이터 준비, 정제

# In[403]:


train_dataset=pd.read_csv('data/train_FINAL.csv',encoding='cp949')


# In[405]:


X = train_dataset.drop('DLY', axis=1)
X2 = train_dataset.drop('DLY', axis=1)
y = train_dataset['DLY']


# In[406]:


min_max=MinMaxScaler()


# In[407]:


X=min_max.fit_transform(X) #변수 정규화


# In[409]:


X=pd.DataFrame(X,columns=X2.columns)


# In[410]:


X


# In[411]:


print("Total Dataset Shape")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = np.nan_to_num(X_train)
print("Train / Test Set Size")
print(len(X_train), len(y_train), len(X_test), len(y_test))


# In[412]:


np.random.seed(5)


# In[413]:


LEARNING_RATE = 0.05
DROPOUT_RATIO = 0.3
BATCH_SIZE = 100
EPSILON = 0.01 

def delay_model(x_size, y_size):
    model = Sequential()
    model.add(Dense(30, activation='relu', input_shape=(x_size, )))
    Dropout(DROPOUT_RATIO)
    model.add(Dense(15, activation='relu'))
    Dropout(DROPOUT_RATIO)
    model.add(Dense(y_size, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=LEARNING_RATE, epsilon=EPSILON),
                  metrics=['accuracy'])
    return model


# In[435]:


model1 = delay_model(x_size=X_train.shape[1], y_size=1)


# In[436]:


history = model1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=20, verbose=1, validation_data=(X_test, y_test), shuffle=True)


# In[439]:


def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)
    
    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['acc'])
    plt.plot(h['val_acc'])
    plt.title('Training vs Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return


# In[440]:


plot_hist(history.history, xsize=8, ysize=12)


# # 모델적합

# In[441]:


model1.save('model/predict_winrating.h1')

np.save('model/X_train.npy', X_train)
np.save('model/X_test.npy', X_test)
np.save('model/y_train.npy', y_train)
np.save('model/y_test.npy', y_test)


# In[442]:


X_train_predicted = model1.predict(X_train)


# In[443]:


X_test_predicted = model1.predict(X_test)


# # 모델평가

# ## Train data

# In[444]:


df=pd.DataFrame(X_train_predicted)
df2=pd.DataFrame(y_train)
df2=df2.reset_index()
df=pd.concat([df2,df], axis=1)
df.columns=['index', 'DLY', 'Predicted']
df['Predicted_DLY']=''
df


# In[445]:


df.groupby('DLY').mean()


# In[446]:


for i in range(len(df)):
    if df.Predicted[i]>0.174999:
        df['Predicted_DLY'][i] = 1
    else : 
        df['Predicted_DLY'][i] = 0


# In[447]:


df.Predicted_DLY=df.Predicted_DLY.astype('int')


# In[448]:


confusion_matrix(df.DLY, df.Predicted_DLY, labels=[0, 1])


# In[449]:


accuracy_score(df.DLY, df.Predicted_DLY) # Train set accuracy score


# ## Validation Data

# In[450]:


dft=pd.DataFrame(X_test_predicted)
dft2=pd.DataFrame(y_test)
dft2=dft2.reset_index()
dft=pd.concat([dft2,dft], axis=1)
dft.columns=['index', 'DLY', 'Predicted']
dft['Predicted_DLY']=''
dft


# In[451]:


for i in range(len(dft)):
    if dft.Predicted[i]>0.174999:
        dft['Predicted_DLY'][i] = 1
    else : 
        dft['Predicted_DLY'][i] = 0


# In[452]:


dft.Predicted_DLY=dft.Predicted_DLY.astype('int')


# In[453]:


confusion_matrix(dft.DLY, dft.Predicted_DLY, labels=[0, 1])


# In[454]:


accuracy_score(dft.DLY, dft.Predicted_DLY) # Validation set accuracy score


# # Test set Fitting

# In[466]:


test_dataset=pd.read_csv('data/test_FINAL.csv',encoding='cp949')


# In[467]:


test_dataset


# In[468]:


X3 = test_dataset.drop('DLY', axis=1)
X4 = test_dataset.drop('DLY', axis=1)
y2 = test_dataset['DLY']


# In[469]:


X3=min_max.fit_transform(X3) #변수 정규화
X3=pd.DataFrame(X3,columns=X4.columns)


# In[470]:


X3


# In[472]:


X_realtest_predicted = model1.predict(X3) # 모형 적합


# In[479]:


dfrt=pd.DataFrame(X_realtest_predicted)
dfrt.columns=['DLY_RATE']
dfrt['DLY']=''


# In[486]:


for i in range(len(dfrt)):
    if dfrt.DLY_RATE[i]>0.174999:
        dfrt['DLY'][i] = 1
    else : 
        dfrt['DLY'][i] = 0


# In[500]:


dfrt


# In[490]:


result=pd.read_csv('data/AFSNT_DLY.csv',encoding='cp949')


# In[514]:


result['ARP'][2]='ARP1'


# In[515]:


result['DLY']=dfrt['DLY']
result['DLY_RATE']=dfrt['DLY_RATE']


# In[516]:


result.to_csv('data/AFSNT_DLY.csv',encoding='cp949') # save

