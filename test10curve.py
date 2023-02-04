import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
#%%
ks = pd.read_excel(r'C:/Users/Personal/Documents/GitHub/lstm_srail/data_for_training.xlsx')
ks.head()
#%% plotdata
plt.figure(figsize=(6,6))
plt.title('Force-Disp')
plt.plot(ks['Disp'],ks['Force'])
plt.xlabel('Disp', fontsize=18)
plt.ylabel('Force', fontsize=18)
plt.show()
#%% Normalize Data
df = ks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
def scale_data(df):
  data = df.values
  scaler = MinMaxScaler(feature_range=(-1,1))
  scaled_data = scaler.fit_transform(data)
  data = pd.DataFrame(scaled_data, columns=[df.columns])
  return data, scaler

scaled_df, scaler = scale_data(df)
scaled_df
#%% Reshape data to (batch_size, time_step, feature_dimension)
def setup_data(df, time_step = 2):
  X = []
  Y = []
  for i in range(df.index.size - time_step -1):
    if df.iloc[i, 0] == df.iloc[i+1,0] and df.iloc[i,1] == df.iloc[i+1,1] and df.iloc[i, 2] == df.iloc[i+1,2] and df.iloc[i,3] == df.iloc[i+1,3]:
     X.append(df[i:i + time_step].values)
     Y.append(df.iloc[i + time_step, [4,5]])
  X = np.array(X)
  Y = np.array(Y)
  return X, Y

X_data, Y_data = setup_data(scaled_df)  
X_data.shape, Y_data.shape

#%%
row_step = np.arange(398,32318,399, dtype=int)
Y_data = np.delete(Y_data,row_step,axis=0)
X_data = np.delete(X_data,row_step,axis=0)
#%% LSTM
import keras
import tensorflow as tf
from keras import backend
from keras.layers import Dropout
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Flatten
model = Sequential()
model.add(LSTM(50, activation="relu", return_sequences=True, use_bias=True, input_shape=(X_data.shape[1], X_data.shape[2])))
model.add(LSTM(25, activation="relu"))
model.add(Flatten())
model.add(Dense(25))
model.add(Dense(2))
model.add(Dense(Y_data.shape[1], activation="linear"))

model.summary()
#%%
import keras.backend as K

def clear_session():
    K.clear_session()

clear_session()
#%%
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
#Train the model
history = model.fit(X_data, Y_data, batch_size=64, verbose=1, epochs=50)
#model.save('lstm_81.h5')
#%%
# test model
result = model.evaluate(X_data, Y_data)
print("Test MSE loss: ", result[0])
print("Test MSE accuracy: ", result[1])
#%%
# define constant 
test_scaled = scaler.fit_transform(ks)
test_scaled
test_df = ks
#%% executable with unseen data
time_step = 2
test_constant = [-1, -1, -1, -1] #test_scaled[0][:4]  #first 4 columns (TR A L T)
test_constant = np.array([test_constant] * time_step)
print("constant: ", test_constant)
N = 398 #test_df.index.size - time_step
print("N predict: ", N)
#%%
# get first time step data
test_data = [-1, -1]
test_data = np.array([test_data] * time_step)
#test_data = test_scaled[:time_step, -2:]
#test_data
#%%
pred_result = test_data.tolist()
pred_result

#model = tf.keras.models.load_model('lstm_81.h5')
#%%
# loop to predict all data
for i in range(N):
  # get time step test data 
  test_input = np.array(pred_result[i:i + time_step])
  # transfrom test data to input format
  test_input = np.array([np.concatenate((test_constant, test_input), 1)])
  pred = model.predict(test_input)[0]
  # append prediction result 
  pred_result.append(pred.tolist())

pred_result = np.array(pred_result)
pred_result.shape
#%%
model.save('lstm_81.h5')
#%%
pred_df = pd.DataFrame(data=pred_result)
pred_df.head()
plt.plot(pred_df[1],pred_df[0],linewidth = 0.8, color = 'red')
