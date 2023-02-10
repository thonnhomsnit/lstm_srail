import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ks = pd.read_excel(r'C:/Users/Personal/Documents/GitHub/lstm_srail/data_for_training.xlsx', sheet_name='data_for_training')
df=ks

from sklearn.preprocessing import MinMaxScaler, StandardScaler
def scale_data(df):
  data = df.values
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data)
  data = pd.DataFrame(scaled_data, columns=[df.columns])
  return data, scaler

scaled_df, scaler = scale_data(df)
scaled_df
#%%
def setup_data(df, time_step = 2):
  X = []
  Y = []
  for i in range(df.index.size - time_step -1):
    if df.iloc[i, 0] == df.iloc[i+1,0] and df.iloc[i,1] == df.iloc[i+1,1] and df.iloc[i, 2] == df.iloc[i+1,2] and df.iloc[i,3] == df.iloc[i+1,3]:
     X.append(df[i:i + time_step].values)
     Y.append(df.iloc[i + time_step, [4,5,6]])
  X = np.array(X)
  Y = np.array(Y)
  return X, Y

X_data, Y_data = setup_data(scaled_df)  
X_data.shape, Y_data.shape
#%%
import keras
from keras import backend
from keras.layers import Dropout
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Flatten

model = Sequential()
model.add(LSTM(50, activation="relu", return_sequences=True, use_bias=True, input_shape=(X_data.shape[1], X_data.shape[2])))
model.add(LSTM(25, activation="relu"))
model.add(Flatten())
model.add(Dense(25))
#model.add(Flatten())
model.add(Dense(2))
model.add(Dense(Y_data.shape[1], activation="tanh"))

model.summary()
# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])

#Train the model
history = model.fit(X_data, Y_data, batch_size=32, verbose=1, epochs=50)
#%%
model.save('lstm_colab.h5')