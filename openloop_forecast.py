import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sample read data
ks = pd.read_excel(r'C:\Users\Personal\Desktop\lstmdata.xlsx')

df=pd.DataFrame()
for step in range(1,50,55): #StartStopstep
  df_step=ks.iloc[::step, :].reset_index(drop=True)
  df=df.append(df_step,ignore_index=True)
#%%
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
    if df.iloc[i, 0] == df.iloc[i+1,0] and df.iloc[i,1] == df.iloc[i+1,1] and df.iloc[i,2] == df.iloc[i+1,2] and df.iloc[i,3] == df.iloc[i+1,3]:
     X.append(df[i:i + time_step].values)
     Y.append(df.iloc[i + time_step, [4, 5, 6]])
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
model.add(Dense(4))
model.add(Dense(Y_data.shape[1], activation="tanh"))

model.summary()

# Compile the model
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])

#Train the model
history = model.fit(X_data, Y_data, batch_size=128, verbose=1, epochs=50) #validation_split=0.1,
#%%
tdf = pd.read_excel(r'C:\Users\Personal\Desktop\lstmtest.xlsx')
test_df=pd.DataFrame()
for step in range(1,10,10):
  tdf_step=tdf.iloc[::step, :].reset_index(drop=True)
  test_df=test_df.append(tdf_step,ignore_index=True)

test_df
#%%
test_scaled = scaler.fit_transform(test_df)

time_step = 2
test_constant = test_scaled[0][:4]
test_constant = np.array([list(test_constant)] * time_step)
print("constant: ", test_constant)
N = test_df.index.size - time_step
print("N predict: ", N)

test_constant.shape

test_data = test_scaled[:time_step, -3:]
test_data
#%%
pred_result = test_data.tolist()
pred_result
#%%
test_scaled = pd.DataFrame(test_scaled, columns=[test_df.columns])
test_scaled
test_X, test_Y = setup_data(test_scaled, time_step = time_step)
test_X.shape ,test_Y.shape
pred = model.predict(test_X)
pred.shape
pred = np.concatenate((test_data, pred))
pred.shape
#%%
add_constant = np.array([list(test_scaled.iloc[0][:4].values)] * 3192)
add_constant.shape
add_constant
pred_result_step = np.concatenate((add_constant, pred), axis=1)
pred_result_step.shape
pred_df_step = pd.DataFrame(scaler.inverse_transform(pred_result_step), columns=[test_df.columns])
#%%
plt.plot(test_df['Force'])
plt.plot(pred_df_step['Force'])
#plt.xlim((2400, 4800))
#%%
plt.plot(test_df['Disp'])
plt.plot(pred_df_step['Disp'])
#plt.xlim((2400, 4800))