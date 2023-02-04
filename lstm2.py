import pandas as pd
import numpy as np

data = pd.read_excel(r'C:/Users/Personal/Documents/GitHub/lstm_srail/data_for_training.xlsx')
data = data.values
#%% normalize data
# minmax method at each column
norm_TR = (data[:32400,0]-np.amin(data[:32400,0]))/(np.amax(data[:32400,0])-np.amin(data[:32400,0]));
norm_A = (data[:32400,1]-np.amin(data[:32400,1]))/(np.amax(data[:32400,1])-np.amin(data[:32400,1]));
norm_L = (data[:32400,2]-np.amin(data[:32400,2]))/(np.amax(data[:32400,2])-np.amin(data[:32400,2]));
norm_T = (data[:32400,3]-np.amin(data[:32400,3]))/(np.amax(data[:32400,3])-np.amin(data[:32400,3]));

norm_force = (data[:32400,4]-np.amin(data[:32400,4]))/(np.amax(data[:32400,4])-np.amin(data[:32400,4]));
norm_disp  = (data[:32400,5]-np.amin(data[:32400,5]))/(np.amax(data[:32400,5])-np.amin(data[:32400,5]));
# concatenate each normalized column
norm_data = np.column_stack((norm_TR,norm_A,norm_L,norm_T,norm_force,norm_disp))
#%% normalize in a cool way
zeros = np.zeros((32400))
col_ind = list(range(0,6))
for column in col_ind:
    eiei = (data[:32400,column]-np.amin(data[:32400,column]))/(np.amax(data[:32400,column])-np.amin(data[:32400,column]));
    zeros = np.column_stack((zeros,eiei))
norm_data = zeros[:,1:]
#%% reshape data into (batch_size,time_step,feature_dim) input shape and (batch_size,target_dim) target shape
norm_data = pd.DataFrame(norm_data)
def setup_data(norm_data, time_step = 2):
  X = []
  Y = []
  for i in range(norm_data.index.size - time_step -1):
    if norm_data.iloc[i, 0] == norm_data.iloc[i+1,0] and norm_data.iloc[i,1] == norm_data.iloc[i+1,1] and norm_data.iloc[i, 2] == norm_data.iloc[i+1,2] and norm_data.iloc[i,3] == norm_data.iloc[i+1,3]:
     X.append(norm_data[i:i + time_step].values)
     Y.append(norm_data.iloc[i + time_step, [4,5]])
  X = np.array(X)
  Y = np.array(Y)
  return X, Y

X_data, Y_data = setup_data(norm_data)  
#%% delete data from overlap window
row_step = np.arange(398,32318,399, dtype=int)
Y_data = np.delete(Y_data,row_step,axis=0)
X_data = np.delete(X_data,row_step,axis=0)
#%% LSTM configuration
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define the model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(2, 6)))
model.add(LSTM(units=25, activation="relu"))
model.add(Dense(units=25))
model.add(Dense(units=4))
model.add(Dense(units=2, activation='relu'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Print a summary of the model
model.summary()

# Train the model
model.fit(X_data, Y_data, epochs=100, verbose=1, batch_size=32)
#%%
result = model.evaluate(X_data, Y_data)
print("Test MSE loss: ", result)
#%% executable with unseen data
time_step = 2

test_constant = [0, 0, 0, 1] #test_scaled[0][:4]  #first 4 columns (TR A L T)
test_constant = np.array([test_constant] * time_step)
print("constant: ", test_constant)

N = 398 #test_df.index.size - time_step
print("N predict: ", N)

# get first time step data
test_data = [0, 0]
test_data = np.array([test_data] * time_step)
print("test data: ", test_data)

pred_result = test_data.tolist()
pred_result
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

pred_df = pd.DataFrame(data=pred_result)
pred_df.head()
plt.plot(pred_df[0],linewidth = 0.8, color = 'red')
plt.plot(pred_df[1],linewidth = 0.8, color = 'blue')
#%%
pred_df = pd.DataFrame(data=pred_result)
pred_df.head()
plt.plot(pred_df[1],pred_df[0],linewidth = 0.8, color = 'red')
#%%
plt.plot(pred_df[0],linewidth = 0.8, color = 'red')
plt.plot(pred_df[1],linewidth = 0.8, color = 'blue')
#%%
model.save('lstm_SP.h5')
