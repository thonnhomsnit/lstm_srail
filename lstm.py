import pandas as pd
import numpy as np
import tensorflow as tf
import csv

test_constant = pd.read_csv(r'C:\Users\Personal\Documents\GitHub\lstm_srail\matlabinput.csv')
test_constant = test_constant.values.tolist()
test_constant = np.array([test_constant[0]] * 2) # 2 = timestep
test_data = [-1, -1]
test_data = np.array([test_data] * 2) # 2 = timestep
N = 398

model = tf.keras.models.load_model(r'C:\Users\Personal\Documents\GitHub\lstm_srail\lstm_81.h5')

# loop to predict all data
pred_result = test_data.tolist()
pred_result
for i in range(N):
  # get time step test data 
  test_input = np.array(pred_result[i:i + 2])
  # transfrom test data to input format
  test_input = np.array([np.concatenate((test_constant, test_input), 1)])
  pred = model.predict(test_input)[0]
  # append prediction result 
  pred_result.append(pred.tolist())

pred_result = np.array(pred_result)

with open(r'C:\Users\Personal\Documents\GitHub\lstm_srail\pythonoutput.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pred_result)
    #%%
    pred_df = pd.DataFrame(data=pred_result)
    pred_df.head()
    plt.plot(pred_df[1],pred_df[0],linewidth = 0.8, color = 'red')
