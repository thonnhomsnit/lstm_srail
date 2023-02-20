import csv
import numpy as np
import pandas as pd
import tensorflow as tf

x = pd.read_csv(r'C:\Users\Personal\Documents\GitHub\optimization-execution\matlabinput.csv')
x = x.values

model = tf.keras.models.load_model(r'C:\Users\Personal\Documents\GitHub\optimization-execution\lstm_force_disp_time_relu.h5')
norm_TR = (x[0,0]-1)/(2-1);
norm_A  = (x[0,1]-120)/(160-120);
norm_L  = (x[0,2]-300)/(500-300);
norm_T  = (x[0,3]-2)/(5-2);

test_constant = [norm_TR, norm_A, norm_L, norm_T]
test_constant = np.array([test_constant] * 2)

test_data = np.array([[0, 0, 0],[0, 0, 0.00250627]])

pred_result = test_data.tolist()
for i in range(360):
      test_input = np.array(pred_result[i:i + 2])
      test_input = np.array([np.concatenate((test_constant, test_input), 1)])
      pred = model.predict(test_input)[0]
      pred_result.append(pred.tolist())

pred_result = np.array(pred_result)

y = (pred_result[:,0])*(163.2073517)
y_list=np.array([y]).reshape(-1,1)
with open(r'C:\Users\Personal\Documents\GitHub\optimization-execution\pythonoutput.csv', 'w', newline='') as file:
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(y_list)

