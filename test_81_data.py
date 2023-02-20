#%% cell1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

data = pd.read_excel(r'C:/Users/Personal/Documents/GitHub/lstm_srail/data_for_training.xlsx', sheet_name='data_for_training')
data = data.values

forcetest = pd.read_excel(r'testing_data.xlsx', sheet_name='forcetime')
forcetest = forcetest.iloc[:400,1:10]
forcetest.columns = [1,2,3,4,5,6,7,8,9]

disptest = pd.read_excel(r'testing_data.xlsx', sheet_name='disptime')
disptest = disptest.iloc[:400,1:10]
disptest.columns = [1,2,3,4,5,6,7,8,9]
#%% cell2
for i in range(1,10):    
    disptest.head()
    plt.plot(disptest[i],forcetest[i])
    plt.show()
#%% cell3
import tensorflow as tf
model = tf.keras.models.load_model('lstm_force_time_relu.h5')
#%% cell4 
config = pd.read_excel(r'testing_data.xlsx', sheet_name='81config')
config = config.iloc[:4,:81]
config = config.values
fea_force = pd.read_excel(r'lstm_full_81.xlsx', sheet_name='Force')
fea_force = fea_force.values
fea_disp = pd.read_excel(r'lstm_full_81.xlsx', sheet_name='Disp')
fea_disp = fea_disp.values

disp = np.zeros(400)
force = np.zeros(400)
    
time_step = 2

for i in range(81):
    TR = config[0,i]
    A = config[1,i]
    L = config[2,i]
    T = config[3,i]
    
    norm_TR = (TR-1)/(2-1);
    norm_A = (A-120)/(160-120);
    norm_L = (L-300)/(500-300);
    norm_T = (T-2)/(5-2);
    
    norm_TR=float(format(norm_TR,'.2f'))
    norm_A=float(format(norm_A,'.2f'))
    norm_L=float(format(norm_L,'.2f'))
    norm_T=float(format(norm_T,'.2f'))

    test_constant = [norm_TR, norm_A, norm_L, norm_T] #(TR A L T)
    test_constant = np.array([test_constant] * time_step)
    print("constant: ", test_constant)

    N = 398 #test_df.index.size - time_step
    print("N predict: ", N)

    # get first time step data
    test_data = np.array([[0, 0],[0, 0.00250627]])
    #test_data = np.array([test_data] * time_step)
    print("test data: ", test_data)

    pred_result = test_data.tolist()
    pred_result
    # loop to predict all data
    for j in range(N):
      # get time step test data 
      test_input = np.array(pred_result[j:j + time_step])
      # transfrom test data to input format
      test_input = np.array([np.concatenate((test_constant, test_input), 1)])
      pred = model.predict(test_input)[0]
      # append prediction result 
      pred_result.append(pred.tolist())

    pred_result = np.array(pred_result)

    denorm_force = (pred_result[:,0])*(np.amax(data[:32400,4])-np.amin(data[:32400,4]))+np.amin(data[:32400,4])
    denorm_time  = (pred_result[:,1])*(np.amax(data[:32400,5])-np.amin(data[:32400,5]))+np.amin(data[:32400,5])
   # denorm_time  = (pred_result[:,2])*(np.amax(data[:32400,6])-np.amin(data[:32400,6]))+np.amin(data[:32400,6])
    
   # denorm_force = (pred_result[:,0]+1)*(np.amax(data[:32400,4])-np.amin(data[:32400,4]))/2+np.amin(data[:32400,4])
   # denorm_disp  = (pred_result[:,1]+1)*(np.amax(data[:32400,5])-np.amin(data[:32400,5]))/2+np.amin(data[:32400,5])
   
    #plt.plot(denorm_disp)
    #disp = np.column_stack((disp,denorm_disp))
    #plt.plot(fea_disp[:,i])
    #plt.show()
    
    plt.plot(denorm_force)
    force = np.column_stack((force,denorm_force))
    plt.plot(fea_force[:,i])
    #plt.show()
   # plt.plot(denorm_time)
    #plt.plot(disptest[i],forcetest[i])
    #plt.xlim([-5, 320])
    plt.show()

from numpy import savetxt
savetxt('force.csv', force, delimiter=',')
savetxt('disp.csv', disp, delimiter=',')
#%%
for k in range(82):
    plt.plot(force[:,k])
    plt.xlim([0, 500])
    plt.ylim([0, 200])
    plt.plot([0,400],[260,260])
    plt.show
#%%
fea_disp = pd.read_excel(r'lstm_full_81.xlsx', sheet_name='Force')
fea_disp = fea_disp.values
for k in range(81):
    plt.plot(fea_disp[:,k])
    plt.xlim([0, 500])
    plt.ylim([0, 200])
    #plt.plot([0,400],[260,260])
    plt.show

