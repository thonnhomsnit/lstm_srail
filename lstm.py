#%%
# input at first timestep
first_input = np.array([[0.5, 0.1, 0.1, 0.1]]) # [TR A L T]
N = 399
#%%
# output at first timestep
first_output = np.array([[0, 0]])  # [force disp]
pred_result = first_output.values.tolist()
#%%
# loop to predict all data
time_step = 1
for i in range(N):
  # get time step test data 
  test_input = np.array(pred_result[i:i + time_step])
  # transfrom test data to input format
  test_input = np.array([np.concatenate((feature_input,test_input),1)])
  pred = model.predict(test_input)[0]
  # append prediction result 
  pred_result.append(pred.tolist())

pred_result = np.array(pred_result)
pred_result.shape