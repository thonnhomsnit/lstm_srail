#%%
model.save('lstm_81.h5')
#%%
model = tf.keras.models.load_model('lstm_81.h5')
model.weights
