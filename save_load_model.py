#%%
model.save('lstm_81.h5')
#%%
loadedmodel = tf.keras.models.load_model('lstm_1.h5')
