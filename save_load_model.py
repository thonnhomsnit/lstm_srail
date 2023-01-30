#%%
model.save('lstm_1.h5')
#%%
loadedmodel = tf.keras.models.load_model('lstm_1.h5')
