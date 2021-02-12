# unpickle train Dict and plot losses and other curves 
import matplotlib.pyplot as plt
import pickle
## Unet
with open('D:/trainHistoryDict_Dense1_all_in_resume_freeze', 'rb') as pi:
    u = pickle._Unpickler(pi)
    u.encoding = 'latin1'
    history = u.load()
    
with open('D:/trainHistoryDict_unet_all_in_resume', 'rb') as pi:
    u = pickle._Unpickler(pi)
    u.encoding = 'latin1'
    history_D = u.load()
## Dense
with open('D:/Work/mre_t2/results_normal_new/keras_all_seq_0.001/model_1/trainHistoryDict_Dense1_all_in', 'rb') as pi:
    u = pickle._Unpickler(pi)
    u.encoding = 'latin1'
    history_D = u.load()
    
loss=history['loss']

# plot the losses
fig = plt.figure(0)
plt.plot(history['loss'], label='MAE (training data Unet)')
plt.plot(history['val_loss'], label='MAE (validation data Unet)')
plt.plot(history_D['loss'], label='MAE (training data Dense)')
plt.plot(history_D['val_loss'], label='MAE (validation data Dense)')

plt.title('MAE for Losses')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# plot dice scores 
fig = plt.figure(2)
plt.plot(history['final_1_dsc'], label='MAE (training data Unet PR)')
plt.plot(history['val_final_1_dsc'], label='MAE (validation data Unet PR)')
plt.plot(history_D['final_1_dsc'], label='MAE (training data Dense PR)')
plt.plot(history_D['val_final_1_dsc'], label='MAE (validation data Dense PR)')

plt.plot(history['final_2_dsc'], label='MAE (training data Unet CZ)')
plt.plot(history['val_final_2_dsc'], label='MAE (validation data Unet CZ)')
plt.plot(history_D['final_2_dsc'], label='MAE (training data Dense CZ)')
plt.plot(history_D['val_final_2_dsc'], label='MAE (validation data Dense CZ)')

plt.plot(history['final_3_dsc'], label='MAE (training data Unet PZ)')
plt.plot(history['val_final_3_dsc'], label='MAE (validation data Unet PZ)')
plt.plot(history_D['final_3_dsc'], label='MAE (training data Dense PZ)')
plt.plot(history_D['val_final_3_dsc'], label='MAE (validation data Dense PZ)')

plt.title('MAE for dice scores')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()