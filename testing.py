import numpy as np 
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import keras
import h5py
import sys
from chalnewala import get_activations
from kerassurgeon.operations import delete_channels
np.set_printoptions(threshold=sys.maxsize)

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data
def data_preprocess(x_data):
    return x_data/255

bd_model = keras.models.load_model("models/anonymous_bd_net.h5")
print(bd_model.loss)
print(bd_model.optimizer)
print(bd_model.optimizer.lr)
# print(bd_model.summary())

#Load the data
x_data, y_data = data_loader("data/clean_validation_data.h5")
x_data = data_preprocess(x_data)

x_poisoned, y_poisoned = data_loader("data/sunglasses_poisoned_data.h5")
x_posioned =data_preprocess(x_poisoned)

x_test, y_test = data_loader("data/clean_test_data.h5")
x_test = data_preprocess(x_test)

#Layer to prune is the last convolutional layer in the network.
layer_to_prune = bd_model.layers[7]
print(layer_to_prune)


'''#Fine-Tuning-------------------------------------------------------------------------'''
# bd_model.fit(x_data, y_data, epochs = 7)
# poisoned_label_p = np.argmax(bd_model.predict(x_poisoned), axis=1)
# poisoned_accu = np.mean(np.equal(poisoned_label_p, y_poisoned))*100
# print('Classification accuracy:', poisoned_accu)

# clean_label_p = np.argmax(bd_model.predict(x_test), axis=1)
# class_accu = np.mean(np.equal(clean_label_p, y_test))*100
# print('Classification accuracy:', class_accu)

'''Avg Activations calculation------------------------------------------------
The code below calculates the activations of each layer in the network and the takes the average of the activations of the channels in the last covolutional layer which we need to prune.
So, after taking the average, we get a 1x80 matrix because there are 80 channels in the conv_4 layer.'''

# cumul_activations = np.zeros((11547,4,3,80))
# for i in range(0, (np.shape(y_data)[0]-1)):
#     print(i)
#     keract_inputs = x_data[i:i+1]
#     keract_targets = y_data[i:i+1]
#     activations = get_activations(bd_model, keract_inputs)
#     cumul_activations[i] = cumul_activations[i] + activations['conv_4']

# avg_activations = np.mean(cumul_activations, axis=0)

# np.save('clean_activations_without_fine_tuning_bd2.npy', avg_activations)

'''The code below sorts the array according to the activations of each channel'''

avg_activations_channels = np.zeros((1,80))
clean = np.load('clean_activations_without_fine_tuning_bd2.npy')
for i in range(0, 80):
    avg_activations_channels[0][i] = np.mean(clean[:,:,i])
id_sort = np.argsort(avg_activations_channels, axis=-1)
print(id_sort[0])

'''#Pruning'''
'''To check how pruning affects the accuracy, I have created a loop which prunes the network based on the sorted activations array. So, first, the lest activated channel get prunes. 
In the second iteration, the 2 least activated channels get pruned, in the thord iteration, the three least activated channels get pruned and so on and so forth.'''

# clean_accuracies = np.zeros((1,80))
# poisoned_accuracies = np.zeros((1,80))
# for j in range(0,80):
#     new_model = delete_channels(bd_model, layer_to_prune, id_sort[0][:j])
#     # new_model.compile(loss ='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#     # poisoned_label_p = np.argmax(new_model.predict(x_poisoned), axis=1)
#     # poisoned_accu = np.mean(np.equal(poisoned_label_p, y_poisoned))*100
#     # print('Classification accuracy:', poisoned_accu)
#     # poisoned_accuracies[0][j] = poisoned_accu 

#     clean_label_p = np.argmax(new_model.predict(x_data), axis=1)
#     class_accu = np.mean(np.equal(clean_label_p, y_data))*100
#     print('Classification accuracy:', class_accu)
#     clean_accuracies[0][j] = class_accu

# # np.save('posioned_accuracy.npy', poisoned_accuracies)
# np.save('clean_accuracies_bd2.npy', clean_accuracies)

clean_accuracies_loaded = np.load('clean_accuracies_bd2.npy')
# poisoned_accuracies_loaded = np.load('posioned_accuracy.npy')
x = np.linspace(0,80,num=80)

plt.figure(1)
# plt.plot(x, np.transpose(poisoned_accuracies_loaded), label ='Posioned Accuracies')
plt.plot(x, np.transpose(clean_accuracies_loaded), label = 'Clean Accuracies')
plt.legend(loc='best')
plt.show()

# #Fine-Pruning------------------------------------------------

pruned_model = delete_channels(bd_model, layer_to_prune, id_sort[0][:73])
optimizer = keras.optimizers.Adadelta(lr=1)
pruned_model.compile(loss ='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(pruned_model.summary())

pruned_model.fit(x_data, y_data, epochs = 10)

# poisoned_label_p = np.argmax(pruned_model.predict(x_poisoned), axis=1)
# poisoned_accu = np.mean(np.equal(poisoned_label_p, y_poisoned))*100
# print('Classification accuracy:', poisoned_accu)

print(pruned_model.predict(x_test))

clean_label_p = np.argmax(pruned_model.predict(x_test), axis=1)
class_accu = np.mean(np.equal(clean_label_p, y_test))*100
print('Classification accuracy:', class_accu)





# plt.figure(1)
# for j in range(0,80):
#     plt.subplot(9,9,j+1)
#     plt.imshow(clean[:,:,j], cmap='gray')
# plt.show()
# # display_activations(avg_activations, cmap="gray", save=False)