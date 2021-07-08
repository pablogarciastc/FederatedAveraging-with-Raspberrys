import os
import numpy as np
current_dir = os.getcwd()
# Import mnist data stored in the following path: current directory -> mnist.npz
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'/mnist.npz')

split_number = 13

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))

unique, counts = np.unique(Y_test, return_counts=True)
print(dict(zip(unique, counts)))


splitted_X_train = np.array_split(X_train,split_number)
splitted_Y_train = np.array_split(Y_train,split_number)
splitted_X_test = np.array_split(X_test,split_number)
splitted_Y_test = np.array_split(Y_test,split_number)


for i in range(0,split_number):
    np.savez_compressed(current_dir+'\data\\tmp15\mnist' + str(i+1), x_test=X_test, y_test=Y_test, x_train = splitted_X_train[i], y_train = splitted_Y_train[i])


