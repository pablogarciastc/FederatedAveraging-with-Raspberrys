
import tensorflow as tf
import numpy as np

#Load MNIST data
data = np.load('data/tmp10/mnist1.npz')

print('X_train: ' + str(data['x_train'].shape))
print('Y_train: ' + str(data['y_train'].shape))
print('X_test:  '  + str(data['x_test'].shape))
print('Y_test:  '  + str(data['y_test'].shape))

x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']


# Preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add one domention to make 3D images
x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

# Track the data type
dataType, dataShape = x_train.dtype, x_train.shape
print(f"Data type and shape x_train: {dataType} {dataShape}")
labelType, labelShape = y_train.dtype, y_train.shape
print(f"Data type and shape y_train: {labelType} {labelShape}")



"""## Training"""
##Model BUILDING
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Model training
acc_results_list = []
loss_results_list = []

for i in range(50):
  model.fit(x_train, y_train, batch_size=8,
              epochs=1, verbose=0, validation_split=0.2)
  model.save('SavedModel.h5')
  fo = open('SavedModel.h5', "rb")
  ByteArray = fo.read()
  arrayLength = len(ByteArray)
  print("longitud del array: ", arrayLength)
  eval_loss, eval_acc = model.evaluate(x_test,  y_test, verbose=1)
  print('Eval accuracy percentage: {:.2f}'.format(eval_acc * 100))
  print('Eval loss percentage: {:.2f}'.format(eval_loss * 100))

  acc_results_list.append(round(eval_acc*100,2))
  loss_results_list.append(round(eval_loss*100,2))

f = open("logs/server_acc.txt", "a")
f.truncate(0)
f.write(str(acc_results_list))
f.close()

f = open("logs/server_loss.txt", "a")
f.truncate(0)
f.write(str(loss_results_list))
f.close()



"""## Evaluation"""

