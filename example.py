import numpy as np
import math

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from numpynet.nn.model import Model
from numpynet.nn.layers import Linear
from numpynet.nn.activations import ReLU
from numpynet.nn.losses import MSE

# Create the model
loss = MSE()
model = Model(loss)

model.add_layer(Linear(784, 256))
model.add_layer(ReLU())
model.add_layer(Linear(256, 256))
model.add_layer(ReLU())
model.add_layer(Linear(256, 10))

# Load mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.reshape(train_images, (60_000, 784)) / 255.0
test_images = np.reshape(test_images, (10_000, 784)) / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Training settings
epochs = 10
batch_size = 16
learning_rate = 0.01

n_batches = math.ceil(60_000 / batch_size)

# Training loop
for e in range(epochs):
  print(f"Epoch: {e + 1}/{epochs}")

  total_loss = 0
  steps = 0

  for b in range(n_batches):
    start = b * batch_size
    end = min(start + batch_size, 60_000)

    X = train_images[start:end]
    t = train_labels[start:end]

    # Forward through the model
    output = model(X)

    # Compute loss 
    loss = model.loss(output, t)
    
    # Backward propagate the loss, update parameters and reset gradients
    model.backward()
    model.update_step(learning_rate)
    model.zero_grad()

    total_loss += loss
    steps += 1

    print(f"{b + 1}/{n_batches} - loss: {round(total_loss / steps, 4)}", end="\r")
  
  # Evaluate the model's performance on the test set
  val_acc = 0
  output = model(test_images, grad=False)
  val_loss = model.loss(output, test_labels, grad=False)

  for i in range(10_000):
    val_acc += np.argmax(output[i]) == np.argmax(test_labels[i])
  val_acc /= 10_000

  print(f"{b + 1}/{n_batches} - loss: {round(total_loss / steps, 4)} - val_loss: {round(val_loss, 4)} - val_accuracy: {round(val_acc, 4)}")
