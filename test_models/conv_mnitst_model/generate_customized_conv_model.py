import tensorflow as tf
import numpy as np
import keras
from keras import layers

num_classes = 10
input_shape = (28, 28, 1)

# def preprocess_input(input):
#   res = layers.Multiply(input, 1/255)
#   res = layers.Reshape(res)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save tflite model model.
with open('custom_conv_mnist_model.tflite', 'wb') as f:
  f.write(tflite_model)