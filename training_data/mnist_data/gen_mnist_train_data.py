import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras
import numpy as np
import pathlib
from keras.utils import np_utils

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


def generate_train_data(batches_number):
    # Load the data and split it between train and test sets
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)

    current_dir = pathlib.Path(__file__).parent
    in_data_path = current_dir/'mnist.input.bin'
    out_data_path = current_dir/'mnist.output.bin'
    with open(in_data_path, 'wb') as in_stream:
        in_stream.write(x_train[0:batches_number,:,:,:].tobytes())
    with open(out_data_path, 'wb') as out_stream:
        out_stream.write(y_train[0:batches_number].tobytes())
    return in_data_path,out_data_path

def main(batches_number):
    generate_train_data(batches_number)

if __name__ == "__main__":
    import sys

    if len(sys.argv)!= 2:
        print("Usage: python main.py <batches_number>")
        sys.exit(1)

    try:
        batches_number = int(sys.argv[1])
    except ValueError:
        print("Error: Batch size must be an integer")
        sys.exit(1)

    main(batches_number)
