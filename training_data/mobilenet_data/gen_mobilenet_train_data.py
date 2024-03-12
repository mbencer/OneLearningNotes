import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pathlib

tf.get_logger().setLevel('ERROR')
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

def generate_train_data(batches_number):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)

    current_dir = pathlib.Path(__file__).parent
    in_data_path = current_dir/'cats_and_dogs.input.bin'
    out_data_path = current_dir/'cats_and_dogs.output.bin'
    with open(in_data_path, 'wb') as in_stream:
        with open(out_data_path, 'wb') as out_stream:
            for batch in train_dataset.take(batches_number):
                in_stream.write(batch[0].numpy().tobytes())
                out_stream.write(batch[1].numpy().tobytes())
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
