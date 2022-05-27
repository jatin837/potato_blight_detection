import config
import tensorflow as tf
from tensorflow.keras import models, layers, Sequential
import matplotlib.pyplot as plt
from utils import get_train_test_val_split


# Load dataset from disk to memory as tf.data.Dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    directory=config.DATA_DIR,
    labels='inferred',
    label_mode='int',
    shuffle=True,
    image_size=config.IMAGE_SIZE,
    batch_size=config.BATCH_SIZE
)

# split the dataset for training, testing and validation purposes
train_data, test_data, val_data = get_train_test_val_split(
    0.8,
    0.1,
    0.1,
    dataset
)

# Enable cache and prefetching
train_data = train_data.cache().shuffle(1000).prefetch(
    buffer_size=tf.data.AUTOTUNE
)
test_data = test_data.cache().shuffle(1000).prefetch(
    buffer_size=tf.data.AUTOTUNE
)
val_data = val_data.cache().shuffle(1000).prefetch(
    buffer_size=tf.data.AUTOTUNE
)
# END Enable cache and prefetching


# Preprocessing layer (Resizing(to our image size) and Rescaling the image)
preprocess_layer = Sequential([
    layers.experimental.preprocessing.Resizing(*config.IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

# Data Augmentation layer (flip and rotation)
data_augmentation_layer = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
model = Sequential([
    preprocess_layer,
    data_augmentation_layer,
    layers.Conv2D(
        32, (3, 3),
        activation='relu',
        input_shape=(
            config.BATCH_SIZE,
            config.IMAGE_SIZE[0],
            config.IMAGE_SIZE[1],
            config.CHANNELS)
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.build(input_shape=(
    config.BATCH_SIZE,
    config.IMAGE_SIZE[0],
    config.IMAGE_SIZE[0],
    config.CHANNELS
))
