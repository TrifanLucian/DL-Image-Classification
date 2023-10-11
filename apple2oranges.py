import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

def dnn_model():
  new_model = tf.keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    # layers.InputLayer((img_height, img_width, 3)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
  ])
  return compile_model(new_model)

def cnn_model():
  new_model = tf.keras.Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
  ])
  return compile_model(new_model)

def compile_model(new_model):
  new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  print(new_model.summary())
  return new_model

test_dir = r'C:\Users\Lucian\Desktop\Masterat CTI\Deep Learn-ing\dataset_applesandoranges\valid'
data_dir = r'C:\Users\Lucian\Desktop\Masterat CTI\Deep Learn-ing\dataset_applesandoranges\train'

epochs = 30
img_height=80
img_width=80
batch_size=20

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  shuffle=True,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
print(len(train_ds))
print(len(test_ds))

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

labels_batch

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

model = cnn_model()

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model1.png')


epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

test_images = []
test_labels = []
for image, label in test_ds.take(-1):
  test_images.extend(image)
  test_labels.extend(label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print(test_images.shape)
print(test_labels.shape)

print(model.summary())

test_loss, test_acc = model.evaluate(test_images,  test_labels, ver-bose=2)
print(test_acc)
