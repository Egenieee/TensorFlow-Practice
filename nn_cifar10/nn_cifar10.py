import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    return it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # ADD the batch dimension
    image = tf.expand_dims(image, 0)
    return image

# Cifar10 데이터셋을 로드하여 준비
cifar10 = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images = train_images.reshape((50000, 32, 32, 3))
test_images = test_images.reshape((10000, 32, 32, 3))

train_labels = train_labels.reshape(50000,)
test_labels = test_labels.reshape(10000,)

# 샘플 값을 정수에서 부동소수로 변환 0과 1 사이의 값으로 변환
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

logdir = "logs/nn_cifar10/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer = tf.summary.create_file_writer(logdir)

# training set 확인
training_set = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

with file_writer.as_default():
    tf.summary.image("training_set", plot_to_image(training_set), step=0)

# test set 확인
test_set = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])

with file_writer.as_default():
    tf.summary.image("test_set", plot_to_image(test_set), step=0)

# 층을 차례대로 쌓아 모델을 만든다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 훈련에 사용할 최적화 알고리즘과 loss 함수를 선택한다.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델을 training시킨다.
result = model.fit(train_images, train_labels, epochs=10, callbacks=[tensorboard_callback])

# training된 모델을 평가한다.
model.evaluate(test_images, test_labels, verbose=2)

# 모델 예측 확인

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_labels, img):
    true_labels, img = true_labels[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_labels:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f} % ({})".format(class_names[predicted_label], 100*np.max(predictions_array),
                                          class_names[true_labels]), color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

for i in range(10):
    sample = plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)

    with file_writer.as_default():
        tf.summary.image("samples", plot_to_image(sample), step=i)

# 몇개의 이미지의 예측을 한번에 출력해보자

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
samples = plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()

with file_writer.as_default():
    tf.summary.image("samples_15", plot_to_image(samples), step=0)