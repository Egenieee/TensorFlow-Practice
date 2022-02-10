import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

# dataset download
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 픽셀 값을 0 ~ 1 사이로 정규화한다.
train_images, test_images = train_images / 255.0, test_images / 255.0

logdir = "logs/cnn_fashion_mnist/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
file_writer = tf.summary.create_file_writer(logdir)

# training set에서 처음 25개의 이미지와 그 아래 클래스 이름을 출력해보자.
# 데이터 포맷이 올바른지 확인하고 네트워크 구성과 훈련할 준비를 마치자!

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


# Conv2D와 MaxPooling2D 층을 쌓자.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# 모델을 완성하려면 마지막 층에 dense층을 추가해야 한다. dense층은 벡터를 입력으로 받는데, 현재는 3D 텐서니까 1D로 펼쳐준다.
# MNIST 데이터는 10개의 클래스가 있으므로 마지막에 Dense 층에 10개의 출력과 softmax 활성화함수를 사용한다.
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 모델에 추가적인 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
result = model.fit(train_images, train_labels, epochs=10, callbacks=[tensorboard_callback])

# 모델 평가
model.evaluate(test_images, test_labels, verbose=2)

# 모델이 예측한거 확인해보기

# probability_model = tf.keras.Sequential([model,
#                                          tf.keras.layers.Softmax()])

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