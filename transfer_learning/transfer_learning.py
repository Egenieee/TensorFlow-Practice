# 이 튜토리얼에서는 사전 훈련된 네트워크에서 전이 학습을 사용하여 고양이와 개의 이미지를 분류하는 방법을 배우게 된다.
# 사전 훈련된 모델은 이전에 대규모 데이터셋에서 훈련되어 저장된 네트워크로, 일반적으로 대규모 이미지 분류 작업에서 훈련된 것이다.

# 사전 훈련된 모델을 사용자 정의하는 두 가지 방법을 시도한다.
# 1. 특성 추출 : 새 샘플에서 의미있는 특정을 추출하기 위해 이전 네트워크에서 학습한 표현을 사용한다.
# 사전 훈련된 모델 위에 처음부터 훈련한 새 분류자를 추가하기만 하면 이전에 데이터 세트로 학습한 특성 맵의 용도를 재사용할 수 있다.
# 전체 모델을 재 훈련 시킬 필요는 없다.
# 2. 미세 조정 : 고정된 기본 모델의 일부 최상위 층을 고정 해제하고 새로 추가된 분류기 층과 기본 모델의 마지막 층을 함께 훈련시킨다.
# 이를 통해 기본 모델에서 고차원 특징 표현을 "미세 조정" 하여 특정 작업에 보다 관련성이 있도록 할 수 있다.

# 1. 데이터 검사 및 이해
# 2. 입력 파이프 라인 빌드
# 3. 모델 작성
    # 사전 훈련된 기본 모델(또한 사전 훈련된 가중치)에 적재
    # 분류 레이어를 맨 위에 쌓기
# 4. 모델 훈련
# 5. 모델 평가

import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import io

import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

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

logdir = "logs/base_model/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
base_model_file_writer = tf.summary.create_file_writer(logdir)

# 데이터 전처리

# 데이터 다운로드

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

# 훈련용 데이터셋에서 처음 두개의 이미지 및 레이블을 보여준다.
class_names = train_dataset.class_names

training_dataset_sample = plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

with base_model_file_writer.as_default():
    tf.summary.image("training set example", plot_to_image(training_dataset_sample), step=0)

# 원본 데이터세트에는 테스트 세트가 포함되어있지 않으므로 테스트 세트를 생성한다.
# tf.data.experimental.cardinality를 사용하여 검증 세트에서 사용할 수 있는
# 데이터 배치 수를 확인한 다음 그 중 20%를 테스트 세트로 이동한다.

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

# 성능을 높이도록 데이터 세트 구성하기
# 버퍼링된 프리페치를 사용하여 I/O 차단 없이 디스크에서 이미지를 로드한다.
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# data augmentation 사용
# 큰 이미지 데이터 세트가 없는 경우, 회전 및 수평 귀집기와 같이 훈련 이미지에 무작위지만 사실적인 변환을 적용하여 샘플 다양성을 인위적으로 만드는 것이 좋다.
# 이것은 모델을 훈련 데이터의 다양한 측면에 노출시키고 오버피팅을 줄이는 데 도움이 된다.

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# 같은 이미지에 이 레이어를 반복해서 적용하고 결과를 확인해보자.
augmented_image_sample = plt.figure(figsize=(10, 10))
for image, _ in train_dataset.take(1):
    augmented_image_sample = plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
#plt.show()

with base_model_file_writer.as_default():
    tf.summary.image("augmented image sample", plot_to_image(augmented_image_sample), step=0)

# 픽셀 값 재조정
# 기본 모델로 사용할 모델은 [-1, 1] 의 픽셀 값을 예상하므로 크기를 재조정해야한다.
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# 사전 훈련된 컨볼루션 네트워크로부터 기본 모델 생성하기
# Google에서 개발한 MobileNetV2 모델로부터 기본 모델을 생성한다.
# 이 모델을 1.4M 이미지와 1000개의 클래스로 구성된 대규모 데이터셋인 ImageNet 데이터 셋을 사용해 사전 훈련된 모델이다.

# 먼저 ImageNet 으로 훈련된 가중치가 저장된 MobileNet V2 모델을 인스턴스화하자.
# include_top = False로 지정하면 맨 위에 분류 층이 포함되지 않은 네트워크를 로드하므로 특징 추출에 이상적이다.

# Create the base model from the pre_trained model MobileNetV2
IMG_SHAPE = IMG_SIZE + (3, )
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# 이 특징 추출기는 각 160 * 160 * 3 이미지를 5 * 5 * 1280개의 특징 블록으로 변환한다.

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape) # (32, 5, 5, 1280)

# 특징 추출
# 이 단계에서는 이전 단게에서 작성된 convolution 베이스 모델을 동결하고 특징 추출기로 사용한다.
# 또한 그 위에 분류기를 추가하고 최상위 분류기를 훈련시킨다.

# convolution base model 고정하기
# 모델을 컴파일하고 훈련하기 전에 컨볼루션 기반을 고정하는 것이 중요하다.
# 동결을 주어진 레이어의 가중치가 훈련 중에 업데이트 되는 것을 방지한다.
# MobileNet V2애는 많은 레이어가 있으므로 전체 모델의 trainable 플래그를 False로 설정하면 레이어가 모두 동결된다.

base_model.trainable = False

# 기본 모델 아키텍쳐를 살펴보자
base_model.summary()

# 분류 층을 맨 위에 추가하기
# 특징 블록에서 예측을 생성하기 위해 GlobalAveragePooling2D 레이어를 사용하여 특성을
# 이미지당 하나의 1280요소 벡터로 변환하여 5 * 5 공간 위치에 대한 평균을 구한다.
# 이미지 하나를 평균내서 한 픽셀로 만든다. 채널은 그대로

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape) # (32, 1280)

# tf.keras.layers.Dense 레이어를 사용하여 특성을 이미지당 단일 예측으로 변환한다.
# 이 예측은 원시 예측 값으로 취급되므로 활성화 함수가 필요하지 않다. 양수는 클래스 1을 예측하고 음수는 클래스 0을 예측한다.
# 참고 강아지가 1 고양이가 0

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape) # (32, 1)

# Keras Functional API를 사용하여 데이터 증강, 크기 조정, base_model 및 특성 추출기 레이어를 함께 연결하여 모델을 구축하자.
# 모델에 BatchNormalization 레이어가 포함되어 있으므로 training=False를 사용하자.

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# 모델 컴파일
# 학습 하기 전에 모델을 컴파일 해야한다. 두 개의 클래스가 있으므로 모델이 선형 출력을 제공하므로
# from_logits = True와 함께 이진 교차 엔트로피 손실을 이용하자.

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 모델 훈련
# 10 epoch 만큼 훈련한 후, 검증 세트에서 ~94%의 정확도를 볼 수 있다.

initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss : {:.2f}".format(loss0))
print("initial accuracy : {:.2f}".format(accuracy0))


model.fit(train_dataset,
          epochs=initial_epochs,
          validation_data=validation_dataset,
          callbacks=[tensorboard_callback])

# 이제 이 모델을 사용하여 이미지가 고양이인지 개인지 예측해보자.

image_batch, label_batch = test_dataset.as_numpy_iterator().next()
base_model_predictions = model.predict_on_batch(image_batch).flatten()

base_model_predictions = tf.nn.sigmoid(base_model_predictions)
base_model_predictions = tf.where(base_model_predictions < 0.5, 0, 1)

print("Base Model Predictions : {}".format(base_model_predictions.numpy()))
print("Labels : {}".format(label_batch))

text_predictions = "Base Model Predictions : {}".format(base_model_predictions.numpy())
text_labels = "Labels : {}".format(label_batch)


count = 0
for i in range(len(label_batch)):
    if base_model_predictions.numpy()[i] != label_batch[i]:
        count += 1

text_negative_rate = f"image {len(label_batch)}개 중 {count}개 오답. 오답률은 {(count / len(label_batch)) * 100}% 입니다."

text_result = f"""
# Transfer learning
## pre-trained model을 이용한 cats/dogs 분류 모델 학습
### fine tuning 하지 않은 결과
***
{text_predictions}
\n
{text_labels}
***
{text_negative_rate}
***
"""

with base_model_file_writer.as_default():
    tf.summary.text("base_model_evaluate_result", text_result, step=0)

base_model_predictions_sample = plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(class_names[base_model_predictions[i]] + "({})".format(class_names[label_batch[i]]))
    plt.axis("off")

with base_model_file_writer.as_default():
    tf.summary.image("base_model_prediction_result", plot_to_image(base_model_predictions_sample), step=0)


# Fine tuning
# 성능을 더욱 향상 시키는 한 가지 방법은 추가한 분류기의 훈련과 함께 사전 훈련된 모델의 최상위 레이어 가중치를 훈련하는 것이다.
# 훈련을 통해 가중치는 일반적인 특징 맵에서 개별 데이터 셋과 관련된 특징으로 조정된다.

# 최상위 층 고정 해제하기
# base_model을 고정 해제하고 맨 아래 층을 훈련할 수 없도록 설정하자. 그런 다음 모델을 다시 컴파일하고(변경 사항을 적용하기 위해서)
# 훈련을 다시 시작해야 한다.

logdir = "logs/fine_tune_model/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
fine_tuned_model_file_writer = tf.summary.create_file_writer(logdir)

base_model.trainable = True

# base model에 몇개의 레이어가 있는지 보자
print("Number of layers in the base model : {}".format(len(base_model.layers)))

fine_fune_at = 100

for layer in base_model.layers[:fine_fune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()

# 모델 훈련 계속하기
# 이미 수렴 상태로 훈련된 경우에, 이 단게는 정확도를 몇 퍼센트 포인트 향상시킨다.
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

model.fit(train_dataset,
          epochs=total_epochs,
          validation_data=validation_dataset,
          callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(test_dataset)
print("Test accuracy : {}".format(accuracy))

# 이제 이 모델을 사용하여 이미지가 고양이인지 개인지 예측해보자.

image_batch, label_batch = test_dataset.as_numpy_iterator().next()
fine_tuned_model_predictions = model.predict_on_batch(image_batch).flatten()

fine_tuned_model_predictions = tf.nn.sigmoid(fine_tuned_model_predictions)
fine_tuned_model_predictions = tf.where(fine_tuned_model_predictions < 0.5, 0, 1)

print("Fine Tuned Predictions : {}".format(fine_tuned_model_predictions.numpy()))
print("Labels : {}".format(label_batch))

text_predictions = "Fine Tuned Predictions : {}".format(fine_tuned_model_predictions.numpy())
text_labels = "Labels : {}".format(label_batch)


count = 0
for i in range(len(label_batch)):
    if fine_tuned_model_predictions.numpy()[i] != label_batch[i]:
        count += 1

text_negative_rate = f"image {len(label_batch)}개 중 {count}개 오답. 오답률은 {(count / len(label_batch)) * 100}% 입니다."

text_result = f"""
# Transfer learning
## pre-trained model을 이용한 cats/dogs 분류 모델 학습
### fine tuning 한 결과
***
{text_predictions}
\n
{text_labels}
***
{text_negative_rate}
***
"""

with fine_tuned_model_file_writer.as_default():
    tf.summary.text("fine_tuned_model_evaluate_result", text_result, step=0)

plt.cla()
plt.clf()
plt.close()
fine_tuned_model_predictions_sample = plt.figure(figsize=(10, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(class_names[fine_tuned_model_predictions[i]] + "({})".format(class_names[label_batch[i]]))
    plt.axis("off")

# img_ret = plot_to_image(prediction_sample_image)
# print(img_ret.__sizeof__())
# print(type(img_ret))

with fine_tuned_model_file_writer.as_default():
    tf.summary.image("fine_tuned_model_prediction_result", plot_to_image(fine_tuned_model_predictions_sample), step=0)
