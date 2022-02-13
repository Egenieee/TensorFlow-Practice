import imageio
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import  PIL
from tensorflow.keras import layers
import time
import io
from datetime import datetime

from IPython import display

# generator와 discriminator를 훈련시키기 위해 MNIST 데이터셋을 사용한다.
# generator는 손글씨 숫자를 닮은 숫자들을 생성한다.
(train_images, train_labels), (_,_) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # 이미지를 [-1,1]로 정규화한다. norm zero centered

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# 데이터 배치를 만들고 섞는다.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 모델 만들기

# 생성자는 시드값(seed;랜덤한 잡음)으로부터 이미지를 생성하기 위해 업샘플링 층을 이용한다.
# 처음 Dense층은 이 시드값을 인풋으로 받는다. 그 다음 원하는 사이즈 28 * 28 * 1의 이미지(mnist의 사이즈)가 나오도록 업샘플링을 여러번 한다.

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # 주목 : 배치사이즈로 None이 주어진다.

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


# plot 이미지를 PNG 이미지로 변환해주는 함수
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


# discriminator 모델 만들기
# discriminator는 CNN 기반의 이미지 분류기이다.

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # generator가 만든 이미지가 진짜인지 가짜인지 판단 1은 진짜, 0은 가짜

    return model

logdir = 'logs/random_noise/'
file_writer = tf.summary.create_file_writer(logdir)

# 아직 훈련이 되지 않은 생성자를 이용해 이미지를 생성해보자.
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# 아직 훈련이 되지 않은 감별자를 사용해서 생성된 이미지가 진짜인지 가까인지 판별해보자.
# discriminator 모델은 진짜 이미지에는 양수의 값을 가짜 이미지에는 음수의 값을 출력하도록 훈련된다.

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
decision_text = ""
print(decision)
if decision < 0:
    decision_text = "decision : this image is Fake"
    print("this image is Fake")
else :
    decision_text = "decision : this image is Real"
    print("this image is Real")

generated_image_plot = plt.figure(figsize=(8,8))
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.title(decision_text)

# with file_writer.as_default():
#     tf.summary.image("random_noise", plot_to_image(generated_image_plot), step=0)


# loss function과 optimizer 정의

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# discriminator loss function
# 이 메서드는 감별자가 가짜 이미지에서 얼마나 진짜 이미지를 잘 판별하는지 수치화한다. 진짜 이미지에 대한 감별자의 에측과
# 1로 이루어진 행렬을 비교하고, 가짜 이미지에 대한 감별자의 예측과 0으로 이루어진 행렬을 비교한다.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# generator loss function
# 생성자의 손실 함수는 감별자를 얼마나 잘 속였는지에 대해 수치화한다.
# 직관적으로 생성자가 원활히 수행되고 있다면, 감별자는 가짜 이미지를 진짜로 분류할 것이다.
# 여기서 우리는 생성된 이미지에 대한 감별자의 결정을 1로 이루어진 행렬과 비교할 것이다.

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# discriminator와 generator는 따로 훈련되기 때문에, 각각의 optimizer는 다르다.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 체크포인트 저장
# 오랫동안 진행되는 훈련이 방해되는 경우에 유용하게 쓰일 수 있는 모델의 저장방법과 복구방법
checkpoint_dir = './training_checkpoints'
checkpoint_prefic = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# 훈련 루프 정의하기
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16

# 이 시드는 시간이 지나도 재활용
# GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문이다.
seed = tf.random.normal([num_examples_to_generate, noise_dim])
# 랜덤을

# 훈련 루프는 생성자가 입력으로 램덤 시드는 받는 것으로부터 시작된다.
# 그 시드값을 사용하여 이미지를 생성한다.
# discriminator를 사용하여 훈련세트에서 가지고 온 진짜 이미지와 생성자가 만들어낸 가짜이미지를 분류한다.
# 각 모델의 손실을 계산하고, gradient를 사용해 생성자와 감별자를 업데이트 한다.

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # GIF를 위한 이미지를 바로 생성한다.
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if epoch % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefic)

        print(f"Time for epoch {epoch + 1} is {time.time()-start} sec")

    # 마지막 에폭이 끝난 후 생성한다.
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

figures = []

def generate_and_save_images(model, epoch, test_input):
    # training 이 False로 맞춰진 것에 주목하자
    # 이렇게 하면 모든 층들이 추론 모드로 실행된다.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    figures.append(plot_to_image(fig))


    # plt.show()

# 위에 정의된 train() 메서드를 생성자와 감별자를 동시에 훈련하기 위해 호출한다.

train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

logdir = 'logs/100epochs/fashion_mnist'
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
    for i, figure in enumerate(figures):
        tf.summary.image("100epochs result", figure, step=i)

# GIF 생성
# epoch 숫자를 사용하여 하나의 이미지를 보여준다.
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan_100_epochs_fashion_mnist.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)