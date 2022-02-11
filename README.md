# TensorFlow-Practice
### Deep Learning Framework TensorFlow 실습

* Neural Network
  * cifar10 dataset
  * fashion mnist dataset
  * mnist dataset
* Convolution Neural Network
  * cifar10 dataset
  * fashion mnist dataset
  * mnist dataset
* Transfer Learning
* DCGAN
  * mnist dataset
* TensorBoard Tutorial

## Using Cifar10 dataset
#### - cifar10 dataset
![009](./img/009.png)

#### - model task
![029](./img/029.png)

모델은 테스트 이미지가 들어왔을 때 어떤 이미지(레이블)인지 예측해야 한다.

#### - model define
![039](./img/039.png)

* 구현 file
> cnn_cifar10.py   
> nn_cifar10.py

## Using Fashion mnist dataset
#### - fashion mnist dataset
![032](./img/032.png)

#### - model task
![033](./img/033.png)

모델은 테스트 이미지가 들어왔을 때 어떤 이미지(레이블)인지 예측해야 한다.

#### - model define
![041](./img/041.png)

* 구현 file
> cnn_fashion_mnist.py   
> nn_fashion_mnist.py

## Using Mnist dataset
#### - mnist dataset
![043](./img/043.png)

#### - model task
![044](./img/044.png)

모델은 테스트 이미지가 들어왔을 때 어떤 이미지(레이블)인지 예측해야 한다.

#### - model define
![045](./img/045.png)

* 구현 file
> cnn_mnist.py   
> nn_mnist.py

## Transfer Learning
#### - model task
![051](./img/051.png)

IMAGENET dataset으로 pre-trained된 MobileNetV2를 이용하여 개/고양이를 분류한다. 

#### - data augmentation
![052](./img/052.png)
![052](./img/052.png)

#### - base model vs fine tuned model
![055](./img/055.png)

base model과 fine tuned model 두 모델의 성능 차이를 기대한다.    
fine tuned을 함으로써 내가 가지고 있는 데이터 셋에 관련되게 모델이 조정되게끔 한다.

## DCGAN
![059](./img/059.png)

GAN은 generator model과 discriminator model을 함께 학습시킨다.

#### - generator model
![062](./img/062.png)

generator model은 random noise vector를 입력받아 upsampling을 통해 원하는 사이즈의 이미지를 얻는다. 

#### - discriminator model
![065](./img/065.png)

discriminator model은 CNN 기반의 모델이며, generator가 생성한 이미지가 가짜인지(0), 진짜인지(1) 예측한 값을 출력한다. 

#### - 200 epochs result
![dcgan_200epochs](./img/dcgan_200_epochs_1.gif)

200 epoch을 돌린 결과물이다. 

***
#### 참고
* TensorFlow Tutorial : <https://www.tensorflow.org/tutorials?hl=ko>   
