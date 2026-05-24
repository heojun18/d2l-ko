```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Network in Network (NiN)
:label:`sec_nin`

LeNet, AlexNet, VGG는 모두 공통의 설계 패턴을 공유합니다:
합성곱과 풀링 층의 시퀀스를 통해
*공간* 구조를 활용하여 특성을 추출하고
완전 연결 층을 통해 표현을 후처리합니다.
AlexNet과 VGG가 LeNet에 가한 개선은 주로
이러한 후자의 네트워크가 이 두 모듈을 어떻게 넓히고 깊게 만드는지에 있습니다.

이 설계는 두 가지 주요 도전 과제를 제기합니다.
첫째, 아키텍처 끝부분의 완전 연결 층은
엄청난 수의 파라미터를 소비합니다. 예를 들어,
VGG-11과 같은 단순한 모델조차도 단정밀도(FP32)에서 거의 400MB의 RAM을 차지하는
괴물 같은 행렬을 필요로 합니다. 이는 특히 모바일 및 임베디드 장치에서
계산에 상당한 장애입니다. 결국, 하이엔드 휴대폰조차도 8GB 이상의 RAM을 갖추지 않습니다. VGG가 발명되었을 때, 이는 한 자릿수 더 적었습니다(iPhone 4S는 512MB였습니다). 이와 같이, 이미지 분류기에 메모리의 대부분을 소비하는 것을 정당화하기는 어려웠을 것입니다.

둘째, 비선형성의 정도를 증가시키기 위해 네트워크의 더 일찍
완전 연결 층을 추가하는 것도 마찬가지로 불가능합니다: 그렇게 하면 공간 구조를
파괴하고 잠재적으로 훨씬 더 많은 메모리를 필요로 합니다.

*network in network*(*NiN*) 블록 :cite:`Lin.Chen.Yan.2013`은 하나의 간단한 전략으로
두 문제를 모두 해결할 수 있는 대안을 제공합니다.
이들은 매우 간단한 통찰을 기반으로 제안되었습니다: (i) 채널 활성화 전반에 걸쳐 국소적
비선형성을 추가하기 위해 $1 \times 1$ 합성곱을 사용하고 (ii) 마지막 표현 층의
모든 위치에 걸쳐 통합하기 위해 전역 평균 풀링을 사용합니다. 추가된 비선형성이 없다면 전역 평균 풀링이
효과적이지 않을 것이라는 점을 참고하세요. 이를 자세히 알아봅시다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## (**NiN 블록**)

:numref:`subsec_1x1`을 떠올려 보세요. 그 안에서 합성곱 층의 입력과 출력은
예제, 채널, 높이, 너비에 해당하는 축을 가진 4차원 텐서로 구성된다고 했습니다.
또한 완전 연결 층의 입력과 출력은
일반적으로 예제와 특성에 해당하는 2차원 텐서임을 떠올려 보세요.
NiN 뒤의 아이디어는 각 픽셀 위치에서(각 높이와 너비에 대해)
완전 연결 층을 적용하는 것입니다.
결과로 나오는 $1 \times 1$ 합성곱은
각 픽셀 위치에서 독립적으로 작동하는
완전 연결 층으로 생각할 수 있습니다.

:numref:`fig_nin`은 VGG와 NiN, 그리고 그들의 블록 간의 주요 구조적
차이를 보여줍니다.
NiN 블록의 차이(초기 합성곱 다음에 $1 \times 1$ 합성곱이 오고, VGG는 $3 \times 3$ 합성곱을 유지함)와 더 이상 거대한 완전 연결 층이 필요하지 않은 끝부분의 차이 모두에 주목하세요.

![VGG와 NiN, 그리고 그들의 블록의 아키텍처 비교.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])
```

## [**NiN 모델**]

NiN은 AlexNet과 동일한 초기 합성곱 크기를 사용합니다(이는 AlexNet 직후에 제안되었습니다).
커널 크기는 각각 $11\times 11$, $5\times 5$, $3\times 3$이며,
출력 채널의 수는 AlexNet의 것과 일치합니다. 각 NiN 블록 뒤에는
스트라이드가 2이고 윈도우 모양이 $3\times 3$인 최대 풀링 층이 옵니다.

NiN과 AlexNet 및 VGG 사이의 두 번째 중요한 차이점은
NiN이 완전 연결 층을 완전히 피한다는 것입니다.
대신, NiN은 레이블 클래스의 수와 같은 수의 출력 채널을 가진 NiN 블록을 사용하고, 그 뒤에 *전역* 평균 풀링 층이 오며,
로짓의 벡터를 산출합니다.
이 설계는 잠재적인 학습 시간 증가의 대가로, 필요한 모델 파라미터의 수를 크게 줄입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab jax
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # global avg pooling
            lambda x: x.reshape((x.shape[0], -1))  # flatten
        ])
```

[**각 블록의 출력 모양을 보기 위해**] 데이터 예제를 만듭니다.

```{.python .input}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
NiN().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
NiN(training=False).layer_summary((1, 224, 224, 1))
```

## [**학습**]

이전과 마찬가지로 AlexNet과 VGG에 사용한 것과 동일한 옵티마이저를 사용하여 Fashion-MNIST를 사용해
모델을 학습합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## 요약

NiN은 AlexNet과 VGG보다 극적으로 적은 파라미터를 가지고 있습니다. 이는 주로 거대한 완전 연결 층이 필요하지 않다는 사실에서 비롯됩니다. 대신, 네트워크 본체의 마지막 단계 이후 모든 이미지 위치에 걸쳐 통합하기 위해 전역 평균 풀링을 사용합니다. 이는 비싼 (학습된) 축소 연산의 필요성을 없애고 간단한 평균으로 대체합니다. 당시 연구자들을 놀라게 한 것은 이 평균화 연산이 정확도를 해치지 않는다는 사실이었습니다. 저해상도 표현(많은 채널이 있는)에 걸쳐 평균화하는 것은 또한 네트워크가 처리할 수 있는 이동 불변성의 양에 추가됨에 주목하세요.

넓은 커널을 가진 더 적은 합성곱을 선택하고 이를 $1 \times 1$ 합성곱으로 대체하는 것은 더 적은 파라미터에 대한 탐구를 추가로 돕습니다. 이는 주어진 어떤 위치 내에서도 채널 간 상당한 양의 비선형성을 제공할 수 있습니다. $1 \times 1$ 합성곱과 전역 평균 풀링 모두 후속 CNN 설계에 상당한 영향을 미쳤습니다.

## 연습문제

1. NiN 블록당 두 개의 $1\times 1$ 합성곱 층이 있는 이유는 무엇인가요? 그 수를 세 개로 늘리세요. 그 수를 하나로 줄이세요. 무엇이 변하나요?
1. $1 \times 1$ 합성곱을 $3 \times 3$ 합성곱으로 대체하면 무엇이 변하나요?
1. 전역 평균 풀링을 완전 연결 층으로 대체하면 어떻게 되나요(속도, 정확도, 파라미터 수)?
1. NiN의 자원 사용량을 계산하세요.
    1. 파라미터의 수는 얼마인가요?
    1. 계산량은 얼마인가요?
    1. 학습 중에 필요한 메모리 양은 얼마인가요?
    1. 예측 중에 필요한 메모리 양은 얼마인가요?
1. $384 \times 5 \times 5$ 표현을 한 번에 $10 \times 5 \times 5$ 표현으로 축소하는 것의 가능한 문제는 무엇인가요?
1. VGG-11, VGG-16, VGG-19로 이어진 VGG의 구조적 설계 결정을 사용하여 NiN과 유사한 네트워크 패밀리를 설계하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18003)
:end_tab:
