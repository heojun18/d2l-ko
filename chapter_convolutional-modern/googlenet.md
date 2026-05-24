```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 다중 분기 네트워크 (GoogLeNet)
:label:`sec_googlenet`

2014년, *GoogLeNet*은
NiN :cite:`Lin.Chen.Yan.2013`, 반복 블록 :cite:`Simonyan.Zisserman.2014`,
그리고 합성곱 커널의 칵테일의 장점을 결합한 구조를 사용하여
ImageNet 챌린지에서 우승했습니다 :cite:`Szegedy.Liu.Jia.ea.2015`. 이는 또한 틀림없이 CNN에서 줄기(데이터 수집), 본체(데이터 처리), 머리(예측) 사이의 명확한 구분을 보여준 최초의 네트워크였습니다. 이 설계 패턴은 그 이후로 심층 네트워크 설계에서 지속되어 왔습니다: *줄기*는 이미지에서 작동하는 처음 두세 개의 합성곱으로 주어집니다. 이들은 기본 이미지에서 저수준 특성을 추출합니다. 이 다음에 합성곱 블록의 *본체*가 옵니다. 마지막으로, *머리*는 지금까지 얻은 특성을 당면한 필요한 분류, 세그멘테이션, 검출, 또는 추적 문제로 매핑합니다.

GoogLeNet의 핵심 기여는 네트워크 본체의 설계였습니다. 이는 합성곱 커널을 선택하는 문제를
독창적인 방식으로 해결했습니다. 다른 작업들이 $1 \times 1$에서 $11 \times 11$까지 어떤 합성곱이 가장 좋은지 식별하려 했던 반면, 이는 단순히 다중 분기 합성곱을 *연결*했습니다.
다음에서 저희는 GoogLeNet의 약간 단순화된 버전을 소개합니다: 원래 설계는 네트워크의 여러 층에 적용된 중간 손실 함수를 통해 학습을 안정화하기 위한 여러 트릭을 포함했습니다.
이들은 개선된 학습 알고리즘의 가용성으로 인해 더 이상 필요하지 않습니다.

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
from torch.nn import functional as F
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
from jax import numpy as jnp
import jax
```

## (**인셉션 블록**)

GoogLeNet의 기본 합성곱 블록은 *인셉션 블록*이라고 하며,
영화 *Inception*의 "we need to go deeper" 밈에서 유래합니다.

![인셉션 블록의 구조.](../img/inception.svg)
:label:`fig_inception`

:numref:`fig_inception`에 묘사된 바와 같이,
인셉션 블록은 네 개의 병렬 분기로 구성됩니다.
처음 세 분기는 서로 다른 공간 크기에서 정보를 추출하기 위해
$1\times 1$, $3\times 3$, $5\times 5$의 윈도우 크기를 가진
합성곱 층을 사용합니다.
중간 두 분기는 또한 모델의 복잡도를 줄이기 위해 입력에 $1\times 1$ 합성곱을 추가하여
채널의 수를 줄입니다.
네 번째 분기는 $3\times 3$ 최대 풀링 층을 사용하고,
그 다음에 채널의 수를 변경하기 위한 $1\times 1$ 합성곱 층이 옵니다.
네 분기 모두 입력과 출력에 동일한 높이와 너비를 제공하기 위해 적절한 패딩을 사용합니다.
마지막으로, 각 분기를 따라 출력은 채널 차원을 따라
연결되어 블록의 출력을 구성합니다.
인셉션 블록의 일반적으로 조정되는 하이퍼파라미터는
층당 출력 채널의 수, 즉 서로 다른 크기의 합성곱 간에 용량을 어떻게 할당할지입니다.

```{.python .input}
%%tab mxnet
class Inception(nn.Block):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Branch 2
        self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Branch 3
        self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Branch 4
        self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return np.concatenate((b1, b2, b3, b4), axis=1)
```

```{.python .input}
%%tab pytorch
class Inception(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

```{.python .input}
%%tab tensorflow
class Inception(tf.keras.Model):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])
```

```{.python .input}
%%tab jax
class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each branch
    c1: int
    c2: tuple
    c3: tuple
    c4: int

    def setup(self):
        # Branch 1
        self.b1_1 = nn.Conv(self.c1, kernel_size=(1, 1))
        # Branch 2
        self.b2_1 = nn.Conv(self.c2[0], kernel_size=(1, 1))
        self.b2_2 = nn.Conv(self.c2[1], kernel_size=(3, 3), padding='same')
        # Branch 3
        self.b3_1 = nn.Conv(self.c3[0], kernel_size=(1, 1))
        self.b3_2 = nn.Conv(self.c3[1], kernel_size=(5, 5), padding='same')
        # Branch 4
        self.b4_1 = lambda x: nn.max_pool(x, window_shape=(3, 3),
                                          strides=(1, 1), padding='same')
        self.b4_2 = nn.Conv(self.c4, kernel_size=(1, 1))

    def __call__(self, x):
        b1 = nn.relu(self.b1_1(x))
        b2 = nn.relu(self.b2_2(nn.relu(self.b2_1(x))))
        b3 = nn.relu(self.b3_2(nn.relu(self.b3_1(x))))
        b4 = nn.relu(self.b4_2(self.b4_1(x)))
        return jnp.concatenate((b1, b2, b3, b4), axis=-1)
```

이 네트워크가 왜 그렇게 잘 작동하는지에 대한 직관을 얻으려면,
필터의 조합을 고려해 보세요.
이들은 다양한 필터 크기로 이미지를 탐색합니다.
이는 서로 다른 범위의 세부 사항이
서로 다른 크기의 필터에 의해 효율적으로 인식될 수 있음을 의미합니다.
동시에, 서로 다른 필터에 대해
서로 다른 양의 파라미터를 할당할 수 있습니다.


## [**GoogLeNet 모델**]

:numref:`fig_inception_full`에 표시된 바와 같이, GoogLeNet은 총 9개의 인셉션 블록을 사용하고, 그 사이에 최대 풀링과 함께 세 그룹으로 배열되며,
추정치를 생성하기 위해 머리에 전역 평균 풀링을 사용합니다.
인셉션 블록 사이의 최대 풀링은 차원성을 줄입니다.
줄기 부분에서, 첫 번째 모듈은 AlexNet 및 LeNet과 유사합니다.

![GoogLeNet 아키텍처.](../img/inception-full-90.svg)
:label:`fig_inception_full`

이제 GoogLeNet을 조각별로 구현할 수 있습니다. 줄기부터 시작합시다.
첫 번째 모듈은 64채널 $7\times 7$ 합성곱 층을 사용합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GoogleNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                              activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class GoogleNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10

    def setup(self):
        self.net = nn.Sequential([self.b1(), self.b2(), self.b3(), self.b4(),
                                  self.b5(), nn.Dense(self.num_classes)])

    def b1(self):
        return nn.Sequential([
                nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                      padding='same')])
```

두 번째 모듈은 두 개의 합성곱 층을 사용합니다:
먼저, 64채널 $1\times 1$ 합성곱 층,
그 다음에 채널의 수를 세 배로 늘리는 $3\times 3$ 합성곱 층입니다. 이는 인셉션 블록의 두 번째 분기에 해당하며 본체의 설계를 결론짓습니다. 이 시점에서 192개의 채널이 있습니다.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b2(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([nn.Conv(64, kernel_size=(1, 1)),
                              nn.relu,
                              nn.Conv(192, kernel_size=(3, 3), padding='same'),
                              nn.relu,
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

세 번째 모듈은 두 개의 완전한 인셉션 블록을 직렬로 연결합니다.
첫 번째 인셉션 블록의 출력 채널 수는
$64+128+32+32=256$입니다. 이는 네 분기의 출력 채널 수의 비율이
$2:4:1:1$에 해당합니다. 이를 달성하기 위해, 먼저 두 번째와 세 번째 분기에서
입력 차원을 각각 $\frac{1}{2}$와 $\frac{1}{12}$로 줄여
$96 = 192/2$와 $16 = 192/12$ 채널에 도달합니다.

두 번째 인셉션 블록의 출력 채널 수는
$128+192+96+64=480$으로 증가하여, $128:192:96:64 = 4:6:3:2$의 비율을 산출합니다. 이전과 같이,
두 번째와 세 번째 채널의 중간 차원 수를 줄여야 합니다.
각각 $\frac{1}{2}$와 $\frac{1}{8}$의 스케일이면 충분하며, 각각 $128$과 $32$ 채널을 산출합니다.
이는 다음 `Inception` 블록 생성자의 인수에 의해 캡처됩니다.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b3(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                             Inception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(64, (96, 128), (16, 32), 32),
                              Inception(128, (128, 192), (32, 96), 64),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

네 번째 모듈은 더 복잡합니다.
이는 다섯 개의 인셉션 블록을 직렬로 연결하며,
이들은 각각 $192+208+48+64=512$, $160+224+64+64=512$,
$128+256+64+64=512$, $112+288+64+64=528$,
$256+320+128+128=832$개의 출력 채널을 가집니다.
이러한 분기에 할당된 채널의 수는
세 번째 모듈의 그것과 유사합니다:
$3\times 3$ 합성곱 층을 가진 두 번째 분기가
가장 많은 채널을 출력하고,
$1\times 1$ 합성곱 층만 있는 첫 번째 분기,
$5\times 5$ 합성곱 층을 가진 세 번째 분기,
그리고 $3\times 3$ 최대 풀링 층을 가진 네 번째 분기가 뒤따릅니다.
두 번째와 세 번째 분기는 먼저 비율에 따라
채널의 수를 줄일 것입니다.
이러한 비율은 서로 다른 인셉션 블록에서 약간 다릅니다.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b4(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(192, (96, 208), (16, 48), 64),
                              Inception(160, (112, 224), (24, 64), 64),
                              Inception(128, (128, 256), (24, 64), 64),
                              Inception(112, (144, 288), (32, 64), 64),
                              Inception(256, (160, 320), (32, 128), 128),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

다섯 번째 모듈은 $256+320+128+128=832$와
$384+384+128+128=1024$개의 출력 채널을 가진 두 개의 인셉션 블록을 가집니다.
각 분기에 할당된 채널의 수는
세 번째와 네 번째 모듈과 동일하지만,
구체적인 값에서 차이가 있습니다.
다섯 번째 블록 다음에 출력 층이 옴에 유의해야 합니다.
이 블록은 NiN에서와 마찬가지로
각 채널의 높이와 너비를 1로 변경하기 위해 전역 평균 풀링 층을 사용합니다.
마지막으로, 저희는 출력을 2차원 배열로 만든 다음
출력의 수가 레이블 클래스의 수인
완전 연결 층을 적용합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b5(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D())
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten()])
    if tab.selected('jax'):
        return nn.Sequential([Inception(256, (160, 320), (32, 128), 128),
                              Inception(384, (192, 384), (48, 128), 128),
                              # Flax does not provide a GlobalAvgPool2D layer
                              lambda x: nn.avg_pool(x,
                                                    window_shape=x.shape[1:3],
                                                    strides=x.shape[1:3],
                                                    padding='valid'),
                              lambda x: x.reshape((x.shape[0], -1))])
```

이제 모든 블록 `b1`부터 `b5`까지 정의했으니, 이들을 모두 완전한 네트워크로 조립하는 일만 남았습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.Sequential([
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            tf.keras.layers.Dense(num_classes)])
```

GoogLeNet 모델은 계산적으로 복잡합니다. 채널의 수 선택, 차원성 축소 이전의 블록 수, 채널 간 용량의 상대적 분할 등 비교적 임의적인 하이퍼파라미터가 많음에 주목하세요. 그 대부분은
GoogLeNet이 도입되었을 때 네트워크 정의나 설계 탐색을 위한 자동 도구가
아직 사용 가능하지 않았다는 사실 때문입니다. 예를 들어, 이제 저희는 유능한 딥러닝 프레임워크가 입력 텐서의 차원성을 자동으로 추론할 수 있다는 것을 당연하게 여깁니다. 당시에는 많은 그러한 구성이 실험자에 의해 명시적으로 지정되어야 했고, 따라서 종종 활발한 실험을 느리게 했습니다. 더욱이, 자동 탐색에 필요한 도구들은 여전히 변동하고 있었고 초기 실험들은 주로 비용이 많이 드는 무차별 대입 탐색, 유전 알고리즘, 그리고 유사한 전략에 해당했습니다.

지금으로서는 저희가 수행할 유일한 수정은
[**Fashion-MNIST에서 합리적인 학습 시간을 갖기 위해
입력 높이와 너비를 224에서 96으로 줄이는 것입니다.**]
이는 계산을 단순화합니다. 다양한 모듈 사이에서
출력 모양의 변화를 살펴봅시다.

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow, jax
model = GoogleNet().layer_summary((1, 96, 96, 1))
```

## [**학습**]

이전과 마찬가지로, 저희는 Fashion-MNIST 데이터셋을 사용하여 모델을 학습합니다.
학습 절차를 호출하기 전에 이를 $96 \times 96$ 픽셀 해상도로 변환합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = GoogleNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = GoogleNet(lr=0.01)
    trainer.fit(model, data)
```

## 논의

GoogLeNet의 핵심 특징은 그 선조들보다 계산이 실제로 *더 저렴*하면서도
동시에 향상된 정확도를 제공한다는 것입니다. 이는 오류 감소와 함께 네트워크 평가 비용을 트레이드오프하는
훨씬 더 신중한 네트워크 설계의 시작을 알립니다. 또한 당시에는 전적으로 수동이었지만, 네트워크 설계 하이퍼파라미터에 대한 블록 수준에서의 실험의 시작을 알립니다. 저희는 네트워크 구조 탐색 전략을 논의할 때 :numref:`sec_cnn-design`에서 이 주제를 다시 다룰 것입니다.

다음 절들에서 저희는 네트워크를 상당히 개선할 수 있는 여러 설계 선택(예: 배치 정규화, 잔차 연결, 채널 그룹화)을 마주칠 것입니다. 지금으로서는, 틀림없이 첫 번째 진정으로 현대적인 CNN을 구현한 것에 대해 자랑스러워해도 됩니다.

## 연습문제

1. GoogLeNet은 매우 성공적이어서 점진적으로 속도와 정확도를 향상시키는 여러 차례의 반복을 거쳤습니다. 그 중 일부를 구현하고 실행해 보세요. 다음을 포함합니다:
    1. 나중에 :numref:`sec_batch_norm`에 설명된 대로, 배치 정규화 층 :cite:`Ioffe.Szegedy.2015`을 추가하세요.
    1. :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016`에 설명된 대로, 인셉션 블록에 조정(폭, 합성곱의 선택과 순서)을 가하세요.
    1. :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016`에 설명된 대로, 모델 정규화를 위해 레이블 스무딩을 사용하세요.
    1. 나중에 :numref:`sec_resnet`에 설명된 대로, 잔차 연결 :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`을 추가하여 인셉션 블록에 추가적인 조정을 가하세요.
1. GoogLeNet이 작동하는 데 필요한 최소 이미지 크기는 얼마인가요?
1. Fashion-MNIST의 기본 해상도인 $28 \times 28$ 픽셀에서 작동하는 GoogLeNet의 변형을 설계할 수 있나요? 네트워크의 줄기, 본체, 머리를 어떻게 변경해야 하나요, 만약 변경한다면?
1. AlexNet, VGG, NiN, GoogLeNet의 모델 파라미터 크기를 비교하세요. 후자의 두 네트워크
   아키텍처는 모델 파라미터 크기를 어떻게 크게 줄이나요?
1. GoogLeNet과 AlexNet에 필요한 계산량을 비교하세요. 이것이 가속기 칩의 설계, 예를 들어 메모리 크기, 메모리 대역폭, 캐시 크기, 계산량, 그리고 특수 연산의 이점 측면에서 어떤 영향을 미치나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18004)
:end_tab:
