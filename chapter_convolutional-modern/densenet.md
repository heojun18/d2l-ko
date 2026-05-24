```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 밀집 연결 네트워크 (DenseNet)
:label:`sec_densenet`

ResNet은 심층 네트워크에서 함수를 어떻게 파라미터화할지에 대한 관점을 크게 변화시켰습니다. *DenseNet*(밀집 합성곱 네트워크)은 어느 정도 이것의 논리적 확장입니다 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
DenseNet은 각 층이 모든 이전 층에 연결되는
연결성 패턴과
이전 층의 특성을 보존하고 재사용하기 위한 연결 연산(ResNet의 덧셈 연산자 대신)으로
특징지어집니다.
이에 어떻게 도달하는지 이해하기 위해, 수학으로 잠깐 우회해 봅시다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
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
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## ResNet에서 DenseNet으로

함수에 대한 테일러 전개를 떠올려 보세요. $x = 0$ 점에서 다음과 같이 쓸 수 있습니다.

$$f(x) = f(0) + x \cdot \left[f'(0) + x \cdot \left[\frac{f''(0)}{2!}  + x \cdot \left[\frac{f'''(0)}{3!}  + \cdots \right]\right]\right].$$


핵심은 함수를 점점 더 높은 차수의 항으로 분해한다는 것입니다. 유사한 맥락에서, ResNet은 함수를 다음과 같이 분해합니다.

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

즉, ResNet은 $f$를 단순한 선형 항과 더 복잡한
비선형 항으로 분해합니다.
두 항 이상의 정보를 캡처(반드시 더하는 것은 아님)하고 싶다면 어떨까요?
그러한 해결책 중 하나는 DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`입니다.

![교차 층 연결에서 ResNet(왼쪽)과 DenseNet(오른쪽)의 주요 차이: 덧셈과 연결의 사용.](../img/densenet-block.svg)
:label:`fig_densenet_block`

:numref:`fig_densenet_block`에 표시된 바와 같이, ResNet과 DenseNet의 핵심 차이는 후자의 경우 출력이 더해지는 것이 아니라 *연결*된다는 것입니다($[,]$로 표시됨).
결과적으로, 저희는 점점 더 복잡한 함수 시퀀스를 적용한 후의 값으로 $\mathbf{x}$에서 매핑을 수행합니다:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), f_3\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right)\right]\right), \ldots\right].$$

마지막으로, 모든 이러한 함수는 MLP에서 결합되어 특성의 수를 다시 줄입니다. 구현 측면에서 이것은 매우 단순합니다:
항을 더하는 대신, 저희는 이들을 연결합니다. DenseNet이라는 이름은 변수 간의 의존성 그래프가 상당히 밀집된다는 사실에서 비롯됩니다. 그러한 체인의 마지막 층은 모든 이전 층에 밀집 연결됩니다. 밀집 연결은 :numref:`fig_densenet`에 표시됩니다.

![DenseNet의 밀집 연결. 깊이에 따라 차원성이 어떻게 증가하는지 주목하세요.](../img/densenet.svg)
:label:`fig_densenet`

DenseNet을 구성하는 주요 구성 요소는 *밀집 블록*과 *전이 층*입니다. 전자는 입력과 출력이 어떻게 연결되는지 정의하는 반면, 후자는 채널의 수를 제어하여 너무 크지 않도록 합니다.
확장 $\mathbf{x} \to \left[\mathbf{x}, f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), \ldots \right]$는 상당히 고차원일 수 있기 때문입니다.


## [**밀집 블록**]

DenseNet은 ResNet의 수정된 "배치 정규화, 활성화, 합성곱"
구조(:numref:`sec_resnet`의 연습문제 참조)를 사용합니다.
먼저, 이 합성곱 블록 구조를 구현합니다.

```{.python .input}
%%tab mxnet
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
%%tab pytorch
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```

```{.python .input}
%%tab tensorflow
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

```{.python .input}
%%tab jax
class ConvBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        Y = nn.relu(nn.BatchNorm(not self.training)(X))
        Y = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))(Y)
        Y = jnp.concatenate((X, Y), axis=-1)
        return Y
```

*밀집 블록*은 동일한 수의 출력 채널을 사용하는 여러 합성곱 블록으로 구성됩니다. 그러나 순방향 전파에서, 저희는 채널 차원에서 각 합성곱 블록의 입력과 출력을 연결합니다. 지연 평가는 차원성을 자동으로 조정할 수 있게 합니다.

```{.python .input}
%%tab mxnet
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
%%tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
%%tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

```{.python .input}
%%tab jax
class DenseBlock(nn.Module):
    num_convs: int
    num_channels: int
    training: bool = True

    def setup(self):
        layer = []
        for i in range(self.num_convs):
            layer.append(ConvBlock(self.num_channels, self.training))
        self.net = nn.Sequential(layer)

    def __call__(self, X):
        return self.net(X)
```

다음 예제에서,
저희는 10개의 출력 채널을 가진 두 개의 합성곱 블록으로 [**`DenseBlock` 인스턴스를 정의합니다**].
세 개의 채널을 가진 입력을 사용할 때, 저희는 $3 + 10 + 10=23$개의 채널을 가진 출력을 얻을 것입니다. 합성곱 블록 채널의 수는 입력 채널 수에 대한 출력 채널 수의 증가를 제어합니다. 이는 또한 *성장률*이라고도 합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = DenseBlock(2, 10)
if tab.selected('mxnet'):
    X = np.random.uniform(size=(4, 3, 8, 8))
    blk.initialize()
if tab.selected('pytorch'):
    X = torch.randn(4, 3, 8, 8)
if tab.selected('tensorflow'):
    X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = DenseBlock(2, 10)
X = jnp.zeros((4, 8, 8, 3))
Y = blk.init_with_output(d2l.get_key(), X)[0]
Y.shape
```

## [**전이 층**]

각 밀집 블록이 채널의 수를 증가시키므로, 너무 많이 추가하면 지나치게 복잡한 모델로 이어집니다. *전이 층*은 모델의 복잡도를 제어하는 데 사용됩니다. 이는 $1\times 1$ 합성곱을 사용하여 채널의 수를 줄입니다. 또한, 스트라이드 2를 가진 평균 풀링을 통해 높이와 너비를 절반으로 만듭니다.

```{.python .input}
%%tab mxnet
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
%%tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```{.python .input}
%%tab jax
class TransitionBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        X = nn.BatchNorm(not self.training)(X)
        X = nn.relu(X)
        X = nn.Conv(self.num_channels, kernel_size=(1, 1))(X)
        X = nn.avg_pool(X, window_shape=(2, 2), strides=(2, 2))
        return X
```

이전 예제의 밀집 블록 출력에 10개의 채널을 가진 [**전이 층을 적용합니다**]. 이는 출력 채널의 수를 10으로 줄이고, 높이와 너비를 절반으로 만듭니다.

```{.python .input}
%%tab mxnet
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
%%tab pytorch
blk = transition_block(10)
blk(Y).shape
```

```{.python .input}
%%tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

```{.python .input}
%%tab jax
blk = TransitionBlock(10)
blk.init_with_output(d2l.get_key(), Y)[0].shape
```

## [**DenseNet 모델**]

다음으로, 저희는 DenseNet 모델을 구성할 것입니다. DenseNet은 먼저 ResNet과 동일한 단일 합성곱 층과 최대 풀링 층을 사용합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class DenseNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    64, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(
                    pool_size=3, strides=2, padding='same')])
```

```{.python .input}
%%tab jax
class DenseNet(d2l.Classifier):
    num_channels: int = 64
    growth_rate: int = 32
    arch: tuple = (4, 4, 4, 4)
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3),
                                  strides=(2, 2), padding='same')
        ])
```

그런 다음, ResNet이 사용하는 잔차 블록으로 구성된 네 개의 모듈과 유사하게,
DenseNet은 네 개의 밀집 블록을 사용합니다.
ResNet과 마찬가지로, 저희는 각 밀집 블록에서 사용되는 합성곱 층의 수를 설정할 수 있습니다. 여기서, 저희는 :numref:`sec_resnet`의 ResNet-18 모델과 일치하도록 이를 4로 설정합니다. 또한, 저희는 밀집 블록의 합성곱 층에 대한 채널의 수(즉, 성장률)를 32로 설정하므로, 각 밀집 블록에 128개의 채널이 추가될 것입니다.

ResNet에서, 높이와 너비는 스트라이드 2를 가진 잔차 블록에 의해 각 모듈 사이에서 줄어듭니다. 여기서, 저희는 높이와 너비를 절반으로 만들고 채널 수를 절반으로 만들기 위해 전이 층을 사용합니다. ResNet과 유사하게, 전역 풀링 층과 완전 연결 층이 끝에 연결되어 출력을 생성합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(transition_block(num_channels))
        self.net.add(nn.BatchNorm(), nn.Activation('relu'),
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                              growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(TransitionBlock(num_channels))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(DenseNet)
def create_net(self):
    net = self.b1()
    for i, num_convs in enumerate(self.arch):
        net.layers.extend([DenseBlock(num_convs, self.growth_rate,
                                      training=self.training)])
        # The number of output channels in the previous dense block
        num_channels = self.num_channels + (num_convs * self.growth_rate)
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(self.arch) - 1:
            num_channels //= 2
            net.layers.extend([TransitionBlock(num_channels,
                                               training=self.training)])
    net.layers.extend([
        nn.BatchNorm(not self.training),
        nn.relu,
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)
    ])
    return net
```

## [**학습**]

이 절에서는 더 깊은 네트워크를 사용하고 있으므로, 계산을 단순화하기 위해 입력 높이와 너비를 224에서 96으로 줄입니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = DenseNet(lr=0.01)
    trainer.fit(model, data)
```

## 요약 및 논의

DenseNet을 구성하는 주요 구성 요소는 밀집 블록과 전이 층입니다. 후자의 경우, 채널의 수를 다시 줄이는 전이 층을 추가하여 네트워크를 구성할 때 차원성을 제어해야 합니다.
교차 층 연결 측면에서, 입력과 출력이 함께 더해지는 ResNet과 대조적으로, DenseNet은 채널 차원에서 입력과 출력을 연결합니다.
이러한 연결 연산이 계산 효율성을 달성하기 위해 특성을 재사용하지만,
불행히도 무거운 GPU 메모리 소비로 이어집니다.
결과적으로,
DenseNet을 적용하려면 학습 시간을 증가시킬 수 있는 더 메모리 효율적인 구현이 필요할 수 있습니다 :cite:`pleiss2017memory`.


## 연습문제

1. 전이 층에서 최대 풀링이 아닌 평균 풀링을 사용하는 이유는 무엇인가요?
1. DenseNet 논문에서 언급된 장점 중 하나는 모델 파라미터가 ResNet의 것보다 작다는 것입니다. 왜 그런가요?
1. DenseNet이 비판받은 한 가지 문제는 그 높은 메모리 소비입니다.
    1. 정말로 그런가요? 실제 GPU 메모리 소비를 경험적으로 비교하기 위해 입력 모양을 $224\times 224$로 변경해 보세요.
    1. 메모리 소비를 줄이는 대안적인 수단을 생각할 수 있나요? 프레임워크를 어떻게 변경해야 하나요?
1. DenseNet 논문 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`의 Table 1에 제시된 다양한 DenseNet 버전을 구현하세요.
1. DenseNet 아이디어를 적용하여 MLP 기반 모델을 설계하세요. :numref:`sec_kaggle_house`의 주택 가격 예측 작업에 적용하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18008)
:end_tab:
