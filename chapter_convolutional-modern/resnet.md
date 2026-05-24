```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 잔차 네트워크 (ResNet)와 ResNeXt
:label:`sec_resnet`

저희가 점점 더 깊은 네트워크를 설계함에 따라 층을 추가하는 것이 어떻게 네트워크의 복잡도와 표현력을 증가시킬 수 있는지 이해하는 것이 필수적이 됩니다.
훨씬 더 중요한 것은 층을 추가하는 것이 단지 다르게 만드는 것이 아니라 네트워크를 엄격하게 더 표현력 있게 만드는 네트워크를 설계할 수 있는 능력입니다.
약간의 진전을 이루기 위해서는 약간의 수학이 필요합니다.

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

## 함수 클래스

특정 네트워크 아키텍처(학습률 및 기타 하이퍼파라미터 설정과 함께)가 도달할 수 있는 함수의 클래스 $\mathcal{F}$를 고려하세요.
즉, 모든 $f \in \mathcal{F}$에 대해 적절한 데이터셋에서의 학습을 통해 얻을 수 있는 어떤 파라미터 세트(예: 가중치와 편향)가 존재합니다.
저희가 정말로 찾고 싶은 "진실" 함수가 $f^*$라고 가정합시다.
만약 그것이 $\mathcal{F}$에 있다면, 저희는 좋은 상황에 있지만 일반적으로 그렇게 운이 좋지는 않을 것입니다.
대신, 저희는 $\mathcal{F}$ 내에서 저희의 최선의 베팅인 어떤 $f^*_\mathcal{F}$를 찾으려고 시도할 것입니다.
예를 들어,
특성 $\mathbf{X}$와
레이블 $\mathbf{y}$를 가진 데이터셋이 주어지면,
저희는 다음 최적화 문제를 풀어 그것을 찾으려고 시도할 수 있습니다:

$$f^*_\mathcal{F} \stackrel{\textrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \textrm{ subject to } f \in \mathcal{F}.$$

저희는 정규화 :cite:`tikhonov1977solutions,morozov2012methods`가 $\mathcal{F}$의 복잡도를 제어하고
일관성을 달성할 수 있으므로, 학습 데이터의 더 큰 크기가
일반적으로 더 나은 $f^*_\mathcal{F}$로 이어진다는 것을 알고 있습니다.
저희가 다르고 더 강력한 아키텍처 $\mathcal{F}'$을 설계하면 더 나은 결과에 도달해야 한다고 가정하는 것이 합리적일 뿐입니다. 즉, 저희는 $f^*_{\mathcal{F}'}$이 $f^*_{\mathcal{F}}$보다 "더 낫다"고 기대할 것입니다. 그러나, $\mathcal{F} \not\subseteq \mathcal{F}'$이라면 이런 일이 일어나야 한다는 보장조차 없습니다. 사실, $f^*_{\mathcal{F}'}$이 더 나쁠 수도 있습니다.
:numref:`fig_functionclasses`에 묘사된 바와 같이,
비중첩 함수 클래스의 경우, 더 큰 함수 클래스가 항상 "진실" 함수 $f^*$에 더 가깝게 이동하지는 않습니다. 예를 들어,
:numref:`fig_functionclasses`의 왼쪽에서,
$\mathcal{F}_3$이 $\mathcal{F}_1$보다 $f^*$에 더 가깝지만, $\mathcal{F}_6$은 멀어지고 복잡도를 더 증가시키는 것이 $f^*$로부터의 거리를 줄일 수 있다는 보장은 없습니다.
:numref:`fig_functionclasses`의 오른쪽에 있는
$\mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_6$의
중첩된 함수 클래스를 사용하면
저희는 비중첩 함수 클래스로부터의 앞서 언급한 문제를 피할 수 있습니다.


![비중첩 함수 클래스의 경우, 더 큰(면적으로 표시됨) 함수 클래스가 "진실" 함수 ($\mathit{f}^*$)에 더 가까워질 것이라는 보장은 없습니다. 이는 중첩된 함수 클래스에서는 발생하지 않습니다.](../img/functionclasses.svg)
:label:`fig_functionclasses`

따라서,
더 큰 함수 클래스가 더 작은 것들을 포함할 때에만 이들을 증가시키는 것이 네트워크의 표현력을 엄격하게 증가시킨다는 보장이 있습니다.
심층 신경망의 경우,
새로 추가된 층을 항등 함수 $f(\mathbf{x}) = \mathbf{x}$로 학습시킬 수 있다면, 새로운 모델은 원래 모델만큼 효과적일 것입니다. 새로운 모델이 학습 데이터셋에 적합한 더 나은 해를 얻을 수 있으므로, 추가된 층은 학습 오류를 줄이는 것을 더 쉽게 만들 수 있습니다.

이것이 :citet:`He.Zhang.Ren.ea.2016`이 매우 깊은 컴퓨터 비전 모델을 다룰 때 고려한 질문입니다.
그들이 제안한 *잔차 네트워크*(*ResNet*)의 핵심에는 모든 추가 층이
그 요소 중 하나로 항등 함수를
더 쉽게 포함해야 한다는 아이디어가 있습니다.
이러한 고려사항은 상당히 깊이 있지만 놀랍도록 단순한 해결책인
*잔차 블록*으로 이어졌습니다.
이를 통해, ResNet은 2015년 ImageNet Large Scale Visual Recognition Challenge에서 우승했습니다. 이 설계는 심층 신경망을 구축하는 방법에
깊은 영향을 미쳤습니다. 예를 들어, 잔차 블록은 순환 네트워크에도 추가되었습니다 :cite:`prakash2016neural,kim2017residual`. 마찬가지로, 트랜스포머 :cite:`Vaswani.Shazeer.Parmar.ea.2017`는 많은 층의 네트워크를 효율적으로 쌓기 위해 이들을 사용합니다. 또한 그래프 신경망 :cite:`Kipf.Welling.2016`에서도 사용되며, 기본 개념으로, 컴퓨터 비전에서 광범위하게 사용되었습니다 :cite:`Redmon.Farhadi.2018,Ren.He.Girshick.ea.2015`.
잔차 네트워크는 항등 함수를 중심으로 하는 우아한 파라미터화는 없지만 동기 중 일부를 공유하는 하이웨이 네트워크 :cite:`srivastava2015highway`에 의해 선행됨을 참고하세요.


## (**잔차 블록**)
:label:`subsec_residual-blks`

:numref:`fig_residual_block`에 묘사된 바와 같이, 신경망의 국소적인 부분에 집중해 봅시다. 입력을 $\mathbf{x}$로 표시합니다.
저희는 학습을 통해 얻고자 하는 원하는 기본 매핑인 $f(\mathbf{x})$가 상단의 활성화 함수에 대한 입력으로 사용될 것이라고 가정합니다.
왼쪽에서,
점선 상자 내의 부분은
직접 $f(\mathbf{x})$를 학습해야 합니다.
오른쪽에서,
점선 상자 내의 부분은
*잔차 매핑* $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$를 학습해야 하며,
이것이 잔차 블록이 그 이름을 얻는 방식입니다.
항등 매핑 $f(\mathbf{x}) = \mathbf{x}$가 원하는 기본 매핑이라면,
잔차 매핑은 $g(\mathbf{x}) = 0$에 해당하며 따라서 학습하기 더 쉽습니다:
저희는 점선 상자 내의 상단 가중치 층(예: 완전 연결 층 및 합성곱 층)의
가중치와 편향을
0으로
밀어내기만 하면 됩니다.
오른쪽 그림은 ResNet의 *잔차 블록*을 보여주며,
층 입력 $\mathbf{x}$를 덧셈 연산자로 전달하는
실선은
*잔차 연결*(또는 *숏컷 연결*)이라고 합니다.
잔차 블록을 사용하면, 입력은
잔차 연결을 통해 층을 가로질러 더 빠르게 순방향 전파될 수 있습니다.
사실,
잔차 블록은
다중 분기 인셉션 블록의 특별한 경우로 생각할 수 있습니다:
두 개의 분기를 가지며
그 중 하나는 항등 매핑입니다.

![일반 블록(왼쪽)에서, 점선 상자 내의 부분은 직접 매핑 $\mathit{f}(\mathbf{x})$를 학습해야 합니다. 잔차 블록(오른쪽)에서, 점선 상자 내의 부분은 잔차 매핑 $\mathit{g}(\mathbf{x}) = \mathit{f}(\mathbf{x}) - \mathbf{x}$를 학습해야 하며, 이는 항등 매핑 $\mathit{f}(\mathbf{x}) = \mathbf{x}$를 학습하기 더 쉽게 만듭니다.](../img/residual-block.svg)
:label:`fig_residual_block`


ResNet은 VGG의 완전한 $3\times 3$ 합성곱 층 설계를 가지고 있습니다. 잔차 블록은 동일한 수의 출력 채널을 가진 두 개의 $3\times 3$ 합성곱 층을 가집니다. 각 합성곱 층 다음에는 배치 정규화 층과 ReLU 활성화 함수가 옵니다. 그런 다음, 이 두 합성곱 연산을 건너뛰고 최종 ReLU 활성화 함수 직전에 입력을 직접 더합니다.
이런 종류의 설계는 두 합성곱 층의 출력이 함께 더해질 수 있도록 입력과 동일한 모양이어야 합니다. 채널의 수를 변경하려면, 덧셈 연산을 위한 원하는 모양으로 입력을 변환하기 위해 추가적인 $1\times 1$ 합성곱 층을 도입해야 합니다. 아래 코드를 살펴봅시다.

```{.python .input}
%%tab mxnet
class Residual(nn.Block):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
%%tab tensorflow
class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

```{.python .input}
%%tab jax
class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        if self.use_1x1conv:
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)
```

이 코드는 두 가지 유형의 네트워크를 생성합니다: `use_1x1conv=False`일 때 ReLU 비선형성을 적용하기 전에 입력을 출력에 더하는 네트워크; 그리고 더하기 전에 $1 \times 1$ 합성곱을 통해 채널과 해상도를 조정하는 네트워크. :numref:`fig_resnet_block`이 이를 보여줍니다.

![$1 \times 1$ 합성곱이 있는 ResNet 블록과 없는 ResNet 블록으로, 덧셈 연산을 위한 원하는 모양으로 입력을 변환합니다.](../img/resnet-block.svg)
:label:`fig_resnet_block`

이제 [**입력과 출력이 동일한 모양인 상황**]을 살펴봅시다, 여기서 $1 \times 1$ 합성곱은 필요하지 않습니다.

```{.python .input}
%%tab mxnet, pytorch
if tab.selected('mxnet'):
    blk = Residual(3)
    blk.initialize()
if tab.selected('pytorch'):
    blk = Residual(3)
X = d2l.randn(4, 3, 6, 6)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = Residual(3)
X = d2l.normal((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = Residual(3)
X = jax.random.normal(d2l.get_key(), (4, 6, 6, 3))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

저희는 또한 [**출력 채널의 수를 늘리면서 출력 높이와 너비를 절반으로 만들 수 있는**] 옵션도 가지고 있습니다.
이 경우 `use_1x1conv=True`를 통해 $1 \times 1$ 합성곱을 사용합니다. 이는 `strides=2`를 통해 공간 차원성을 줄이기 위해 각 ResNet 블록의 시작 부분에서 유용합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
if tab.selected('mxnet'):
    blk.initialize()
blk(X).shape
```

```{.python .input}
%%tab jax
blk = Residual(6, use_1x1conv=True, strides=(2, 2))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## [**ResNet 모델**]

ResNet의 처음 두 층은 저희가 앞서 설명한 GoogLeNet의 것과 동일합니다: 64개의 출력 채널과 스트라이드 2를 가진 $7\times 7$ 합성곱 층 다음에 스트라이드 2를 가진 $3\times 3$ 최대 풀링 층이 옵니다. 차이점은 ResNet에서 각 합성곱 층 다음에 배치 정규화 층이 추가된다는 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet(d2l.Classifier):
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
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                       padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class ResNet(d2l.Classifier):
    arch: tuple
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                  padding='same')])
```

GoogLeNet은 인셉션 블록으로 구성된 네 개의 모듈을 사용합니다.
그러나, ResNet은 잔차 블록으로 구성된 네 개의 모듈을 사용하며, 각 모듈은 동일한 수의 출력 채널을 가진 여러 잔차 블록을 사용합니다.
첫 번째 모듈의 채널 수는 입력 채널의 수와 동일합니다. 스트라이드 2를 가진 최대 풀링 층이 이미 사용되었으므로, 높이와 너비를 줄일 필요가 없습니다. 후속 모듈 각각의 첫 번째 잔차 블록에서, 채널의 수는 이전 모듈의 것에 비해 두 배가 되고, 높이와 너비는 절반이 됩니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = tf.keras.models.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab jax
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True,
                                strides=(2, 2), training=self.training))
        else:
            blk.append(Residual(num_channels, training=self.training))
    return nn.Sequential(blk)
```

그런 다음, 저희는 모든 모듈을 ResNet에 추가합니다. 여기서, 각 모듈에 대해 두 개의 잔차 블록이 사용됩니다. 마지막으로, GoogLeNet과 마찬가지로, 저희는 전역 평균 풀링 층 다음에 완전 연결 층 출력을 추가합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
# %%tab jax
@d2l.add_to_class(ResNet)
def create_net(self):
    net = nn.Sequential([self.b1()])
    for i, b in enumerate(self.arch):
        net.layers.extend([self.block(*b, first_block=(i==0))])
    net.layers.extend([nn.Sequential([
        # Flax does not provide a GlobalAvg2D layer
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

각 모듈에는 네 개의 합성곱 층이 있습니다($1\times 1$ 합성곱 층 제외). 첫 번째 $7\times 7$ 합성곱 층과 최종 완전 연결 층을 합쳐, 총 18개의 층이 있습니다. 따라서, 이 모델은 일반적으로 ResNet-18로 알려져 있습니다.
모듈에서 다른 채널 수와 잔차 블록을 구성함으로써, 저희는 더 깊은 152층 ResNet-152와 같은 다른 ResNet 모델을 만들 수 있습니다. ResNet의 주요 아키텍처가 GoogLeNet의 것과 유사하지만, ResNet의 구조는 더 단순하고 수정하기 더 쉽습니다. 이 모든 요소가 ResNet의 빠르고 광범위한 사용으로 이어졌습니다. :numref:`fig_resnet18`은 전체 ResNet-18을 묘사합니다.

![ResNet-18 아키텍처.](../img/resnet18-90.svg)
:label:`fig_resnet18`

ResNet을 학습시키기 전에, [**ResNet의 다양한 모듈에서 입력 모양이 어떻게 변하는지 관찰해 봅시다**]. 이전의 모든 아키텍처와 마찬가지로, 해상도는 감소하고 채널 수는 전역 평균 풀링 층이 모든 특성을 통합할 때까지 증가합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)
```

```{.python .input}
%%tab jax
class ResNet18(ResNet):
    arch: tuple = ((2, 64), (2, 128), (2, 256), (2, 512))
    lr: float = 0.1
    num_classes: int = 10
```

```{.python .input}
%%tab pytorch, mxnet
ResNet18().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
ResNet18().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
ResNet18(training=False).layer_summary((1, 96, 96, 1))
```

## [**학습**]

이전과 같이, 저희는 Fashion-MNIST 데이터셋에서 ResNet을 학습시킵니다. ResNet은 상당히 강력하고 유연한 아키텍처입니다. 학습 및 검증 손실을 캡처하는 도표는 두 그래프 사이의 상당한 격차를 보여주며, 학습 손실이 상당히 더 낮습니다. 이러한 유연성의 네트워크에는, 더 많은 학습 데이터가 격차를 좁히고 정확도를 향상시키는 데 뚜렷한 이점을 제공할 것입니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = ResNet18(lr=0.01)
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
    model = ResNet18(lr=0.01)
    trainer.fit(model, data)
```

## ResNeXt
:label:`subsec_resnext`

ResNet 설계에서 마주치는 도전 중 하나는 주어진 블록 내에서 비선형성과 차원성 사이의 트레이드오프입니다. 즉, 저희는 층의 수를 늘리거나 합성곱의 너비를 늘려 더 많은 비선형성을 추가할 수 있습니다. 대안적인 전략은 블록 간에 정보를 운반할 수 있는 채널의 수를 늘리는 것입니다. 불행히도, 후자는 $c_\textrm{i}$ 채널을 수용하고 $c_\textrm{o}$ 채널을 방출하는 계산 비용이 $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$에 비례하므로 이차 페널티가 따릅니다(:numref:`sec_channels`의 저희의 논의 참조).

저희는 별도의 그룹에서 블록을 통해 정보가 흐르는 :numref:`fig_inception`의 인셉션 블록에서 영감을 얻을 수 있습니다. :numref:`fig_resnet_block`의 ResNet 블록에 여러 독립적인 그룹의 아이디어를 적용하면 ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`의 설계로 이어졌습니다.
인셉션의 다양한 변환의 모듬과는 달리,
ResNeXt는 모든 분기에서 *동일한* 변환을 채택하여,
각 분기의 수동 조정의 필요성을 최소화합니다.

![ResNeXt 블록. $\mathit{g}$ 그룹의 그룹화된 합성곱의 사용은 밀집 합성곱보다 $\mathit{g}$배 빠릅니다. 중간 채널 수 $\mathit{b}$가 $\mathit{c}$보다 작을 때 병목 잔차 블록입니다.](../img/resnext-block.svg)
:label:`fig_resnext_block`

$c_\textrm{i}$에서 $c_\textrm{o}$ 채널로의 합성곱을 크기 $c_\textrm{i}/g$의 $g$ 그룹으로 분할하여 크기 $c_\textrm{o}/g$의 $g$ 출력을 생성하는 것은, 매우 적절하게도, *그룹화된 합성곱*이라고 합니다. 계산 비용은 (비례적으로) $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$에서 $\mathcal{O}(g \cdot (c_\textrm{i}/g) \cdot (c_\textrm{o}/g)) = \mathcal{O}(c_\textrm{i} \cdot c_\textrm{o} / g)$로 감소합니다. 즉, $g$배 더 빠릅니다. 더 좋은 점은, 출력을 생성하는 데 필요한 파라미터의 수도 $c_\textrm{i} \times c_\textrm{o}$ 행렬에서 크기 $(c_\textrm{i}/g) \times (c_\textrm{o}/g)$의 $g$개의 더 작은 행렬로 감소하며, 다시 $g$배의 감소입니다. 다음에서는 $c_\textrm{i}$와 $c_\textrm{o}$ 모두 $g$로 나누어진다고 가정합니다.

이 설계에서의 유일한 도전은 $g$ 그룹 간에 정보가 교환되지 않는다는 것입니다.
:numref:`fig_resnext_block`의 ResNeXt 블록은 두 가지 방식으로 이를 수정합니다: $3 \times 3$ 커널을 가진 그룹화된 합성곱이 두 개의 $1 \times 1$ 합성곱 사이에 끼워집니다. 두 번째는 채널의 수를 다시 변경하는 이중 역할을 합니다. 이점은 저희가 $1 \times 1$ 커널에 대해서만 $\mathcal{O}(c \cdot b)$ 비용을 지불하고 $3 \times 3$ 커널에 대해 $\mathcal{O}(b^2 / g)$ 비용으로 처리할 수 있다는 것입니다.
:numref:`subsec_residual-blks`의 잔차 블록 구현과 유사하게, 잔차 연결은 $1 \times 1$ 합성곱으로 대체(따라서 일반화)됩니다.

:numref:`fig_resnext_block`의 오른쪽 그림은 결과 네트워크 블록의 훨씬 더 간결한 요약을 제공합니다. 이는 또한 :numref:`sec_cnn-design`의 일반적인 현대 CNN 설계에서 주요 역할을 할 것입니다. 그룹화된 합성곱의 아이디어는 AlexNet의 구현 :cite:`Krizhevsky.Sutskever.Hinton.2012`까지 거슬러 올라간다는 점에 주목하세요. 제한된 메모리를 가진 두 GPU에 걸쳐 네트워크를 분산할 때, 구현은 각 GPU를 자체 채널로 취급했으며 부작용은 없었습니다.

다음 `ResNeXtBlock` 클래스의 구현은 `bot_channels`($b$) 중간(병목) 채널과 함께
인수로 `groups`($g$)를 받습니다. 마지막으로, 저희가 표현의 높이와 너비를 줄여야 할 때, `use_1x1conv=True, strides=2`로 설정하여 스트라이드 $2$를 추가합니다.

```{.python .input}
%%tab mxnet
class ResNeXtBlock(nn.Block):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1, 
                               strides=strides, groups=bot_channels//groups)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, 
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
```

```{.python .input}
%%tab tensorflow
class ResNeXtBlock(tf.keras.Model):  #@save
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1)
        self.conv2 = tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                            padding="same",
                                            groups=bot_channels//groups)
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                                                strides=strides)
            self.bn4 = tf.keras.layers.BatchNormalization()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return tf.keras.activations.relu(Y + X)
```

```{.python .input}
%%tab jax
class ResNeXtBlock(nn.Module):  #@save
    """The ResNeXt block."""
    num_channels: int
    groups: int
    bot_mul: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        bot_channels = int(round(self.num_channels * self.bot_mul))
        self.conv1 = nn.Conv(bot_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.conv2 = nn.Conv(bot_channels, kernel_size=(3, 3),
                               strides=self.strides, padding='same',
                               feature_group_count=bot_channels//self.groups)
        self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)
        self.bn3 = nn.BatchNorm(not self.training)
        if self.use_1x1conv:
            self.conv4 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                       strides=self.strides)
            self.bn4 = nn.BatchNorm(not self.training)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)
```

그 사용은 앞서 논의된 `ResNetBlock`의 사용과 완전히 유사합니다. 예를 들어, (`use_1x1conv=False, strides=1`)을 사용할 때, 입력과 출력은 동일한 모양입니다. 대안적으로, `use_1x1conv=True, strides=2`로 설정하면 출력 높이와 너비를 절반으로 만듭니다.

```{.python .input}
%%tab mxnet, pytorch
blk = ResNeXtBlock(32, 16, 1)
if tab.selected('mxnet'):
    blk.initialize()
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = ResNeXtBlock(32, 16, 1)
X = d2l.normal((4, 96, 96, 32))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = ResNeXtBlock(32, 16, 1)
X = jnp.zeros((4, 96, 96, 32))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## 요약 및 논의

중첩된 함수 클래스는 용량을 추가할 때 미묘하게 *다른* 함수 클래스가 아닌 엄격하게 *더 강력한* 함수 클래스를 얻을 수 있게 해주므로 바람직합니다. 이를 달성하는 한 가지 방법은 추가 층이 단순히 입력을 출력으로 통과시키도록 하는 것입니다. 잔차 연결이 이를 가능하게 합니다. 결과적으로, 이는 단순 함수가 $f(\mathbf{x}) = 0$ 형태인 것에서 $f(\mathbf{x}) = \mathbf{x}$처럼 보이는 것으로 귀납 편향을 변경합니다.


잔차 매핑은 가중치 층의 파라미터를 0으로 미는 것과 같이 항등 함수를 더 쉽게 학습할 수 있습니다. 저희는 잔차 블록을 가짐으로써 효과적인 *심층* 신경망을 학습할 수 있습니다. 입력은 잔차 연결을 통해 층을 가로질러 더 빠르게 순방향 전파될 수 있습니다. 결과적으로, 저희는 따라서 훨씬 더 깊은 네트워크를 학습할 수 있습니다. 예를 들어, 원래 ResNet 논문 :cite:`He.Zhang.Ren.ea.2016`은 최대 152개의 층을 허용했습니다. 잔차 네트워크의 또 다른 이점은 학습 과정 *동안* 항등 함수로 초기화된 층을 추가할 수 있다는 것입니다. 결국, 층의 기본 동작은 데이터를 변경 없이 통과시키는 것입니다. 이는 일부 경우에 매우 큰 네트워크의 학습을 가속화할 수 있습니다.

잔차 연결 이전에,
게이팅 단위를 가진 우회 경로가
100층 이상의 하이웨이 네트워크를 효과적으로 학습하기 위해 도입되었습니다
:cite:`srivastava2015highway`.
우회 경로로 항등 함수를 사용하여,
ResNet은 다중 컴퓨터 비전 작업에서
놀랍도록 잘 수행되었습니다.
잔차 연결은 합성곱이든 순차적이든 후속 심층 신경망의 설계에 주요 영향을 미쳤습니다.
저희가 나중에 소개하겠지만,
트랜스포머 아키텍처 :cite:`Vaswani.Shazeer.Parmar.ea.2017`는
잔차 연결(다른 설계 선택과 함께)을 채택하고
언어, 비전, 음성, 강화 학습과 같이
다양한 영역에 만연합니다.

ResNeXt는 합성곱 신경망의 설계가 시간이 지남에 따라 어떻게 진화했는지에 대한 예입니다: 계산에 더 절약적이고 활성화의 크기(채널 수)와 트레이드오프함으로써, 더 낮은 비용으로 더 빠르고 더 정확한 네트워크를 가능하게 합니다. 그룹화된 합성곱을 보는 대안적인 방법은 합성곱 가중치에 대한 블록 대각 행렬을 생각하는 것입니다. 더 효율적인 네트워크로 이어지는 그러한 "트릭"이 꽤 많이 있다는 점에 주목하세요. 예를 들어, ShiftNet :cite:`wu2018shift`은 단순히 채널에 이동된 활성화를 추가함으로써 $3 \times 3$ 합성곱의 효과를 모방하며, 이번에는 어떤 계산 비용도 없이 증가된 함수 복잡도를 제공합니다.

지금까지 저희가 논의한 설계의 공통 특징은 네트워크 설계가 상당히 수동적이며, 주로 "올바른" 네트워크 하이퍼파라미터를 찾기 위한 설계자의 독창성에 의존한다는 것입니다. 분명히 실행 가능하지만, 이는 또한 인간 시간 측면에서 매우 비용이 많이 들고 결과가 어떤 의미에서든 최적이라는 보장이 없습니다. :numref:`sec_cnn-design`에서 저희는 보다 자동화된 방식으로 고품질 네트워크를 얻기 위한 여러 전략을 논의할 것입니다. 특히, 저희는 RegNetX/Y 모델
:cite:`Radosavovic.Kosaraju.Girshick.ea.2020`로 이어진 *네트워크 설계 공간*의 개념을 검토할 것입니다.

## 연습문제

1. :numref:`fig_inception`의 인셉션 블록과 잔차 블록 간의 주요 차이점은 무엇인가요? 계산, 정확도, 그리고 이들이 설명할 수 있는 함수의 클래스 측면에서 어떻게 비교되나요?
1. 네트워크의 다양한 변형을 구현하기 위해 ResNet 논문 :cite:`He.Zhang.Ren.ea.2016`의 Table 1을 참조하세요.
1. 더 깊은 네트워크의 경우, ResNet은 모델 복잡도를 줄이기 위해 "병목" 아키텍처를 도입합니다. 이를 구현해 보세요.
1. ResNet의 후속 버전에서, 저자들은 "합성곱, 배치 정규화, 활성화" 구조를 "배치 정규화, 활성화, 합성곱" 구조로 변경했습니다. 이 개선을 직접 만들어 보세요. 자세한 내용은 :citet:`He.Zhang.Ren.ea.2016*1`의 Figure 1을 참조하세요.
1. 함수 클래스가 중첩되어 있다 하더라도, 왜 함수의 복잡도를 무한히 증가시킬 수 없나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8737)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18006)
:end_tab:
