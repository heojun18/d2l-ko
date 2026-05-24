```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 블록을 사용한 네트워크 (VGG)
:label:`sec_vgg`

AlexNet은 심층 CNN이 좋은 결과를 달성할 수 있다는 경험적 증거를
제공했지만, 후속 연구자들이 새로운 네트워크를 설계하도록 안내하는
일반적인 템플릿은 제공하지 않았습니다.
다음 절에서는 심층 네트워크를 설계하는 데 일반적으로 사용되는
몇 가지 휴리스틱 개념을 소개하겠습니다.

이 분야의 진보는 칩 설계에서의 VLSI(very large scale integration)와
유사합니다.
엔지니어들이 트랜지스터를 배치하는 것에서
논리 요소, 그리고 논리 블록으로 이동한 것처럼 :cite:`Mead.1980` 말입니다.
마찬가지로, 신경망 아키텍처의 설계도
점점 더 추상화되어 왔고,
연구자들은 개별 뉴런의 관점에서 전체 층으로,
그리고 이제는 블록(층의 반복 패턴)으로 사고하는 방향으로 이동했습니다. 10년이 지난 지금, 이는
연구자들이 다른, 관련된 작업에 재사용하기 위해 전체 학습된 모델을 사용하는 단계까지
진행되었습니다. 이러한 대규모 사전 학습된 모델은 일반적으로
*파운데이션 모델* :cite:`bommasani2021opportunities`이라고 불립니다.

네트워크 설계로 돌아갑시다. 블록을 사용한다는 아이디어는 Oxford 대학교의
Visual Geometry Group(VGG)에서 처음 등장했으며,
같은 이름의 *VGG* 네트워크 :cite:`Simonyan.Zisserman.2014`에서였습니다.
어떤 현대 딥러닝 프레임워크를 사용해서도 루프와 서브루틴을 사용하여
이러한 반복 구조를 코드로 구현하는 것은 쉽습니다.

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
```

## (**VGG 블록**)
:label:`subsec_vgg-blocks`

CNN의 기본 구성 블록은
다음과 같은 시퀀스입니다:
(i) 해상도를 유지하기 위한 패딩이 있는
합성곱 층,
(ii) ReLU와 같은 비선형성,
(iii) 해상도를 줄이기 위한 최대 풀링과 같은 풀링 층. 이 접근법의
한 가지 문제는 공간 해상도가 상당히 빠르게 감소한다는 것입니다. 특히,
이는 모든 차원($d$)이 다 쓰이기 전에 네트워크에 $\log_2 d$ 합성곱 층의
엄격한 제한을 부과합니다. 예를 들어, ImageNet의 경우, 이 방식으로는
8개 이상의 합성곱 층을 가지는 것이 불가능합니다.

:citet:`Simonyan.Zisserman.2014`의 핵심 아이디어는 블록 형태로
최대 풀링을 통한 다운샘플링 사이에 *여러* 합성곱을 사용하는 것이었습니다. 그들은 주로
심층 네트워크와 광범위한 네트워크 중 어느 것이 더 잘 수행되는지에 관심이 있었습니다. 예를 들어, 두 개의 $3 \times 3$ 합성곱을 연속적으로 적용하는 것은
단일 $5 \times 5$ 합성곱이 닿는 것과 동일한 픽셀에 닿습니다. 동시에, 후자는 세 개의 $3 \times 3$ 합성곱이
사용하는 것($3 \cdot 9 \cdot c^2$)과 거의 같은 수의 파라미터($25 \cdot c^2$)를 사용합니다.
상당히 상세한 분석에서 그들은 심층 좁은 네트워크가 얕은 대응물보다 훨씬 더 뛰어나다는 것을 보였습니다. 이는 딥러닝을 일반적인 응용에 대해 100층 이상의 더 깊은 네트워크를 향한 탐구로 이끌었습니다.
$3 \times 3$ 합성곱을 쌓는 것은
이후 심층 네트워크의 황금 표준이 되었습니다(이는 :citet:`liu2022convnet`에 의해 최근에야
재검토된 설계 결정입니다). 결과적으로, 작은 합성곱을 위한 빠른 구현은 GPU의 필수 요소가 되었습니다 :cite:`lavin2016fast`.

VGG로 돌아가서: VGG 블록은 패딩이 1인 $3\times3$ 커널을 가진 합성곱 시퀀스(높이와 너비를 유지),
그 다음에 스트라이드가 2인 $2 \times 2$ 최대 풀링 층(각 블록 이후 높이와 너비를 절반으로 만듦)으로 구성됩니다.
아래 코드에서, 저희는 하나의 VGG 블록을 구현하기 위해
`vgg_block`이라는 함수를 정의합니다.

아래 함수는 합성곱 층의 수 `num_convs`와
출력 채널의 수 `num_channels`에 해당하는
두 개의 인수를 받습니다.

```{.python .input  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input  n=4}
%%tab tensorflow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab jax
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## [**VGG 네트워크**]
:label:`subsec_vgg-network`

AlexNet과 LeNet처럼,
VGG 네트워크는 두 부분으로 나눌 수 있습니다:
첫 번째는 주로 합성곱과 풀링 층으로 구성되고,
두 번째는 AlexNet의 것과 동일한 완전 연결 층으로 구성됩니다.
주요 차이점은
합성곱 층이 차원성을 변경하지 않는 비선형 변환으로 그룹화되고, 그 다음에
:numref:`fig_vgg`에 묘사된 대로 해상도 감소 단계가 뒤따른다는 것입니다.

![AlexNet에서 VGG로. 주요 차이점은 VGG가 층의 블록으로 구성되는 반면, AlexNet의 층은 모두 개별적으로 설계되었다는 것입니다.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

네트워크의 합성곱 부분은 :numref:`fig_vgg`의 여러 VGG 블록(`vgg_block` 함수에도 정의됨)을
연속적으로 연결합니다. 이 합성곱의 그룹화는
지난 10년 동안 거의 변하지 않은 패턴이지만, 연산의 구체적인
선택은 상당한 수정을 거쳤습니다.
변수 `arch`는 튜플의 리스트(블록당 하나)로 구성되며,
각 튜플에는 두 개의 값, 즉 합성곱 층의 수와
출력 채널의 수가 포함되며,
이는 정확히 `vgg_block` 함수를
호출하는 데 필요한 인수입니다. 이와 같이, VGG는 특정 표현만이 아닌
*패밀리*의 네트워크를 정의합니다. 특정 네트워크를 구축하려면 단순히 `arch`를 반복하여 블록을 구성합니다.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```{.python .input  n=5}
%%tab jax
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

원래 VGG 네트워크는 5개의 합성곱 블록을 가지고 있었으며,
그 중 처음 두 개는 각각 하나의 합성곱 층을 가지고
나머지 세 개는 각각 두 개의 합성곱 층을 포함합니다.
첫 번째 블록은 64개의 출력 채널을 가지며
각 후속 블록은 출력 채널의 수를 두 배로 만들어,
그 수가 512에 도달할 때까지 계속합니다.
이 네트워크는 8개의 합성곱 층과
3개의 완전 연결 층을 사용하기 때문에, 종종 VGG-11이라고 불립니다.

```{.python .input  n=6}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

```{.python .input}
%%tab jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    training=False).layer_summary((1, 224, 224, 1))
```

보시다시피, 저희는 각 블록에서 높이와 너비를 절반으로 만들어,
마침내 네트워크의 완전 연결 부분에 의한 처리를 위해
표현을 평탄화하기 전에 7의 높이와 너비에 도달합니다.
:citet:`Simonyan.Zisserman.2014`는 VGG의 여러 다른 변형을 설명했습니다.
사실, 새로운 아키텍처를 도입할 때 다른 속도(정확도 트레이드오프)를 가진
네트워크의 *패밀리*를 제안하는 것이 표준이 되었습니다.

## 학습

[**VGG-11은 AlexNet보다 계산적으로 더 까다롭기 때문에
더 적은 수의 채널을 가진 네트워크를 구성합니다.**]
이는 Fashion-MNIST에서 학습하기에 충분합니다.
[**모델 학습**] 과정은 :numref:`sec_alexnet`의 AlexNet의 것과 유사합니다.
다시 한번 검증 손실과 학습 손실 사이의 밀접한 일치를 관찰하세요,
이는 적은 양의 과적합만을 시사합니다.

```{.python .input  n=8}
%%tab mxnet, pytorch, jax
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## 요약

VGG가 진정으로 첫 번째 현대 합성곱 신경망이라고 주장할 수 있습니다. AlexNet은 대규모 딥러닝을 효과적으로 만드는 많은 구성 요소를 도입했지만, 다중 합성곱 블록과 깊고 좁은 네트워크에 대한 선호와 같은 핵심 속성을 도입한 것은 틀림없이 VGG입니다. 또한 이는 실제로 유사하게 파라미터화된 모델의 전체 패밀리인 최초의 네트워크로, 실무자에게 복잡성과 속도 간의 충분한 트레이드오프를 제공합니다. 이는 또한 현대 딥러닝 프레임워크가 빛을 발하는 곳입니다. 네트워크를 지정하기 위해 XML 구성 파일을 생성할 필요가 더 이상 없고, 오히려 간단한 Python 코드를 통해 해당 네트워크를 조립할 수 있습니다.

더 최근에 ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021`은 많은 수의 병렬 계산을 통해 훨씬 더 얕은 아키텍처를 사용하여 경쟁력 있는 성능을 달성할 수 있음을 보여주었습니다. 이는 흥미로운 발전이며 미래의 아키텍처 설계에 영향을 미칠 것이라는 희망이 있습니다. 그러나 이 장의 나머지 부분에서는 지난 10년간의 과학적 진보의 길을 따라가겠습니다.

## 연습문제


1. AlexNet과 비교하여, VGG는 계산 측면에서 훨씬 더 느리고, 더 많은 GPU 메모리도 필요합니다.
    1. AlexNet과 VGG에 필요한 파라미터 수를 비교하세요.
    1. 합성곱 층과 완전 연결 층에서 사용되는 부동 소수점 연산의 수를 비교하세요.
    1. 완전 연결 층으로 인해 생성된 계산 비용을 어떻게 줄일 수 있나요?
1. 네트워크가 11개의 층을 가지고 있음에도 불구하고, 네트워크의 다양한 층과 관련된 차원을 표시할 때, 우리는 8개의 블록(일부 보조 변환 포함)과 관련된 정보만 봅니다. 나머지 세 개의 층은 어디로 갔나요?
1. VGG-16 또는 VGG-19와 같은 다른 일반적인 모델을 구성하기 위해 VGG 논문 :cite:`Simonyan.Zisserman.2014`의 Table 1을 사용하세요.
1. Fashion-MNIST에서 해상도를 $28 \times 28$에서 $224 \times 224$ 차원으로 8배 업샘플링하는 것은 매우 낭비입니다. 네트워크 아키텍처와 해상도 변환을 예를 들어, 입력에 대해 56 또는 84 차원으로 수정해 보세요. 네트워크의 정확도를 줄이지 않고 그렇게 할 수 있나요? 다운샘플링 전에 더 많은 비선형성을 추가하는 아이디어에 대해서는 VGG 논문 :cite:`Simonyan.Zisserman.2014`을 참고하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18002)
:end_tab:
