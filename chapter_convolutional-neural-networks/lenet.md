```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 합성곱 신경망 (LeNet)
:label:`sec_lenet`

이제 저희는 완전히 기능하는 CNN을 조립하는 데 필요한
모든 재료를 가지고 있습니다.
이미지 데이터와의 이전 만남에서, 저희는 Fashion-MNIST 데이터셋의
의류 사진에 소프트맥스 회귀(:numref:`sec_softmax_scratch`)를 가진
선형 모델과 MLP(:numref:`sec_mlp-implementation`)를 적용했습니다.
이러한 데이터를 다루기 쉽게 만들기 위해, 저희는 먼저 각 이미지를 $28\times28$ 행렬에서
고정 길이 $784$차원 벡터로 평탄화한 다음,
완전 연결 계층에서 처리했습니다.
이제 저희는 합성곱 계층을 다룰 수 있게 되었으니,
이미지의 공간 구조를 보존할 수 있습니다.
완전 연결 계층을 합성곱 계층으로 대체하는 것의 추가적인 이점으로,
저희는 훨씬 적은 매개변수를 필요로 하는 더 경제적인 모델을 누릴 것입니다.

이 절에서, 저희는 컴퓨터 비전 과제에서의 성능으로
폭넓은 주목을 받은 최초의 출판된 CNN 중 하나인
*LeNet*을 소개합니다.
이 모델은 당시 AT&T Bell Labs의 연구원이었던 Yann LeCun이
이미지에서 손글씨 숫자를 인식할 목적으로 도입(그리고 그의 이름을 따서 명명)했습니다 :cite:`LeCun.Bottou.Bengio.ea.1998`.
이 작업은 그 기술을 개발하는 10년간의 연구의 정점을 나타냈습니다.
LeCun의 팀은 역전파를 통해 CNN을 성공적으로 훈련시킨
최초의 연구를 출판했습니다 :cite:`LeCun.Boser.Denker.ea.1989`.

당시 LeNet은 지도 학습에서 지배적인 접근법이었던
서포트 벡터 머신의 성능과 일치하는 뛰어난 결과를 달성했으며,
자릿수당 1% 미만의 오류율을 달성했습니다.
LeNet은 결국 ATM 기계에서 예금을 처리하기 위해 숫자를 인식하는 데
적응되었습니다.
오늘날까지도, 일부 ATM은 1990년대에 Yann LeCun과 그의 동료 Leon Bottou가
작성한 코드를 여전히 실행하고 있습니다!

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
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
from types import FunctionType
```

## LeNet

높은 수준에서, (**LeNet (LeNet-5)은 두 부분으로 구성됩니다:
(i) 두 개의 합성곱 계층으로 구성된 합성곱 인코더와
(ii) 세 개의 완전 연결 계층으로 구성된 밀집(dense) 블록**).
아키텍처는 :numref:`img_lenet`에 요약되어 있습니다.

![LeNet의 데이터 흐름. 입력은 손글씨 숫자이고, 출력은 10개의 가능한 결과에 대한 확률입니다.](../img/lenet.svg)
:label:`img_lenet`

각 합성곱 블록의 기본 단위는
합성곱 계층, 시그모이드 활성화 함수,
그리고 후속 평균 풀링 연산입니다.
ReLU와 맥스 풀링이 더 잘 작동하지만,
당시에는 아직 발견되지 않았다는 점에 주목하세요.
각 합성곱 계층은 $5\times 5$ 커널과
시그모이드 활성화 함수를 사용합니다.
이 계층들은 공간적으로 배열된 입력을
다수의 2차원 특성 맵에 매핑하며, 일반적으로
채널 수를 늘립니다.
첫 번째 합성곱 계층은 6개의 출력 채널을 가지고,
두 번째는 16개를 가집니다.
각 $2\times2$ 풀링 연산(스트라이드 2)은
공간적 다운샘플링을 통해 차원성을 $4$의 인수만큼 줄입니다.
합성곱 블록은
(배치 크기, 채널 수, 높이, 너비)로 주어진 모양의 출력을 방출합니다.

합성곱 블록에서 밀집 블록으로 출력을 전달하기 위해,
저희는 미니배치의 각 예제를 평탄화해야 합니다.
다시 말해, 저희는 이 4차원 입력을 받아
완전 연결 계층이 기대하는 2차원 입력으로 변환합니다.
상기하자면, 저희가 원하는 2차원 표현은 첫 번째 차원을 미니배치의 예제를 인덱싱하는 데 사용하고
두 번째 차원은 각 예제의 평탄 벡터 표현을 제공하는 데 사용합니다.
LeNet의 밀집 블록은 세 개의 완전 연결 계층을 가지며,
각각 120, 84, 10개의 출력을 가집니다.
저희가 여전히 분류를 수행하고 있기 때문에,
10차원 출력 계층은
가능한 출력 클래스 수에 대응합니다.

LeNet 내부에서 일어나고 있는 일을 진정으로 이해하는 지점에
도달하는 데에는 약간의 작업이 필요했을 수 있지만,
다음 코드 스니펫이 모던 딥러닝 프레임워크로
그러한 모델을 구현하는 것이 놀라울 정도로 단순함을
여러분에게 납득시키기를 바랍니다.
저희는 단지 `Sequential` 블록을 인스턴스화하고
:numref:`subsec_xavier`에서 도입된 Xavier 초기화를 사용하여
적절한 계층들을 함께 연결하기만 하면 됩니다.

```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])
```

저희는 가우시안 활성화 계층을 소프트맥스 계층으로 대체했다는 점에서 LeNet의 재현에 약간의 자유를 취했습니다. 이는 가우시안 디코더가 요즘 거의 사용되지 않는다는 점도 한몫하여 구현을 크게 단순화합니다. 그 외에는, 이 네트워크는 원래의 LeNet-5 아키텍처와 일치합니다.

:begin_tab:`pytorch, mxnet, tensorflow`
네트워크 내부에서 무슨 일이 일어나는지 봅시다. 단일 채널(흑백)
$28 \times 28$ 이미지를 네트워크에 통과시키고
각 계층에서 출력 모양을 출력함으로써,
저희는 그것의 연산이 :numref:`img_lenet_vert`에서 기대하는 것과 일치하는지
확인하기 위해 [**모델을 점검**]할 수 있습니다.
:end_tab:

:begin_tab:`jax`
네트워크 내부에서 무슨 일이 일어나는지 봅시다. 단일 채널(흑백)
$28 \times 28$ 이미지를 네트워크에 통과시키고
각 계층에서 출력 모양을 출력함으로써,
저희는 그것의 연산이 :numref:`img_lenet_vert`에서 기대하는 것과 일치하는지
확인하기 위해 [**모델을 점검**]할 수 있습니다.
Flax는 네트워크의 계층과 매개변수를 요약하는 멋진 메서드인
`nn.tabulate`를 제공합니다. 여기서 저희는 바운드된 모델을 생성하기 위해 `bind` 메서드를
사용합니다. 변수는 이제 `d2l.Module` 클래스에 바인딩되며, 즉 이 바운드된 모델은
`Sequential` 객체 속성 `net`과 그 안의 `layers`에 접근하는 데 사용될 수 있는
상태가 있는 객체가 됩니다. `bind` 메서드는 대화형 실험을 위해서만
사용되어야 하며, `apply` 메서드의 직접적인 대체물이 아니라는 점에
주목하세요.
:end_tab:

![LeNet-5에 대한 압축된 표기법.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape, key=d2l.get_key()):
    X = jnp.zeros(X_shape)
    params = self.init(key, X)
    bound_model = self.clone().bind(params, mutable=['batch_stats'])
    _ = bound_model(X)
    for layer in bound_model.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

합성곱 블록 전반에 걸쳐 각 계층에서의 표현의 높이와 너비가
(이전 계층에 비해) 감소된다는 점에 주목하세요.
첫 번째 합성곱 계층은 $5 \times 5$ 커널을 사용함으로써 발생할
높이와 너비의 감소를 보상하기 위해
2픽셀의 패딩을 사용합니다.
참고로, 원래 MNIST OCR 데이터셋의 $28 \times 28$ 픽셀이라는 이미지 크기는
$32 \times 32$ 픽셀이던 원본 스캔에서 두 픽셀 행(및 열)을
*잘라낸* 결과입니다. 이는 메가바이트가 중요하던 시절에
공간을 절약하기 위해(30% 감소) 주로 행해진 일입니다.

이와 대조적으로, 두 번째 합성곱 계층은 패딩을 포기하여,
높이와 너비가 모두 네 픽셀씩 감소합니다.
저희가 계층 스택을 올라감에 따라,
채널 수는 계층마다 입력의 1에서
첫 번째 합성곱 계층 후 6,
두 번째 합성곱 계층 후 16으로 증가합니다.
하지만, 각 풀링 계층은 높이와 너비를 절반으로 만듭니다.
마지막으로, 각 완전 연결 계층은 차원성을 줄여서,
최종적으로 그 차원이 클래스 수와 일치하는
출력을 방출합니다.


## 훈련

이제 저희가 모델을 구현했으므로,
[**LeNet-5 모델이 Fashion-MNIST에서 어떻게 작동하는지 살펴보기 위한 실험을 실행**]해 봅시다.

CNN은 매개변수가 더 적지만,
각 매개변수가 훨씬 더 많은 곱셈에 참여하기 때문에
유사하게 깊은 MLP보다 계산이 여전히 더 비쌀 수 있습니다.
GPU에 접근할 수 있다면, 이 시점이
훈련 속도를 높이기 위해 그것을 활용하기에 좋은 때일 수 있습니다.
`d2l.Trainer` 클래스가 모든 세부 사항을 처리한다는 점에
주목하세요.
기본적으로, 그것은 사용 가능한 장치에서
모델 매개변수를 초기화합니다.
MLP와 마찬가지로, 저희의 손실 함수는 교차 엔트로피이며,
저희는 미니배치 확률적 경사 하강법을 통해 그것을 최소화합니다.

```{.python .input}
%%tab pytorch, mxnet, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## 요약

저희는 이 장에서 상당한 진전을 이루었습니다. 저희는 1980년대의 MLP에서 1990년대와 2000년대 초의 CNN으로 이동했습니다. 예를 들어 LeNet-5의 형태로 제안된 아키텍처는 오늘날까지도 의미가 있습니다. LeNet-5로 달성 가능한 Fashion-MNIST에서의 오류율을 MLP로 가능한 최선의 결과(:numref:`sec_mlp-implementation`) 및 ResNet(:numref:`sec_resnet`)과 같이 훨씬 더 진보된 아키텍처의 결과 모두와 비교해 보는 것은 가치가 있습니다. LeNet은 전자보다 후자에 훨씬 더 유사합니다. 저희가 보게 되겠지만, 주요 차이점 중 하나는 더 많은 양의 계산이 훨씬 더 복잡한 아키텍처를 가능하게 했다는 것입니다.

두 번째 차이점은 저희가 LeNet을 구현할 수 있었던 상대적인 용이성입니다. 한때 SN이라는 초기 Lisp 기반 딥러닝 도구 :cite:`Bottou.Le-Cun.1988`를 개선하기 위한 C++ 및 어셈블리 코드 작성과 엔지니어링에 수개월의 가치가 있던 엔지니어링 도전이었고, 마지막으로 모델 실험이었던 것이 이제는 몇 분 안에 달성될 수 있습니다. 딥러닝 모델 개발을 엄청나게 민주화한 것은 바로 이 놀라운 생산성 향상입니다. 다음 장에서, 저희는 이 토끼굴 아래로 내려가 그것이 저희를 어디로 데려가는지 볼 것입니다.

## 연습문제

1. LeNet을 현대화해 봅시다. 다음 변경 사항을 구현하고 테스트하세요.
    1. 평균 풀링을 맥스 풀링으로 대체하세요.
    1. 소프트맥스 계층을 ReLU로 대체하세요.
1. 맥스 풀링과 ReLU 외에도 그 정확도를 향상시키기 위해 LeNet 스타일 네트워크의 크기를 변경해 보세요.
    1. 합성곱 윈도우 크기를 조정하세요.
    1. 출력 채널 수를 조정하세요.
    1. 합성곱 계층 수를 조정하세요.
    1. 완전 연결 계층 수를 조정하세요.
    1. 학습률과 기타 훈련 세부 사항(예: 초기화 및 에폭 수)을 조정하세요.
1. 원래 MNIST 데이터셋에서 향상된 네트워크를 시도해 보세요.
1. 다양한 입력(예: 스웨터와 코트)에 대해 LeNet의 첫 번째와 두 번째 계층의 활성값을 표시하세요.
1. 상당히 다른 이미지(예: 고양이, 자동차, 또는 심지어 무작위 잡음)를 네트워크에 공급할 때 활성값에 어떤 일이 발생합니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/275)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18000)
:end_tab:
