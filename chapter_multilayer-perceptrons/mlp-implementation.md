```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 다층 퍼셉트론의 구현
:label:`sec_mlp-implementation`

다층 퍼셉트론((MLP))은 단순한 선형 모델보다 구현이 그리 복잡하지 않습니다. 핵심적인 개념적
차이는 이제 저희가 여러 층을 이어 붙인다는 점입니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

## 처음부터 구현하기

다시 이러한 네트워크를 처음부터 구현하는 것으로 시작해 봅시다.

### 모델 파라미터 초기화

Fashion-MNIST에는 10개의 클래스가 있고,
각 이미지가 $28 \times 28 = 784$ 격자의
회색조 픽셀 값으로 구성된다는 것을 떠올려 봅시다.
이전과 마찬가지로 저희는 당분간 픽셀 사이의
공간 구조를 무시할 것이므로,
이를 784개의 입력 특징과 10개의 클래스를 가진
분류 데이터셋으로 생각할 수 있습니다.
시작하기 위해, 저희는 [**은닉층 하나와 256개의 은닉 유닛을 가진
MLP를 구현할 것입니다.**]
층의 수와 너비는 모두 조정 가능하며
((이들은 하이퍼파라미터로 간주됩니다)).
일반적으로 저희는 층의 너비를 2의 큰 거듭제곱으로 나누어떨어지도록 선택합니다.
이는 하드웨어에서 메모리가 할당되고 주소가 지정되는 방식 때문에
계산적으로 효율적입니다.

다시 한번, 저희는 여러 개의 텐서로 파라미터를 표현할 것입니다.
*모든 층에 대해* 저희는 하나의 가중치 행렬과 하나의 편향 벡터를
추적해야 한다는 점에 유의하세요.
언제나처럼, 저희는 이러한 파라미터에 대한 손실의 기울기를 위한 메모리를
할당합니다.

:begin_tab:`mxnet`
아래 코드에서, 저희는 먼저 파라미터를 정의하고 초기화한 후
기울기 추적을 활성화합니다.
:end_tab:

:begin_tab:`pytorch`
아래 코드에서 저희는 `nn.Parameter`를 사용하여
클래스 속성을 `autograd`(:numref:`sec_autograd`)에 의해 추적될
파라미터로 자동으로 등록합니다.
:end_tab:

:begin_tab:`tensorflow`
아래 코드에서 저희는 `tf.Variable`을 사용하여
모델 파라미터를 정의합니다.
:end_tab:

:begin_tab:`jax`
아래 코드에서 저희는 `flax.linen.Module.param`을 사용하여
모델 파라미터를 정의합니다.
:end_tab:

```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### 모델

모든 것이 어떻게 작동하는지 확실히 알기 위해, 저희는
내장된 `relu` 함수를 직접 호출하는 대신
[**ReLU 활성화를 직접 구현**]할 것입니다.

```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

공간 구조를 무시하고 있으므로,
저희는 각 2차원 이미지를 `num_inputs` 길이의 평탄한 벡터로
`reshape`합니다.
마지막으로, 저희는 (**저희 모델을 단 몇 줄의 코드로 구현합니다**). 저희가 프레임워크 내장 autograd를 사용하므로 이것이 필요한 전부입니다.

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### 훈련

다행히, [**MLP의 훈련 루프는
소프트맥스 회귀의 훈련 루프와 정확히 동일합니다.**] 저희는 모델, 데이터, 트레이너를 정의한 다음, 마지막으로 모델과 데이터에 대해 `fit` 메서드를 호출합니다.

```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 간결한 구현

예상하시는 대로, 고수준 API에 의존함으로써 저희는 MLP를 훨씬 더 간결하게 구현할 수 있습니다.

### 모델

소프트맥스 회귀의 간결한 구현
(:numref:`sec_softmax_concise`)과 비교하여,
유일한 차이점은 이전에는 단 *하나*의 완전 연결 층을 추가했던 것에서
*두 개*의 완전 연결 층을 추가한다는 것입니다.
첫 번째는 [**은닉층**]이고,
두 번째는 출력층입니다.

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

이전에 저희는 모델 파라미터를 사용해 입력을 변환하기 위해 모델에 `forward` 메서드를 정의했습니다.
이러한 연산은 본질적으로 파이프라인입니다.
여러분은 입력을 받아
변환((예: 가중치와의 행렬 곱셈에 이어 편향 덧셈))을 적용한 후,
현재 변환의 출력을 다음 변환의 입력으로
반복적으로 사용합니다.
하지만 여기서는
어떤 `forward` 메서드도 정의되지 않은 것을 눈치채셨을 것입니다.
사실, `MLP`는 `Module` 클래스 (:numref:`subsec_oo-design-models`)로부터 `forward` 메서드를 상속받아
단순히 `self.net(X)` ((`X`는 입력))를 호출하며,
이는 이제 `Sequential` 클래스를 통해 변환들의 시퀀스로
정의되어 있습니다.
`Sequential` 클래스는 순전파 과정을 추상화하여
저희가 변환에 집중할 수 있게 해줍니다.
저희는 :numref:`subsec_model-construction-sequential`에서 `Sequential` 클래스가 어떻게 작동하는지 더 논의할 것입니다.


### 훈련

[**훈련 루프**]는 저희가 소프트맥스 회귀를 구현했을 때와
정확히 동일합니다.
이러한 모듈성 덕분에 저희는 모델 구조에 관한 사항을
직교적인 고려 사항으로부터
분리할 수 있습니다.

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## 요약

이제 심층 네트워크 설계에 대한 경험이 더 쌓였으므로, 단일 층에서 심층 네트워크의 다층으로 가는 단계는 더 이상 그리 큰 도전이 되지 않습니다. 특히, 저희는 훈련 알고리즘과 데이터 로더를 재사용할 수 있습니다. 다만 MLP를 처음부터 구현하는 것은 그럼에도 불구하고 지저분하다는 점에 유의하세요. 모델 파라미터의 이름 지정과 추적은 모델을 확장하기 어렵게 만듭니다. 예를 들어, 42번 층과 43번 층 사이에 또 다른 층을 삽입하고 싶다고 상상해 보세요. 순차적으로 이름을 다시 짓고자 하지 않는 한, 이 층은 이제 42b번 층이 될 수도 있습니다. 게다가, 만약 저희가 네트워크를 처음부터 구현한다면, 프레임워크가 의미 있는 성능 최적화를 수행하기가 훨씬 더 어려워집니다.

그럼에도 불구하고, 여러분은 이제 완전 연결 심층 네트워크가 신경망 모델링에서 선택받는 방법이었던 1980년대 후반의 최첨단 수준에 도달했습니다. 다음 개념적 단계는 이미지를 고려하는 것입니다. 그렇게 하기 전에, 저희는 모델을 효율적으로 계산하는 방법에 대한 여러 통계적 기초와 세부 사항을 검토할 필요가 있습니다.


## 연습문제

1. 은닉 유닛의 수 `num_hiddens`를 바꾸어 보고 그 수가 모델의 정확도에 어떻게 영향을 미치는지 그려보세요. 이 하이퍼파라미터의 최적값은 무엇인가요?
1. 은닉층을 추가하여 결과에 어떻게 영향을 미치는지 시도해 보세요.
1. 단일 뉴런으로 구성된 은닉층을 삽입하는 것이 왜 나쁜 생각인가요? 무엇이 잘못될 수 있나요?
1. 학습률을 변경하면 결과가 어떻게 바뀌나요? 다른 모든 파라미터를 고정한 상태에서, 어떤 학습률이 가장 좋은 결과를 주나요? 이는 에포크 수와 어떻게 관련되나요?
1. 모든 하이퍼파라미터, 즉 학습률, 에포크 수, 은닉층 수, 그리고 층당 은닉 유닛 수에 대해 공동으로 최적화해 봅시다.
    1. 이 모두에 대해 최적화하여 얻을 수 있는 최고의 결과는 무엇인가요?
    1. 여러 하이퍼파라미터를 다루는 것이 왜 훨씬 더 어려운가요?
    1. 여러 파라미터에 대해 공동으로 최적화하기 위한 효율적인 전략을 설명하세요.
1. 도전적인 문제에 대해 프레임워크 구현과 처음부터 구현의 속도를 비교하세요. 네트워크의 복잡성이 증가함에 따라 어떻게 변하나요?
1. 잘 정렬된 행렬과 정렬되지 않은 행렬에 대한 텐서(-)행렬 곱셈의 속도를 측정하세요. 예를 들어, 차원이 1024, 1025, 1026, 1028, 1032인 행렬에 대해 테스트하세요.
    1. 이는 GPU와 CPU 사이에서 어떻게 달라지나요?
    1. 여러분의 CPU와 GPU의 메모리 버스 너비를 확인하세요.
1. 다양한 활성화 함수를 시도해 보세요. 어떤 것이 가장 잘 작동하나요?
1. 네트워크의 가중치 초기화 사이에 차이가 있나요? 그것이 중요한가요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17985)
:end_tab:
