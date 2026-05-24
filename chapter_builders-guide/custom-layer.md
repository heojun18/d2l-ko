```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 커스텀 층

딥러닝의 성공 뒤에 자리한 한 가지 요인은
다양한 과제에 적합한 아키텍처를 설계할 수 있도록
창의적인 방식으로 조합할 수 있는
폭넓은 종류의 층을 사용할 수 있다는 점입니다.
예를 들어 연구자들은 이미지와 텍스트를 다루기 위한 층,
순차 데이터를 반복 처리하기 위한 층,
그리고 동적 계획법을 수행하기 위한 층 등을
특별히 발명해 왔습니다.
조만간 여러분은 딥러닝 프레임워크에 아직 존재하지 않는
어떤 층이 필요해질 것입니다.
이런 경우에는 직접 커스텀 층을 만들어야 합니다.
이 절에서는 그 방법을 보여 드립니다.

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
from torch.nn import functional as F
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

## (**파라미터가 없는 층**)

먼저 자체 파라미터를 가지지 않는
커스텀 층을 만들어 보겠습니다.
:numref:`sec_model_construction`에서 모듈을
소개한 내용을 떠올려 보면 익숙한 형태일 것입니다.
다음 `CenteredLayer` 클래스는 단순히
입력에서 평균을 빼는 역할을 합니다.
이를 구현하려면 기본 층 클래스를 상속하고
순전파 함수를 구현하기만 하면 됩니다.

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

데이터를 일부 흘려보내 저희 층이 의도대로 동작하는지 확인해 보겠습니다.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

이제 [**저희가 만든 층을 구성 요소로 사용해
더 복잡한 모델을 만들 수 있습니다.**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

추가적인 점검을 위해, 무작위 데이터를 신경망에
흘려보내고 평균이 실제로 0인지 확인해 볼 수 있습니다.
부동소수점 수를 다루기 때문에, 양자화 영향으로
여전히 매우 작은 0이 아닌 값을 볼 수도 있습니다.

:begin_tab:`jax`
여기서는 신경망의 출력과 파라미터를 모두 반환하는
`init_with_output` 메서드를 활용합니다.
이 경우에는 출력에만 초점을 맞춥니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**파라미터가 있는 층**]

이제 간단한 층을 정의하는 법을 알았으니,
학습을 통해 조정할 수 있는 파라미터를 가진 층을
정의하는 단계로 넘어가 보겠습니다.
저희는 내장 함수들을 사용해 파라미터를 만들 수 있는데,
이 함수들은 몇 가지 기본적인 관리 기능을 제공합니다.
구체적으로는 모델 파라미터에 대한 접근, 초기화,
공유, 저장, 적재를 관장합니다.
이렇게 하면 다른 이점은 차치하더라도, 모든 커스텀 층마다
직접 직렬화 루틴을 작성할 필요가 없어집니다.

이제 저희만의 완전 연결 층 버전을 구현해 보겠습니다.
이 층은 두 개의 파라미터, 즉 가중치를 나타내는 것 하나와
편향을 나타내는 것 하나가 필요하다는 점을 떠올려 보세요.
이번 구현에서는 기본 활성화로 ReLU를 함께 넣어 둡니다.
이 층은 두 개의 입력 인자가 필요한데, `in_units`와 `units`는
각각 입력과 출력의 개수를 나타냅니다.

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
다음으로 `MyDense` 클래스를 인스턴스화하고
모델 파라미터에 접근해 보겠습니다.
:end_tab:

:begin_tab:`pytorch`
다음으로 `MyLinear` 클래스를 인스턴스화하고
모델 파라미터에 접근해 보겠습니다.
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

[**커스텀 층을 사용해 곧바로 순전파 계산을 수행할 수도 있습니다.**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

(**커스텀 층을 이용해 모델을 구성**)할 수도 있습니다.
일단 만들어 두면 내장 완전 연결 층처럼 사용할 수 있습니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## 요약

기본 층 클래스를 통해 커스텀 층을 설계할 수 있습니다. 이를 통해 라이브러리에 있는 어떤 기존 층과도 다르게 동작하는 유연한 새 층을 정의할 수 있습니다.
한 번 정의해 두면, 커스텀 층은 어떠한 상황과 아키텍처에서도 호출해 사용할 수 있습니다.
층은 지역 파라미터를 가질 수 있으며, 이러한 파라미터는 내장 함수를 통해 만들 수 있습니다.


## 연습문제

1. 입력을 받아 텐서 축약을 계산하는 층을 설계해 보세요.
   즉, $y_k = \sum_{i, j} W_{ijk} x_i x_j$를 반환합니다.
1. 데이터의 푸리에 계수 중 앞쪽 절반을 반환하는 층을 설계해 보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17993)
:end_tab:
