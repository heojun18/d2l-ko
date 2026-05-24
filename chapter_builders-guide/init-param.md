```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 파라미터 초기화

이제 파라미터에 접근하는 방법을 알게 되었으니,
파라미터를 적절히 초기화하는 방법을 살펴보겠습니다.
:numref:`sec_numerical_stability`에서 적절한 초기화의 필요성을 논의했습니다.
딥러닝 프레임워크는 자신의 층들에 기본적인 무작위 초기화를 제공합니다.
그러나 저희는 종종 다양한 다른 규약에 따라
가중치를 초기화하고 싶어 합니다. 프레임워크는 가장 흔히
사용되는 규약을 제공하며, 커스텀 초기화기를 만드는 것도 허용합니다.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

:begin_tab:`mxnet`
기본적으로 MXNet은 균등 분포 $U(-0.07, 0.07)$에서 무작위로
표본을 추출해 가중치 파라미터를 초기화하고,
편향 파라미터는 0으로 비웁니다.
MXNet의 `init` 모듈은 다양한 사전 정의된 초기화 방법을 제공합니다.
:end_tab:

:begin_tab:`pytorch`
기본적으로 PyTorch는 입력과 출력 차원에 따라 계산된 범위에서
표본을 추출하는 방식으로 가중치와 편향 행렬을 균등하게 초기화합니다.
PyTorch의 `nn.init` 모듈은 다양한 사전 정의된 초기화 방법을 제공합니다.
:end_tab:

:begin_tab:`tensorflow`
기본적으로 Keras는 입력과 출력 차원에 따라 계산된 범위에서
표본을 추출하는 방식으로 가중치 행렬을 균등하게 초기화하며, 편향 파라미터는 모두 0으로 설정됩니다.
TensorFlow는 루트 모듈과 `keras.initializers` 모듈 양쪽에서 다양한 초기화 방법을 제공합니다.
:end_tab:

:begin_tab:`jax`
기본적으로 Flax는 `jax.nn.initializers.lecun_normal`을 사용해 가중치를 초기화합니다.
즉, 0을 중심으로 하고 표준편차를 $1 / \textrm{fan}_{\textrm{in}}$의 제곱근으로
설정한 절단 정규 분포(여기서 `fan_in`은 가중치 텐서의 입력 단위 수)에서
표본을 추출합니다. 편향 파라미터는 모두 0으로 설정됩니다.
Jax의 `nn.initializers` 모듈은 다양한 사전 정의된 초기화 방법을 제공합니다.
:end_tab:

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**내장 초기화**]

먼저 내장 초기화기를 호출해 보는 것으로 시작해 보겠습니다.
아래 코드는 모든 가중치 파라미터를 표준편차 0.01인
가우시안 무작위 변수로 초기화하며, 편향 파라미터는 0으로 비웁니다.

```{.python .input}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

모든 파라미터를 주어진 상수 값(예를 들어 1)으로
초기화할 수도 있습니다.

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**특정 블록에 대해 서로 다른 초기화기를 적용할 수도 있습니다.**]
예를 들어 아래에서는 첫 번째 층은 Xavier 초기화기로 초기화하고,
두 번째 층은 상수 값 42로 초기화합니다.

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**커스텀 초기화**]

때로는 저희가 필요로 하는 초기화 방법이
딥러닝 프레임워크에서 제공되지 않을 때도 있습니다.
아래 예제에서는 임의의 가중치 파라미터 $w$에 대해
다음과 같이 별난 분포를 사용하는 초기화기를 정의합니다.

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
여기서는 `Initializer` 클래스의 하위 클래스를 정의합니다.
일반적으로는 텐서 인자(`data`)를 받아
원하는 초기 값을 할당해 주는 `_init_weight` 함수만 구현하면 됩니다.
:end_tab:

:begin_tab:`pytorch`
다시 한 번, `net`에 적용할 `my_init` 함수를 구현합니다.
:end_tab:

:begin_tab:`tensorflow`
여기서는 `Initializer`의 하위 클래스를 정의하고,
주어진 모양과 자료형에 대해 원하는 텐서를 반환하는
`__call__` 함수를 구현합니다.
:end_tab:

:begin_tab:`jax`
Jax 초기화 함수는 인자로 `PRNGKey`, `shape`, `dtype`을 받습니다.
여기서는 주어진 모양과 자료형에 대해 원하는 텐서를 반환하는
`my_init` 함수를 구현합니다.
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
저희에게는 언제든 파라미터를 직접 설정하는
선택지가 있다는 점에 유의하세요.
:end_tab:

:begin_tab:`jax`
JAX와 Flax에서 파라미터를 초기화하면, 반환되는 파라미터 사전은
`flax.core.frozen_dict.FrozenDict` 타입을 가집니다. Jax 생태계에서는
배열의 값을 직접 변경하는 것이 권장되지 않으며, 따라서 자료형들은
대체로 불변입니다. 변경을 가하려면 `params.unfreeze()`를 사용할 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## 요약

내장 초기화기와 커스텀 초기화기를 사용해 파라미터를 초기화할 수 있습니다.

## 연습문제

더 많은 내장 초기화기에 대해 온라인 문서를 찾아보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17991)
:end_tab:
