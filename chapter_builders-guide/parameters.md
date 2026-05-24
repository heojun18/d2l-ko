```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 파라미터 관리

아키텍처를 선택하고 하이퍼파라미터를 설정한 다음에는,
저희 손실 함수를 최소화하는 파라미터 값을 찾는 것이 목표인
학습 루프로 진행하게 됩니다.
학습이 끝나면, 앞으로의 예측을 수행하기 위해
이 파라미터들이 필요할 것입니다.
또한 다른 맥락에서 재사용하기 위해서,
다른 소프트웨어에서 실행할 수 있도록 모델을 디스크에
저장하기 위해서, 혹은 과학적 이해를 얻고 싶은 마음에
살펴보기 위해서 파라미터를 추출하고 싶어질 때도 있습니다.

대부분의 경우, 저희는 무거운 일들을 해 주는 딥러닝 프레임워크에 의존해
파라미터가 어떻게 선언되고 조작되는지에 대한 세세한
세부 사항은 무시할 수 있습니다.
그러나 표준 층들을 쌓아 올린 아키텍처에서 벗어날 때에는,
때때로 파라미터를 선언하고 조작하는 자세한 일들로
파고들 필요가 있습니다.
이 절에서는 다음 내용을 다룹니다.

* 디버깅, 진단, 시각화를 위한 파라미터 접근.
* 서로 다른 모델 구성 요소 사이에서의 파라미터 공유.

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

(**먼저 은닉 층이 하나 있는 MLP에 초점을 맞추는 것으로 시작합니다.**)

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
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

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

## [**파라미터 접근**]
:label:`subsec_param-access`

이미 알고 계신 모델로부터 파라미터에 접근하는 방법부터
시작해 보겠습니다.

:begin_tab:`mxnet, pytorch, tensorflow`
모델이 `Sequential` 클래스를 통해 정의될 때,
저희는 모델을 마치 리스트인 것처럼 인덱싱하여
먼저 어떤 층이든 접근할 수 있습니다.
각 층의 파라미터는 편리하게도 해당 층의 속성에 위치합니다.
:end_tab:

:begin_tab:`jax`
앞에서 정의한 모델들에서 보셨겠지만,
Flax와 JAX는 모델과 파라미터를 분리합니다.
모델이 `Sequential` 클래스를 통해 정의될 때,
먼저 파라미터 사전을 생성하기 위해 신경망을 초기화해야 합니다.
이 사전의 키를 통해 어떤 층의 파라미터든 접근할 수 있습니다.
:end_tab:

두 번째 완전 연결 층의 파라미터는 다음과 같이 살펴볼 수 있습니다.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

이 완전 연결 층이 각각 해당 층의 가중치와 편향에 대응되는
두 개의 파라미터를 가지고 있음을 확인할 수 있습니다.


### [**특정 파라미터**]

각 파라미터는 파라미터 클래스의 인스턴스로
표현된다는 점에 유의하세요.
파라미터로 유용한 일을 하려면, 먼저 그 밑에 있는
수치 값에 접근해야 합니다.
이를 위한 방법은 여러 가지가 있습니다.
어떤 것은 더 간단하고, 어떤 것은 더 일반적입니다.
다음 코드는 두 번째 신경망 층에서 편향을 추출하는데,
이는 파라미터 클래스 인스턴스를 반환하며,
그 파라미터의 값에도 한 단계 더 들어가 접근합니다.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
파라미터는 값, 그래디언트, 추가 정보를 담고 있는
복합 객체입니다.
바로 그래서 값을 명시적으로 요청해야 하는 것입니다.

값 외에도 각 파라미터는 그래디언트에도 접근할 수 있게 해 줍니다. 아직 이 신경망에 대해 역전파를 호출하지 않았으므로, 초기 상태에 있습니다.
:end_tab:

:begin_tab:`jax`
다른 프레임워크들과 달리, JAX는 신경망 파라미터에 대한 그래디언트를
추적해 두지 않으며, 대신 파라미터와 신경망이 분리되어 있습니다.
사용자가 자신의 계산을 파이썬 함수로 표현하고,
같은 목적으로 `grad` 변환을 사용할 수 있게 해 줍니다.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**모든 파라미터를 한꺼번에**]

모든 파라미터에 대해 연산을 수행해야 할 때,
하나씩 접근하는 것은 지루해질 수 있습니다.
중첩 모듈 같은 더 복잡한 모듈을 다룰 때에는
상황이 특히 다루기 어려워질 수 있는데,
각 하위 모듈의 파라미터를 추출하려면 전체 트리를
재귀적으로 순회해야 하기 때문입니다. 아래에서 모든 층의 파라미터에 접근하는 방법을 보여 드립니다.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**묶인 파라미터**]

종종 저희는 여러 층에 걸쳐 파라미터를 공유하고 싶을 때가 있습니다.
이를 우아하게 수행하는 방법을 살펴보겠습니다.
다음에서는 완전 연결 층을 하나 할당한 다음,
그 파라미터를 가져다가 다른 층의 파라미터로 설정하는 데
사용합니다.
여기서는 파라미터에 접근하기 전에 순전파
`net(X)`를 실행해야 합니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

이 예제는 두 번째와 세 번째 층의 파라미터가
묶여 있다는 것을 보여 줍니다.
이들은 단지 같은 값을 가질 뿐 아니라,
완전히 동일한 텐서로 표현됩니다.
따라서 한쪽 파라미터를 바꾸면 다른 쪽도 함께 바뀝니다.

:begin_tab:`mxnet, pytorch, tensorflow`
파라미터가 묶여 있을 때 그래디언트는 어떻게 되는지
궁금하실 수도 있습니다.
모델 파라미터가 그래디언트를 담고 있으므로,
역전파 도중에 두 번째 은닉 층의 그래디언트와
세 번째 은닉 층의 그래디언트가 함께 합산됩니다.
:end_tab:


## 요약

저희에게는 모델 파라미터에 접근하고 묶는 여러 가지 방법이 있습니다.


## 연습문제

1. :numref:`sec_model_construction`에서 정의한 `NestMLP` 모델을 사용하여 다양한 층의 파라미터에 접근해 보세요.
1. 공유 파라미터 층을 포함하는 MLP를 구성하고 학습시켜 보세요. 학습 과정 동안 각 층의 모델 파라미터와 그래디언트를 관찰해 보세요.
1. 파라미터를 공유하는 것이 왜 좋은 아이디어인가요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17990)
:end_tab:
