```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 지연 초기화
:label:`sec_lazy_init`

지금까지는 신경망을 설정할 때 저희가 다소 엉성하게 처리했음에도
어떻게든 잘 넘어간 것처럼 보일 수 있습니다.
구체적으로, 저희는 다음과 같이 그다지 직관적이지 않은,
잘 동작할 것 같지 않은 일들을 해 왔습니다.

* 입력 차원을 지정하지 않은 채 신경망 아키텍처를 정의했습니다.
* 이전 층의 출력 차원을 지정하지 않은 채 층을 추가했습니다.
* 모델에 파라미터가 몇 개나 들어 있어야 할지 결정하기에 충분한
  정보를 제공하지도 않은 상태에서 이러한 파라미터들을 "초기화"하기까지 했습니다.

저희 코드가 도대체 어떻게 동작하는지 의아하실 수도 있습니다.
어쨌든 딥러닝 프레임워크가 신경망의 입력 차원이
얼마일지 알 방법이 없으니까요.
여기서의 비법은 프레임워크가 *초기화를 미루어*,
저희가 모델에 데이터를 처음 흘려보낼 때까지 기다렸다가
그때 각 층의 크기를 즉석에서 추론한다는 것입니다.


나중에 합성곱 신경망을 다룰 때에는,
입력 차원(예를 들어 이미지의 해상도)이
뒤따르는 각 층의 차원에 영향을 주기 때문에
이 기법이 더욱 편리해질 것입니다.
따라서 코드를 작성하는 시점에 차원의 값을 알 필요 없이
파라미터를 설정할 수 있는 능력은,
저희 모델을 명세하고 그 뒤에 수정하는 작업을
크게 단순화해 줄 수 있습니다.
다음으로, 초기화의 작동 방식을 더 깊이 살펴보겠습니다.

```{.python .input}
%%tab mxnet
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

먼저 MLP를 하나 인스턴스화해 보겠습니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

이 시점에서, 입력 차원이 여전히 알려지지 않았기 때문에
신경망은 입력 층 가중치의 차원을 도저히 알 수 없습니다.

:begin_tab:`mxnet, pytorch, tensorflow`
따라서 프레임워크는 아직 어떤 파라미터도 초기화하지 않았습니다.
아래에서 파라미터에 접근을 시도해 보면서 이를 확인해 보겠습니다.
:end_tab:

:begin_tab:`jax`
:numref:`subsec_param-access`에서 언급했듯이, Jax와 Flax에서는
파라미터와 신경망 정의가 분리되어 있고, 사용자가 둘 다 수동으로 다룹니다.
Flax 모델은 상태가 없기 때문에 `parameters` 속성이 존재하지 않습니다.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
파라미터 객체는 존재하지만, 각 층의 입력 차원은 -1로
표시되어 있다는 점에 유의하세요.
MXNet은 파라미터 차원이 아직 알려지지 않았음을 나타내기 위해
특수값 -1을 사용합니다.
이 시점에 `net[0].weight.data()`에 접근을 시도하면,
파라미터에 접근하기 전에 신경망이 먼저 초기화되어야 한다는
런타임 오류가 발생합니다.
이제 `initialize` 메서드를 통해 파라미터를 초기화하려 하면
어떤 일이 벌어지는지 살펴보겠습니다.
:end_tab:

:begin_tab:`tensorflow`
각 층 객체는 존재하지만, 가중치는 비어 있다는 점에 유의하세요.
가중치가 아직 초기화되지 않았으므로 `net.get_weights()`를 사용하면
오류가 발생합니다.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
보시다시피, 아무것도 바뀌지 않았습니다.
입력 차원을 모를 때, initialize를 호출해도
실제로는 파라미터를 초기화하지 않습니다.
대신 이 호출은 저희가 파라미터를 (선택적으로 어떤 분포에 따라)
초기화하고 싶다는 것을 MXNet에 등록해 두는 역할만 합니다.
:end_tab:

다음으로, 프레임워크가 마침내 파라미터를 초기화하도록
신경망에 데이터를 흘려보내 보겠습니다.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

입력 차원이 20이라는 것을 알게 되자마자,
프레임워크는 20이라는 값을 대입해
첫 번째 층의 가중치 행렬 모양을 파악할 수 있습니다.
첫 번째 층의 모양을 알아낸 뒤, 프레임워크는 두 번째 층으로 넘어가고,
이런 식으로 계산 그래프를 따라 모든 모양이 알려질 때까지 진행합니다.
이 경우에는 첫 번째 층만 지연 초기화가 필요하지만,
프레임워크는 순차적으로 초기화한다는 점에 유의하세요.
모든 파라미터의 모양이 알려지면, 프레임워크는 마침내
파라미터를 초기화할 수 있습니다.

:begin_tab:`pytorch`
다음 메서드는 더미 입력을 신경망에 흘려보내
드라이런(dry run)을 수행하여 모든 파라미터의 모양을 추론한 뒤,
이어서 파라미터를 초기화합니다.
기본 무작위 초기화를 원하지 않을 때 나중에 사용하게 될 것입니다.
:end_tab:

:begin_tab:`jax`
Flax에서 파라미터 초기화는 항상 수동으로 이루어지며 사용자가 처리합니다.
다음 메서드는 더미 입력과 키 사전을 인자로 받습니다.
이 키 사전에는 모델 파라미터 초기화를 위한 rngs와
드롭아웃 층이 있는 모델을 위한 드롭아웃 마스크를 생성하기 위한
드롭아웃 rng가 들어 있습니다. 드롭아웃에 대해서는
:numref:`sec_dropout`에서 더 자세히 다룰 것입니다.
최종적으로 이 메서드는 모델을 초기화하고 파라미터를 반환합니다.
저희는 이전 절들에서도 이를 내부적으로 사용해 왔습니다.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## 요약

지연 초기화는 편리할 수 있는데, 프레임워크가 파라미터의 모양을 자동으로 추론하도록 해 주어 아키텍처 수정을 쉽게 만들고 흔한 오류 원인 하나를 없애 줍니다.
모델에 데이터를 흘려보내 프레임워크가 마침내 파라미터를 초기화하도록 만들 수 있습니다.


## 연습문제

1. 첫 번째 층에는 입력 차원을 지정하지만 이후 층들에는 지정하지 않으면 어떻게 될까요? 즉시 초기화가 이루어지나요?
1. 일치하지 않는 차원을 지정하면 어떻게 될까요?
1. 다양한 차원의 입력이 있다면 어떻게 해야 할까요? 힌트: 파라미터 묶기(parameter tying)를 살펴보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17992)
:end_tab:
