```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 파일 입출력

지금까지 저희는 데이터를 처리하는 방법과
딥러닝 모델을 만들고, 학습시키고, 시험하는 방법을 논의해 왔습니다.
그러나 언젠가는 학습된 모델에 대해 만족스러워져
다양한 맥락에서(어쩌면 배포 환경에서 예측을 수행하기 위해서라도)
나중에 사용할 수 있도록 결과를 저장하고 싶어지기를 바랍니다.
또한 긴 학습 프로세스를 실행할 때에는,
서버의 전원 코드에 발이 걸려 며칠치의 계산을 잃어버리지 않도록
중간 결과를 주기적으로 저장(체크포인팅)하는 것이 모범 사례입니다.
따라서 이제는 개별 가중치 벡터와 모델 전체 양쪽 모두를
적재하고 저장하는 방법을 배울 때입니다.
이 절에서는 두 가지 문제를 모두 다룹니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**텐서 적재 및 저장**)

개별 텐서에 대해서는, `load`와 `save` 함수를
직접 호출하여 각각 읽고 쓸 수 있습니다.
두 함수 모두 이름을 제공하도록 요구하며,
`save`는 저장할 변수도 입력으로 요구합니다.

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

이제 저장된 파일에서 데이터를 다시 메모리로 읽어들일 수 있습니다.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

[**텐서들의 리스트를 저장하고 다시 메모리로 읽어들일 수 있습니다.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

심지어 [**문자열을 텐서로 매핑하는 사전을 쓰고 읽을 수도 있습니다.**]
이는 모델 내의 모든 가중치를 읽거나 쓰고 싶을 때 편리합니다.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**모델 파라미터 적재 및 저장**]

개별 가중치 벡터(혹은 다른 텐서)를 저장하는 것은 유용하지만,
모델 전체를 저장(이후 적재)하고 싶다면 매우 지루해집니다.
결국 신경망 곳곳에 수백 개의 파라미터 그룹이
흩뿌려져 있을 수 있기 때문입니다.
이러한 이유로 딥러닝 프레임워크는 신경망 전체를 적재하고
저장하기 위한 내장 기능을 제공합니다.
주의할 중요한 세부 사항은 이것이 모델 전체가 아니라
모델 *파라미터*를 저장한다는 점입니다.
예를 들어, 3층 MLP를 가지고 있다면,
아키텍처는 따로 명시해 주어야 합니다.
이러한 이유는 모델 자체가 임의의 코드를 담을 수 있어서,
자연스럽게 직렬화될 수 없기 때문입니다.
따라서 모델을 복원하려면 코드에서 아키텍처를 생성한 다음
디스크에서 파라미터를 적재해야 합니다.
(**먼저 익숙한 MLP부터 시작해 보겠습니다.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

다음으로, [**모델의 파라미터를 "mlp.params"라는 이름의 파일로 저장**]합니다.

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

모델을 복원하기 위해, 원본 MLP 모델의 복제본을 인스턴스화합니다.
모델 파라미터를 무작위로 초기화하는 대신,
[**파일에 저장된 파라미터를 직접 읽어들입니다**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

두 인스턴스가 동일한 모델 파라미터를 가지므로,
동일한 입력 `X`에 대한 계산 결과가 동일해야 합니다.
이를 확인해 보겠습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## 요약

`save`와 `load` 함수는 텐서 객체에 대해 파일 입출력을 수행하는 데 사용할 수 있습니다.
파라미터 사전을 통해 신경망의 전체 파라미터 집합을 저장하고 적재할 수 있습니다.
아키텍처의 저장은 파라미터가 아니라 코드를 통해 이루어져야 합니다.

## 연습문제

1. 학습된 모델을 다른 장치에 배포할 필요가 없더라도, 모델 파라미터를 저장하는 것의 실용적 이점은 무엇인가요?
1. 신경망의 일부만 재사용하여 다른 아키텍처를 갖는 신경망에 통합하고 싶다고 가정해 보세요. 예를 들어, 이전 신경망의 처음 두 층을 새 신경망에서 사용하려면 어떻게 해야 할까요?
1. 신경망 아키텍처와 파라미터를 함께 저장하려면 어떻게 해야 할까요? 아키텍처에는 어떤 제약을 두시겠습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17994)
:end_tab: