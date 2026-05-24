```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU
:label:`sec_use_gpu`

:numref:`tab_intro_decade`에서 저희는 지난 20년에 걸친
연산의 빠른 성장을 보여 드렸습니다.
요약하자면, GPU 성능은 2000년 이후로
10년마다 1000배씩 증가했습니다.
이는 큰 기회를 제공하지만, 동시에
그러한 성능에 대한 상당한 수요가 있었음을 시사하기도 합니다.


이 절에서는 이러한 계산 성능을 여러분의 연구에 어떻게
활용할 수 있는지 논의하기 시작합니다.
먼저 단일 GPU를 사용하는 것에서 시작해, 나중에는
여러 GPU와 (여러 GPU가 장착된) 여러 서버를 사용하는 방법으로 넘어갑니다.

구체적으로, 단일 NVIDIA GPU를 계산에 사용하는 방법을
논의해 보겠습니다.
먼저, NVIDIA GPU가 최소 한 개 설치되어 있는지 확인하세요.
그다음, [NVIDIA 드라이버와 CUDA](https://developer.nvidia.com/cuda-downloads)를 내려받고
안내에 따라 적절한 경로를 설정하세요.
이러한 준비가 끝나면 `nvidia-smi` 명령을 사용해
(**그래픽 카드 정보를 확인**)할 수 있습니다.

:begin_tab:`mxnet`
MXNet 텐서가 NumPy `ndarray`와 거의 동일해 보인다는 점을
눈치채셨을 것입니다.
그러나 몇 가지 결정적인 차이가 있습니다.
MXNet을 NumPy와 구별 짓는 핵심 기능 중 하나는
다양한 하드웨어 장치에 대한 지원입니다.

MXNet에서 모든 배열은 컨텍스트(context)를 가집니다.
지금까지 기본적으로 모든 변수와
관련된 계산은 CPU에 할당되었습니다.
다른 컨텍스트는 일반적으로 다양한 GPU일 수 있습니다.
여러 서버에 걸쳐 작업을 배포할 때에는
상황이 더욱 복잡해질 수 있습니다.
배열을 컨텍스트에 영리하게 할당함으로써
장치 간 데이터를 옮기는 데 드는 시간을 최소화할 수 있습니다.
예를 들어 GPU가 장착된 서버에서 신경망을 학습시킬 때
저희는 보통 모델의 파라미터가 GPU에 있는 것을 선호합니다.

다음으로, MXNet의 GPU 버전이 설치되어 있는지 확인해야 합니다.
MXNet의 CPU 버전이 이미 설치되어 있다면,
먼저 그것을 제거해야 합니다.
예를 들어 `pip uninstall mxnet` 명령을 사용한 다음,
여러분의 CUDA 버전에 맞는 MXNet 버전을 설치하세요.
CUDA 10.0이 설치되어 있다고 가정하면,
`pip install mxnet-cu100`을 통해 CUDA 10.0을 지원하는
MXNet 버전을 설치할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 모든 배열은 장치(device)를 가지며, 저희는 종종 이를 *컨텍스트*라고 부릅니다.
지금까지 기본적으로 모든 변수와
관련된 계산은 CPU에 할당되었습니다.
다른 컨텍스트는 일반적으로 다양한 GPU일 수 있습니다.
여러 서버에 걸쳐 작업을 배포할 때에는
상황이 더욱 복잡해질 수 있습니다.
배열을 컨텍스트에 영리하게 할당함으로써
장치 간 데이터를 옮기는 데 드는 시간을 최소화할 수 있습니다.
예를 들어 GPU가 장착된 서버에서 신경망을 학습시킬 때
저희는 보통 모델의 파라미터가 GPU에 있는 것을 선호합니다.
:end_tab:

이 절의 프로그램을 실행하려면 최소 두 개의 GPU가 필요합니다.
대부분의 데스크톱 컴퓨터에 대해서는 이것이 과한 사양일 수 있지만,
예를 들어 AWS EC2 다중 GPU 인스턴스를 사용하여
클라우드에서 쉽게 사용할 수 있다는 점에 유의하세요.
거의 모든 다른 절들은 다중 GPU를 *필요로 하지 않으며*, 여기서는 그저 서로 다른 장치 간 데이터 흐름을 보여 드리고자 할 뿐입니다.

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

## [**연산 장치**]

저장과 계산에 사용할 CPU와 GPU 같은 장치를 지정할 수 있습니다.
기본적으로 텐서는 주 메모리에 생성되고,
그다음 계산에 CPU가 사용됩니다.

:begin_tab:`mxnet`
MXNet에서 CPU와 GPU는 `cpu()`와 `gpu()`로 지칭할 수 있습니다.
`cpu()`(혹은 괄호 안의 어떤 정수든)는 모든 물리적 CPU와
메모리를 의미한다는 점에 유의해야 합니다.
이는 MXNet의 계산이 모든 CPU 코어를 사용하려 한다는 뜻입니다.
그러나 `gpu()`는 한 장의 카드와 그에 해당하는 메모리만을 나타냅니다.
GPU가 여러 개 있다면, `gpu(i)`를 사용해
$i^\textrm{th}$ GPU($i$는 0부터 시작)를 나타냅니다.
또한 `gpu(0)`과 `gpu()`는 같은 의미입니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 CPU와 GPU는 `torch.device('cpu')`와 `torch.device('cuda')`로 지칭할 수 있습니다.
`cpu` 장치는 모든 물리적 CPU와 메모리를 의미한다는 점에 유의해야 합니다.
이는 PyTorch의 계산이 모든 CPU 코어를 사용하려 한다는 뜻입니다.
그러나 `gpu` 장치는 한 장의 카드와 그에 해당하는 메모리만을 나타냅니다.
GPU가 여러 개 있다면, `torch.device(f'cuda:{i}')`를 사용해
$i^\textrm{th}$ GPU($i$는 0부터 시작)를 나타냅니다.
또한 `gpu:0`과 `gpu`는 같은 의미입니다.
:end_tab:

```{.python .input}
%%tab pytorch
def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def cpu():  #@save
    """Get the CPU device."""
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')
    if tab.selected('jax'):
        return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    """Get a GPU device."""
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')
    if tab.selected('jax'):
        return jax.devices('gpu')[i]

cpu(), gpu(), gpu(1)
```

(**사용 가능한 GPU의 개수를 조회**)할 수 있습니다.

```{.python .input}
%%tab pytorch
def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

num_gpus()
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def num_gpus():  #@save
    """Get the number of available GPUs."""
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))
    if tab.selected('jax'):
        try:
            return jax.device_count('gpu')
        except:
            return 0  # No GPU backend found

num_gpus()
```

이제 [**요청된 GPU가 존재하지 않더라도 코드를 실행할 수 있게 해 주는
편리한 함수 두 개를 정의합니다.**]

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## 텐서와 GPU

:begin_tab:`pytorch`
기본적으로 텐서는 CPU에 생성됩니다.
[**텐서가 위치한 장치를 조회**]할 수 있습니다.
:end_tab:

:begin_tab:`mxnet`
기본적으로 텐서는 CPU에 생성됩니다.
[**텐서가 위치한 장치를 조회**]할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow, jax`
기본적으로 텐서는 GPU/TPU가 사용 가능하다면 GPU/TPU에 생성되고,
그렇지 않다면 CPU가 사용됩니다.
[**텐서가 위치한 장치를 조회**]할 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
%%tab jax
x = jnp.array([1, 2, 3])
x.device()
```

여러 항에 대해 연산을 수행하려 할 때마다,
이들이 동일한 장치에 있어야 한다는 점에 유의하는 것이 중요합니다.
예를 들어 두 텐서를 더한다면,
두 인자가 동일한 장치에 있어야 함을
반드시 확인해야 합니다(그렇지 않으면 프레임워크가
결과를 어디에 저장해야 할지, 심지어 계산을 어디서 수행할지조차
결정하는 방법을 알 수 없습니다).

### GPU에 저장하기

[**텐서를 GPU에 저장**]하는 방법은 여러 가지가 있습니다.
예를 들어, 텐서를 생성할 때 저장 장치를 지정할 수 있습니다.
다음으로, 첫 번째 `gpu`에 텐서 변수 `X`를 생성합니다.
GPU에서 생성된 텐서는 그 GPU의 메모리만 사용합니다.
GPU 메모리 사용량을 보려면 `nvidia-smi` 명령을 사용할 수 있습니다.
일반적으로 GPU 메모리 한도를 초과하는 데이터를 생성하지 않도록 주의해야 합니다.

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
%%tab jax
# By default JAX puts arrays to GPUs or TPUs if available
X = jax.device_put(jnp.ones((2, 3)), try_gpu())
X
```

GPU가 최소 두 개 있다고 가정하면, 다음 코드는 (**두 번째 GPU에 무작위 텐서 `Y`를 생성**)합니다.

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
%%tab jax
Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                   try_gpu(1))
Y
```

### 복사

[**`X + Y`를 계산하고 싶다면,
이 연산을 어디서 수행할지 결정해야 합니다.**]
예를 들어, :numref:`fig_copyto`에 나타난 것처럼,
`X`를 두 번째 GPU로 옮긴 다음 거기서 연산을 수행할 수 있습니다.
`X`와 `Y`를 단순히 더하지 *마세요*. 그렇게 하면 예외가 발생할 것입니다.
런타임 엔진은 무엇을 해야 할지 모를 것입니다.
같은 장치에 있는 데이터를 찾을 수 없어 실패합니다.
`Y`가 두 번째 GPU에 있으므로,
둘을 더하기 전에 `X`를 그곳으로 옮겨야 합니다.

![동일 장치에서 연산을 수행하기 위해 데이터를 복사한다.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

```{.python .input}
%%tab jax
Z = jax.device_put(X, try_gpu(1))
print(X)
print(Z)
```

이제 [**데이터(`Z`와 `Y` 모두)가 동일한 GPU에 있으므로, 이들을 더할 수 있습니다.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
여러분의 변수 `Z`가 이미 두 번째 GPU에 있다고 상상해 보세요.
그래도 `Z.copyto(gpu(1))`을 호출하면 어떻게 될까요?
그 변수가 이미 원하는 장치에 있더라도, 복사본을 만들고
새로운 메모리를 할당합니다.
저희 코드가 실행되는 환경에 따라, 두 변수가 이미 동일한 장치에
있을 때도 있습니다.
따라서 변수들이 현재 서로 다른 장치에 있을 때에만
복사하기를 원합니다.
이러한 경우에는 `as_in_ctx`를 호출할 수 있습니다.
변수가 이미 지정된 장치에 있다면 이는 아무 동작도 하지 않습니다.
복사를 만드는 것을 특별히 원하는 경우가 아니라면,
`as_in_ctx`가 선택할 메서드입니다.
:end_tab:

:begin_tab:`pytorch`
그런데 변수 `Z`가 이미 두 번째 GPU에 있다면 어떨까요?
그래도 `Z.cuda(1)`을 호출하면 어떻게 될까요?
복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환합니다.
:end_tab:

:begin_tab:`tensorflow`
여러분의 변수 `Z`가 이미 두 번째 GPU에 있다고 상상해 보세요.
같은 장치 스코프 아래에서 그래도 `Z2 = Z`를 호출하면 어떻게 될까요?
복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환합니다.
:end_tab:

:begin_tab:`jax`
여러분의 변수 `Z`가 이미 두 번째 GPU에 있다고 상상해 보세요.
같은 장치 스코프 아래에서 그래도 `Z2 = Z`를 호출하면 어떻게 될까요?
복사본을 만들고 새 메모리를 할당하는 대신 `Z`를 반환합니다.
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

```{.python .input}
%%tab jax
Z2 = jax.device_put(Z, try_gpu(1))
Z2 is Z
```

### 부수적인 주의 사항

사람들은 GPU가 빠를 것이라고 기대해서 머신러닝에 GPU를 사용합니다.
그러나 장치 간 변수를 옮기는 것은 느립니다(계산보다 훨씬 느립니다).
따라서 저희가 여러분이 느린 무언가를 하도록 두기 전에,
여러분이 그것을 하고 싶다는 것을 100% 확신하기를 바랍니다.
딥러닝 프레임워크가 충돌 없이 복사를 자동으로 그냥 해 버린다면,
여러분은 자신이 느린 코드를 작성했다는 사실을 깨닫지 못할 수도 있습니다.

데이터 전송은 느릴 뿐만 아니라 병렬화를 훨씬 더 어렵게 만드는데,
더 많은 연산을 진행하기 전에 데이터가 전송될(혹은 정확히는 수신될) 때까지
기다려야 하기 때문입니다.
바로 이래서 복사 연산을 매우 신중하게 다뤄야 하는 것입니다.
경험에 비춰 봤을 때, 많은 작은 연산은 하나의 큰 연산보다 훨씬 더 나쁩니다.
나아가, 자신이 무엇을 하고 있는지 잘 알고 있는 경우가 아니라면
한 번에 여러 연산을 수행하는 편이, 코드 곳곳에 흩어져 있는 많은 단일 연산보다
훨씬 낫습니다.
이는 한 장치가 다른 장치를 기다려야 다른 일을 할 수 있는 상황에서는
그런 연산들이 막힐 수 있기 때문입니다.
전화로 미리 주문하고 준비되었을 때 알게 되는 것이 아니라,
줄에 서서 커피를 주문하는 것과 약간 비슷합니다.

마지막으로, 텐서를 출력하거나 텐서를 NumPy 형식으로 변환할 때,
데이터가 주 메모리에 없다면, 프레임워크는 먼저 그것을 주 메모리로
복사할 것이며, 이는 추가적인 전송 오버헤드를 초래합니다.
설상가상으로, 이제 이것은 모든 것을 파이썬이 끝날 때까지 기다리게 만드는
악명 높은 글로벌 인터프리터 락(global interpreter lock)의 영향을 받게 됩니다.


## [**신경망과 GPU**]

마찬가지로, 신경망 모델도 장치를 지정할 수 있습니다.
다음 코드는 모델 파라미터를 GPU에 둡니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(1)])

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (10,))  # Dummy input
params = net.init(key2, x)  # Initialization call
```

다음 장들에서는 GPU에서 모델을 실행하는 예제를 훨씬 더 많이 보게 될 것입니다.
모델들이 다소 더 계산 집약적이 되기 때문입니다.

예를 들어, 입력이 GPU에 있는 텐서일 때, 모델은 같은 GPU에서 결과를 계산할 것입니다.

```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

(**모델 파라미터가 같은 GPU에 저장되어 있는지 확인**)해 보겠습니다.

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
%%tab jax
print(jax.tree_util.tree_map(lambda x: x.device(), params))
```

트레이너가 GPU를 지원하도록 합시다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch
```

요컨대, 모든 데이터와 파라미터가 동일한 장치에 있는 한, 모델을 효율적으로 학습시킬 수 있습니다. 다음 장들에서 이러한 예제를 여럿 보게 될 것입니다.

## 요약

저장과 계산을 위해 CPU나 GPU 같은 장치를 지정할 수 있습니다.
  기본적으로 데이터는 주 메모리에 생성되고
  계산에는 CPU가 사용됩니다.
딥러닝 프레임워크는 계산에 사용되는 모든 입력 데이터가
  CPU든 같은 GPU든 동일한 장치에 있을 것을 요구합니다.
주의 없이 데이터를 옮기면 상당한 성능을 잃을 수 있습니다.
  전형적인 실수는 다음과 같습니다. 매 미니배치마다 GPU에서 손실을 계산하여
  명령줄에서 사용자에게 다시 보고하는 것(혹은 NumPy `ndarray`에 로깅하는 것)은
  글로벌 인터프리터 락을 발동시켜 모든 GPU를 멎게 만듭니다.
  로깅을 위한 메모리를 GPU 안에 할당하고, 더 큰 로그만 옮기는 편이 훨씬 낫습니다.

## 연습문제

1. 큰 행렬의 곱셈 같은 더 큰 계산 작업을 시도해 보고,
   CPU와 GPU 간 속도 차이를 살펴보세요.
   계산량이 적은 작업은 어떤가요?
1. GPU에서 모델 파라미터는 어떻게 읽고 써야 할까요?
1. $100 \times 100$ 행렬의 행렬 대 행렬 곱셈을 1000회 계산하는 데 걸리는
   시간을 측정하고, 한 번에 하나의 결과씩 출력 행렬의 프로베니우스(Frobenius)
   노름을 기록해 보세요. GPU에 로그를 유지하고 최종 결과만 옮기는 것과 비교해 보세요.
1. 두 GPU에서 동시에 두 번의 행렬 대 행렬 곱셈을 수행하는 데
   얼마나 시간이 걸리는지 측정해 보세요. 하나의 GPU에서 순차적으로 계산하는 것과
   비교해 보세요. 힌트: 거의 선형적인 스케일링을 볼 수 있어야 합니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17995)
:end_tab:
