# 컴파일러와 인터프리터
:label:`sec_hybridize`

지금까지 이 책은 `print`, `+`, `if` 같은 문장을 사용해 프로그램의 상태를 변경하는 명령형 프로그래밍에 초점을 맞춰 왔습니다. 다음의 간단한 명령형 프로그램 예제를 살펴봅시다.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python은 *인터프리트 언어*입니다. 위의 `fancy_func` 함수를 평가할 때 함수 본문을 구성하는 연산을 *순차적으로* 수행합니다. 즉, `e = add(a, b)`를 평가한 다음 결과를 변수 `e`로 저장하여 프로그램의 상태를 변경합니다. 다음 두 문장 `f = add(c, d)`와 `g = add(e, f)` 또한 비슷하게 실행되어 덧셈을 수행하고 결과를 변수로 저장합니다. :numref:`fig_compute_graph`는 데이터의 흐름을 보여줍니다.

![명령형 프로그램에서의 데이터 흐름.](../img/computegraph.svg)
:label:`fig_compute_graph`

명령형 프로그래밍은 편리하지만 비효율적일 수 있습니다. 한편으로, `add` 함수가 `fancy_func` 전체에서 반복적으로 호출되더라도 Python은 세 번의 함수 호출을 개별적으로 실행할 것입니다. 이들이 예를 들어 GPU(혹은 심지어 여러 GPU)에서 실행된다면, Python 인터프리터에서 발생하는 오버헤드는 압도적이 될 수 있습니다. 게다가, `fancy_func`의 모든 문장이 실행될 때까지 `e`와 `f`의 변수 값을 저장해야 할 것입니다. 이는 `e = add(a, b)`와 `f = add(c, d)` 문장이 실행된 후 변수 `e`와 `f`가 프로그램의 다른 부분에서 사용될지를 저희가 모르기 때문입니다.

## 심볼릭 프로그래밍

대안인 *심볼릭 프로그래밍*을 고려해 봅시다. 여기서 계산은 대개 프로세스가 완전히 정의된 후에만 수행됩니다. 이 전략은 Theano와 TensorFlow를 포함한 여러 딥러닝 프레임워크에서 사용됩니다(후자는 명령형 확장을 도입했습니다). 일반적으로 다음 단계를 포함합니다.

1. 실행할 연산을 정의합니다.
1. 연산을 실행 가능한 프로그램으로 컴파일합니다.
1. 필요한 입력을 제공하고 컴파일된 프로그램을 실행을 위해 호출합니다.

이는 상당한 양의 최적화를 가능하게 합니다. 첫째, 많은 경우 Python 인터프리터를 건너뛸 수 있어, CPU의 단일 Python 스레드와 짝지어진 여러 빠른 GPU에서 상당해질 수 있는 성능 병목을 제거할 수 있습니다. 
둘째, 컴파일러는 위의 코드를 `print((1 + 2) + (3 + 4))` 또는 심지어 `print(10)`으로 최적화하고 재작성할 수 있습니다. 이는 컴파일러가 코드를 머신 명령어로 변환하기 전에 전체 코드를 볼 수 있기 때문에 가능합니다. 예를 들어, 변수가 더 이상 필요하지 않을 때마다 메모리를 해제(또는 절대 할당하지 않음)할 수 있습니다. 또는 코드를 동등한 조각으로 완전히 변환할 수 있습니다.
더 잘 이해하기 위해, 아래의 명령형 프로그래밍 시뮬레이션을 살펴보십시오(결국 Python이니까요).

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

명령형(인터프리트) 프로그래밍과 심볼릭 프로그래밍의 차이는 다음과 같습니다.

* 명령형 프로그래밍이 더 쉽습니다. Python에서 명령형 프로그래밍을 사용할 때, 대부분의 코드는 직관적이고 작성하기 쉽습니다. 또한 명령형 프로그래밍 코드는 디버깅하기 더 쉽습니다. 이는 모든 관련 중간 변수 값을 얻고 출력하거나 Python의 내장 디버깅 도구를 사용하기 더 쉽기 때문입니다.
* 심볼릭 프로그래밍은 더 효율적이고 이식하기 쉽습니다. 심볼릭 프로그래밍은 컴파일 중에 코드를 최적화하는 것을 더 쉽게 만들면서, 프로그램을 Python과 독립적인 형식으로 이식할 수 있는 능력도 가집니다. 이는 프로그램이 Python이 아닌 환경에서 실행될 수 있도록 하여 Python 인터프리터와 관련된 모든 잠재적 성능 문제를 피할 수 있게 합니다.


## 하이브리드 프로그래밍

역사적으로 대부분의 딥러닝 프레임워크는 명령형 또는 심볼릭 접근법 중 하나를 선택합니다. 예를 들어 Theano, TensorFlow(전자에서 영감을 받은), Keras, CNTK는 모델을 심볼릭하게 정식화합니다. 반대로 Chainer와 PyTorch는 명령형 접근을 취합니다. 명령형 모드는 TensorFlow 2.0과 이후 개정판의 Keras에 추가되었습니다.

:begin_tab:`mxnet`
Gluon을 설계할 때, 개발자들은 두 프로그래밍 패러다임의 이점을 결합할 수 있는지 고려했습니다. 이는 사용자가 순수 명령형 프로그래밍으로 개발하고 디버깅하면서, 제품 수준의 컴퓨팅 성능과 배포가 필요할 때 대부분의 프로그램을 심볼릭 프로그램으로 변환하여 실행할 수 있는 능력을 가진 하이브리드 모델로 이어졌습니다.

실제로 이는 `HybridBlock` 또는 `HybridSequential` 클래스를 사용하여 모델을 구축한다는 의미입니다. 기본적으로 둘 중 어느 것도 명령형 프로그래밍에서 `Block` 또는 `Sequential` 클래스가 실행되는 것과 같은 방식으로 실행됩니다. 
`HybridSequential` 클래스는 `HybridBlock`의 하위 클래스입니다(`Sequential`이 `Block`의 하위 클래스인 것처럼). `hybridize` 함수가 호출되면, Gluon은 모델을 심볼릭 프로그래밍에서 사용되는 형태로 컴파일합니다. 이를 통해 모델 구현 방식을 희생하지 않고도 계산 집약적인 구성 요소를 최적화할 수 있습니다. 저희는 시퀀셜 모델과 블록에 초점을 맞춰 아래에서 그 이점을 보여드리겠습니다.
:end_tab:

:begin_tab:`pytorch`
위에서 언급했듯이, PyTorch는 명령형 프로그래밍을 기반으로 하며 동적 계산 그래프를 사용합니다. 심볼릭 프로그래밍의 이식성과 효율성을 활용하기 위해, 개발자들은 두 프로그래밍 패러다임의 이점을 결합할 수 있는지 고려했습니다. 이는 사용자가 순수 명령형 프로그래밍으로 개발하고 디버깅하면서, 제품 수준의 컴퓨팅 성능과 배포가 필요할 때 대부분의 프로그램을 심볼릭 프로그램으로 변환하여 실행할 수 있는 능력을 가진 torchscript로 이어졌습니다.
:end_tab:

:begin_tab:`tensorflow`
명령형 프로그래밍 패러다임은 이제 Tensorflow 2에서 기본이며, 이는 그 언어에 새로 입문한 이들에게 환영할 만한 변화입니다. 그러나 같은 심볼릭 프로그래밍 기법과 이후의 계산 그래프는 여전히 TensorFlow에 존재하며, 사용하기 쉬운 `tf.function` 데코레이터로 접근할 수 있습니다. 이는 명령형 프로그래밍 패러다임을 TensorFlow에 도입했고, 사용자가 더 직관적인 함수를 정의한 다음 이를 감싸고 자동으로 계산 그래프로 컴파일할 수 있게 해주었는데, TensorFlow 팀은 이 기능을 [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph)라고 부릅니다.
:end_tab:

## `Sequential` 클래스 하이브리드화하기

하이브리드화가 어떻게 동작하는지 감을 잡는 가장 쉬운 방법은 여러 층을 가진 심층 네트워크를 고려하는 것입니다. 관습적으로 Python 인터프리터는 모든 층에 대한 코드를 실행하여 그 다음 CPU나 GPU로 전달할 수 있는 명령어를 생성해야 합니다. 단일 (빠른) 컴퓨팅 디바이스의 경우 이는 주요 문제를 일으키지 않습니다. 반면 AWS P3dn.24xlarge 인스턴스 같은 고급 8-GPU 서버를 사용한다면 Python은 모든 GPU를 바쁘게 유지하는 데 어려움을 겪을 것입니다. 단일 스레드 Python 인터프리터가 여기서 병목이 됩니다. `Sequential`을 `HybridSequential`로 대체하여 코드의 상당 부분에 대해 이를 어떻게 해결할 수 있는지 살펴봅시다. 간단한 MLP를 정의하는 것부터 시작합니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
`hybridize` 함수를 호출함으로써, 저희는 MLP에서의 계산을 컴파일하고 최적화할 수 있습니다. 모델의 계산 결과는 변하지 않습니다.
:end_tab:

:begin_tab:`pytorch`
`torch.jit.script` 함수를 사용하여 모델을 변환함으로써, 저희는 MLP에서의 계산을 컴파일하고 최적화할 수 있습니다. 모델의 계산 결과는 변하지 않습니다.
:end_tab:

:begin_tab:`tensorflow`
이전에는 TensorFlow에 구축된 모든 함수가 계산 그래프로 구축되었고, 따라서 기본적으로 JIT 컴파일되었습니다. 그러나 TensorFlow 2.X와 EagerTensor의 출시와 함께 이것은 더 이상 기본 동작이 아닙니다. 
저희는 tf.function으로 이 기능을 다시 활성화할 수 있습니다. tf.function은 함수 데코레이터로 더 일반적으로 사용되지만, 아래에서 보여지는 것처럼 일반 Python 함수처럼 직접 호출할 수도 있습니다. 모델의 계산 결과는 변하지 않습니다.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
이것은 거의 사실이라고 하기에는 너무 좋게 들립니다. 단지 블록을 `HybridSequential`로 지정하고, 이전과 같은 코드를 작성한 후 `hybridize`를 호출하면 됩니다. 일단 이렇게 되면 네트워크는 최적화됩니다(아래에서 성능을 벤치마크할 것입니다). 불행히도 이것은 모든 층에 대해 마법처럼 동작하지는 않습니다. 그렇긴 해도, 어떤 층이 `HybridBlock` 클래스 대신 `Block` 클래스에서 상속받는다면 최적화되지 않을 것입니다.
:end_tab:

:begin_tab:`pytorch`
이것은 거의 사실이라고 하기에는 너무 좋게 들립니다. 이전과 같은 코드를 작성하고 단순히 `torch.jit.script`를 사용해 모델을 변환하기만 하면 됩니다. 일단 이렇게 되면 네트워크는 최적화됩니다(아래에서 성능을 벤치마크할 것입니다).
:end_tab:

:begin_tab:`tensorflow`
이것은 거의 사실이라고 하기에는 너무 좋게 들립니다. 이전과 같은 코드를 작성하고 단순히 `tf.function`을 사용해 모델을 변환하기만 하면 됩니다. 일단 이렇게 되면 네트워크는 TensorFlow의 MLIR 중간 표현에서 계산 그래프로 구축되며 빠른 실행을 위해 컴파일러 수준에서 크게 최적화됩니다(아래에서 성능을 벤치마크할 것입니다).
`tf.function()` 호출에 `jit_compile = True` 플래그를 명시적으로 추가하면 TensorFlow에서 XLA(Accelerated Linear Algebra) 기능이 활성화됩니다. XLA는 특정 경우에서 JIT 컴파일된 코드를 추가로 최적화할 수 있습니다. 이러한 명시적 정의 없이도 그래프 모드 실행은 활성화되지만, XLA는 특정 큰 선형 대수 연산을(딥러닝 응용에서 보는 것과 비슷한 종류) 특히 GPU 환경에서 훨씬 더 빠르게 만들 수 있습니다.
:end_tab:

### 하이브리드화에 의한 가속

컴파일에 의해 얻어지는 성능 향상을 보여주기 위해, 저희는 하이브리드화 전후로 `net(x)`를 평가하는 데 필요한 시간을 비교합니다. 먼저 이 시간을 측정하기 위한 클래스를 정의해 보겠습니다. 이것은 저희가 성능을 측정하고 (개선하기) 시작하면서 챕터 전반에 걸쳐 유용할 것입니다.

```{.python .input}
#@tab all
#@save
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
이제 네트워크를 두 번 호출할 수 있습니다. 한 번은 하이브리드화와 함께, 다른 한 번은 하이브리드화 없이.
:end_tab:

:begin_tab:`pytorch`
이제 네트워크를 두 번 호출할 수 있습니다. 한 번은 torchscript와 함께, 다른 한 번은 torchscript 없이.
:end_tab:

:begin_tab:`tensorflow`
이제 네트워크를 세 번 호출할 수 있습니다. 한 번은 eager로 실행하고, 한 번은 그래프 모드 실행으로, 그리고 다시 JIT 컴파일된 XLA를 사용하여.
:end_tab:

```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
위 결과에서 관찰되듯이, `HybridSequential` 인스턴스가 `hybridize` 함수를 호출한 후에는 심볼릭 프로그래밍의 사용을 통해 컴퓨팅 성능이 향상됩니다.
:end_tab:

:begin_tab:`pytorch`
위 결과에서 관찰되듯이, `nn.Sequential` 인스턴스가 `torch.jit.script` 함수를 사용해 스크립트화된 후에는 심볼릭 프로그래밍의 사용을 통해 컴퓨팅 성능이 향상됩니다.
:end_tab:

:begin_tab:`tensorflow`
위 결과에서 관찰되듯이, `tf.keras.Sequential` 인스턴스가 `tf.function` 함수를 사용해 스크립트화된 후에는 tensorflow에서 그래프 모드 실행을 통한 심볼릭 프로그래밍의 사용을 통해 컴퓨팅 성능이 향상됩니다. 
:end_tab:

### 직렬화

:begin_tab:`mxnet`
모델을 컴파일하는 것의 이점 중 하나는 저희가 모델과 그 매개변수를 디스크로 직렬화(저장)할 수 있다는 것입니다. 이를 통해 선택한 프론트엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다. 이는 학습된 모델을 다른 디바이스에 배포하고 다른 프론트엔드 프로그래밍 언어를 쉽게 사용할 수 있게 해줍니다. 동시에 코드는 종종 명령형 프로그래밍에서 달성할 수 있는 것보다 빠릅니다. `export` 함수를 실제로 살펴봅시다.
:end_tab:

:begin_tab:`pytorch`
모델을 컴파일하는 것의 이점 중 하나는 저희가 모델과 그 매개변수를 디스크로 직렬화(저장)할 수 있다는 것입니다. 이를 통해 선택한 프론트엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다. 이는 학습된 모델을 다른 디바이스에 배포하고 다른 프론트엔드 프로그래밍 언어를 쉽게 사용할 수 있게 해줍니다. 동시에 코드는 종종 명령형 프로그래밍에서 달성할 수 있는 것보다 빠릅니다. `save` 함수를 실제로 살펴봅시다.
:end_tab:

:begin_tab:`tensorflow`
모델을 컴파일하는 것의 이점 중 하나는 저희가 모델과 그 매개변수를 디스크로 직렬화(저장)할 수 있다는 것입니다. 이를 통해 선택한 프론트엔드 언어와 독립적인 방식으로 모델을 저장할 수 있습니다. 이는 학습된 모델을 다른 디바이스에 배포하고 다른 프론트엔드 프로그래밍 언어를 쉽게 사용하거나 서버에서 학습된 모델을 실행할 수 있게 해줍니다. 동시에 코드는 종종 명령형 프로그래밍에서 달성할 수 있는 것보다 빠릅니다. 
tensorflow에서 저장을 가능하게 하는 저수준 API는 `tf.saved_model`입니다. 
`saved_model` 인스턴스를 실제로 살펴봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
모델은 (큰 바이너리) 파라미터 파일과 모델 계산을 실행하는 데 필요한 프로그램의 JSON 설명으로 분해됩니다. 이 파일들은 C++, R, Scala, Perl 같이 Python이나 MXNet에서 지원되는 다른 프론트엔드 언어에서 읽을 수 있습니다. 모델 설명의 처음 몇 줄을 살펴봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
앞서 저희는 `hybridize` 함수를 호출한 후, 모델이 우수한 컴퓨팅 성능과 이식성을 달성할 수 있음을 보여드렸습니다. 그러나 하이브리드화는 특히 제어 흐름 측면에서 모델 유연성에 영향을 줄 수 있다는 점에 유의하십시오. 

또한, `forward` 함수를 사용해야 하는 `Block` 인스턴스와 달리, `HybridBlock` 인스턴스의 경우 `hybrid_forward` 함수를 사용해야 합니다.
:end_tab:

```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
위 코드는 4개의 은닉 유닛과 2개의 출력을 가진 간단한 네트워크를 구현합니다. `hybrid_forward` 함수는 추가 인자 `F`를 받습니다. 이는 코드가 하이브리드화되었는지 여부에 따라 약간 다른 라이브러리(`ndarray` 또는 `symbol`)를 처리에 사용하기 때문에 필요합니다. 두 클래스 모두 매우 비슷한 기능을 수행하며 MXNet은 자동으로 인자를 결정합니다. 무슨 일이 일어나는지 이해하기 위해 함수 호출의 일부로 인자를 출력합니다.
:end_tab:

```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
순방향 계산을 반복해도 같은 출력이 나옵니다(세부 사항은 생략합니다). 이제 `hybridize` 함수를 호출하면 무슨 일이 일어나는지 봅시다.
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
`ndarray`를 사용하는 대신 저희는 이제 `F`에 대해 `symbol` 모듈을 사용합니다. 더욱이, 입력은 `ndarray` 타입이지만, 네트워크를 통해 흐르는 데이터는 컴파일 과정의 일부로 이제 `symbol` 타입으로 변환됩니다. 함수 호출을 반복하면 놀라운 결과가 나옵니다.
:end_tab:

```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
이는 이전에 본 것과는 상당히 다릅니다. `hybrid_forward`에 정의된 모든 print 문이 생략됩니다. 실제로 하이브리드화 후 `net(x)`의 실행은 더 이상 Python 인터프리터를 포함하지 않습니다. 이는 (print 문 같은) 어떤 불필요한 Python 코드도 훨씬 더 효율적인 실행과 더 나은 성능을 위해 생략된다는 의미입니다. 대신 MXNet은 C++ 백엔드를 직접 호출합니다. 또한 일부 함수는 `symbol` 모듈에서 지원되지 않으며(예: `asnumpy`), `a += b`와 `a[:] = a + b` 같은 인플레이스 연산은 `a = a + b`로 재작성되어야 한다는 점에 유의하십시오. 그럼에도 불구하고, 속도가 중요할 때마다 모델의 컴파일은 노력할 가치가 있습니다. 그 이점은 모델의 복잡성, CPU의 속도, GPU의 속도와 개수에 따라 작은 비율부터 두 배 이상의 속도까지 다양할 수 있습니다.
:end_tab:

## 요약


* 명령형 프로그래밍은 제어 흐름이 있는 코드를 작성하고 Python 소프트웨어 생태계의 많은 부분을 사용할 수 있기 때문에 새로운 모델을 설계하기 쉽게 만듭니다.
* 심볼릭 프로그래밍은 저희가 프로그램을 명시하고 실행하기 전에 컴파일할 것을 요구합니다. 이점은 향상된 성능입니다.

:begin_tab:`mxnet` 
* MXNet은 필요에 따라 두 접근법의 장점을 결합할 수 있습니다.
* `HybridSequential`과 `HybridBlock` 클래스로 구성된 모델은 `hybridize` 함수를 호출하여 명령형 프로그램을 심볼릭 프로그램으로 변환할 수 있습니다.
:end_tab:


## 연습문제


:begin_tab:`mxnet` 
1. 이 절의 `HybridNet` 클래스의 `hybrid_forward` 함수 첫 줄에 `x.asnumpy()`를 추가하십시오. 코드를 실행하고 발생하는 오류를 관찰하십시오. 왜 발생합니까?
1. `hybrid_forward` 함수에 제어 흐름, 즉 Python 문장인 `if`와 `for`를 추가하면 어떻게 됩니까?
1. 이전 챕터들에서 관심을 끄는 모델들을 검토하십시오. 재구현하여 그들의 계산 성능을 향상시킬 수 있습니까?
:end_tab:

:begin_tab:`pytorch,tensorflow` 
1. 이전 챕터들에서 관심을 끄는 모델들을 검토하십시오. 재구현하여 그들의 계산 성능을 향상시킬 수 있습니까?
:end_tab:




:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/2492)
:end_tab:
