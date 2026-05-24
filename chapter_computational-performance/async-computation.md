# 비동기 계산
:label:`sec_async`

오늘날의 컴퓨터는 고도로 병렬화된 시스템으로, 여러 개의 CPU 코어(보통 코어당 여러 스레드), GPU당 여러 처리 요소, 그리고 종종 디바이스당 여러 개의 GPU로 구성되어 있습니다. 요컨대, 저희는 많은 서로 다른 작업을 동시에, 종종 서로 다른 디바이스에서 처리할 수 있습니다. 불행히도 Python은 적어도 어느 정도의 추가적인 도움 없이는 병렬 및 비동기 코드를 작성하기에 좋은 방법이 아닙니다. 결국 Python은 단일 스레드이며, 이는 앞으로도 바뀔 가능성이 낮습니다. MXNet과 TensorFlow 같은 딥러닝 프레임워크는 성능을 향상시키기 위해 *비동기 프로그래밍* 모델을 채택하는 반면,
PyTorch는 Python 자체의 스케줄러를 사용하여 다른 성능 트레이드오프를 가집니다.
PyTorch에서는 기본적으로 GPU 연산이 비동기로 동작합니다. GPU를 사용하는 함수를 호출하면, 연산은 특정 디바이스의 큐에 들어가지만 반드시 즉시 실행되는 것은 아닙니다. 이를 통해 저희는 CPU나 다른 GPU에서의 연산을 포함하여 더 많은 계산을 병렬로 실행할 수 있습니다.

따라서, 비동기 프로그래밍이 어떻게 동작하는지 이해하면 계산 요구 사항과 상호 의존성을 사전에 줄임으로써 더 효율적인 프로그램을 개발하는 데 도움이 됩니다. 이를 통해 메모리 오버헤드를 줄이고 프로세서 활용도를 높일 수 있습니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## 백엔드를 통한 비동기성

:begin_tab:`mxnet`
워밍업으로 다음의 간단한 예제를 살펴보겠습니다. 저희는 랜덤 행렬을 생성한 뒤 이를 곱하고자 합니다. 차이를 보기 위해 NumPy와 `mxnet.np`로 각각 수행해 보겠습니다.
:end_tab:

:begin_tab:`pytorch`
워밍업으로 다음의 간단한 예제를 살펴보겠습니다. 저희는 랜덤 행렬을 생성한 뒤 이를 곱하고자 합니다. 차이를 보기 위해 NumPy와 PyTorch tensor로 각각 수행해 보겠습니다.
PyTorch `tensor`는 GPU상에 정의되어 있다는 점에 유의하십시오.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
MXNet을 통한 벤치마크 출력은 수십 배 이상 더 빠릅니다. 둘 다 같은 프로세서에서 실행되므로 무언가 다른 일이 벌어지고 있는 것이 분명합니다.
MXNet이 반환되기 전에 백엔드의 모든 계산을 강제로 끝내도록 하면 앞서 일어난 일이 드러납니다. 계산은 백엔드에서 수행되는 동안 프론트엔드는 Python에 제어권을 반환합니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch를 통한 벤치마크 출력은 수십 배 이상 더 빠릅니다.
NumPy의 내적은 CPU 프로세서에서 실행되는 반면
PyTorch의 행렬 곱셈은 GPU에서 실행되므로 후자가
훨씬 빠를 것으로 예상됩니다. 그러나 시간 차이가 워낙 크다는 점은 무언가
다른 일이 벌어지고 있음을 시사합니다.
PyTorch에서는 기본적으로 GPU 연산이 비동기로 동작합니다.
PyTorch가 반환되기 전에 모든 계산을 강제로 끝내도록 하면
앞서 일어난 일이 드러납니다. 계산은 백엔드에서 수행되는 동안
프론트엔드는 Python에 제어권을 반환합니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
대체로 MXNet은 예를 들어 Python을 통한 사용자와의 직접적인 상호작용을 위한 프론트엔드와, 시스템이 계산을 수행하기 위해 사용하는 백엔드를 가지고 있습니다. 
:numref:`fig_frontends`에 나타난 것처럼, 사용자는 Python, R, Scala, C++ 같은 다양한 프론트엔드 언어로 MXNet 프로그램을 작성할 수 있습니다. 사용된 프론트엔드 프로그래밍 언어가 무엇이든, MXNet 프로그램의 실행은 주로 C++ 구현의 백엔드에서 일어납니다. 프론트엔드 언어에서 발행된 연산은 실행을 위해 백엔드로 전달됩니다. 
백엔드는 큐에 쌓인 작업을 지속적으로 수집하고 실행하는 자체 스레드를 관리합니다. 이것이 동작하려면 백엔드가 계산 그래프 내 다양한 단계 사이의 의존성을 추적할 수 있어야 한다는 점에 유의하십시오. 따라서 서로 의존하는 연산은 병렬화할 수 없습니다.
:end_tab:

:begin_tab:`pytorch`
대체로 PyTorch는 예를 들어 Python을 통한 사용자와의 직접적인 상호작용을 위한 프론트엔드와, 시스템이 계산을 수행하기 위해 사용하는 백엔드를 가지고 있습니다. 
:numref:`fig_frontends`에 나타난 것처럼, 사용자는 Python과 C++ 같은 다양한 프론트엔드 언어로 PyTorch 프로그램을 작성할 수 있습니다. 사용된 프론트엔드 프로그래밍 언어가 무엇이든, PyTorch 프로그램의 실행은 주로 C++ 구현의 백엔드에서 일어납니다. 프론트엔드 언어에서 발행된 연산은 실행을 위해 백엔드로 전달됩니다.
백엔드는 큐에 쌓인 작업을 지속적으로 수집하고 실행하는 자체 스레드를 관리합니다.
이것이 동작하려면 백엔드가 계산 그래프 내
다양한 단계 사이의 의존성을 추적할 수 있어야 한다는 점에 유의하십시오.
따라서 서로 의존하는 연산은 병렬화할 수 없습니다.
:end_tab:

![프로그래밍 언어 프론트엔드와 딥러닝 프레임워크 백엔드.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

의존성 그래프를 좀 더 잘 이해하기 위해 또 다른 간단한 예제를 살펴보겠습니다.

```{.python .input}
#@tab mxnet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![백엔드는 계산 그래프 내 다양한 단계 사이의 의존성을 추적합니다.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`



위 코드 스니펫은 :numref:`fig_asyncgraph`에도 나타나 있습니다.
Python 프론트엔드 스레드가 처음 세 문장 중 하나를 실행할 때마다, 단지 작업을 백엔드 큐로 반환할 뿐입니다. 마지막 문장의 결과를 *출력*해야 할 때, Python 프론트엔드 스레드는 C++ 백엔드 스레드가 변수 `z`의 결과를 계산하는 것을 완료할 때까지 기다립니다. 이 설계의 한 가지 장점은 Python 프론트엔드 스레드가 실제 계산을 수행할 필요가 없다는 점입니다. 따라서 Python의 성능과 무관하게 프로그램 전체 성능에는 거의 영향이 없습니다. :numref:`fig_threading`은 프론트엔드와 백엔드가 어떻게 상호작용하는지를 보여줍니다.

![프론트엔드와 백엔드의 상호작용.](../img/threading.svg)
:label:`fig_threading`




## 배리어와 블로커

:begin_tab:`mxnet`
Python을 강제로 완료될 때까지 대기시키는 여러 연산이 있습니다.

* 가장 명백하게 `npx.waitall()`은 계산 명령이 언제 발행되었는지에 상관없이 모든 계산이 완료될 때까지 기다립니다. 실제로는 이 연산자는 성능 저하로 이어질 수 있으므로 반드시 필요한 경우가 아니면 사용하지 않는 것이 좋습니다.
* 특정 변수가 사용 가능해질 때까지만 기다리고 싶다면 `z.wait_to_read()`를 호출할 수 있습니다. 이 경우 MXNet은 변수 `z`가 계산될 때까지 Python으로의 반환을 차단합니다. 그 외의 계산은 그 후에도 계속될 수 있습니다.

실제로 이것이 어떻게 동작하는지 살펴보겠습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
두 연산 모두 완료하는 데 거의 같은 시간이 걸립니다. 명백한 차단 연산 외에도 *암묵적* 블로커를 인지하는 것을 권장합니다. 변수를 출력하려면 변수가 사용 가능해야 함이 분명하므로 이는 블로커입니다. 마지막으로, `z.asnumpy()`를 통한 NumPy로의 변환과 `z.item()`을 통한 스칼라로의 변환은 NumPy에 비동기성의 개념이 없기 때문에 차단됩니다. 이는 `print` 함수와 마찬가지로 값에 접근해야 합니다. 

MXNet의 스코프에서 NumPy로 그리고 그 반대로 적은 양의 데이터를 자주 복사하면, 이러한 각 연산은 다른 어떤 작업도 수행되기 *전에* 관련 항을 얻기 위해 필요한 모든 중간 결과를 계산 그래프가 평가하도록 요구하기 때문에, 그렇지 않으면 효율적인 코드의 성능을 망칠 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 계산 개선하기

:begin_tab:`mxnet`
멀티 스레드 부하가 큰 시스템에서는(일반적인 노트북도 4개 이상의 스레드를 가지며 멀티 소켓 서버에서는 이 수가 256개를 넘을 수 있습니다) 연산 스케줄링의 오버헤드가 상당히 커질 수 있습니다. 그래서 계산과 스케줄링이 비동기적이고 병렬로 일어나는 것이 매우 바람직합니다. 그렇게 하는 것의 이점을 보여주기 위해, 변수를 1만큼 여러 번 증가시킬 때 순차적으로 혹은 비동기적으로 수행하면 무슨 일이 일어나는지 살펴보겠습니다. 저희는 각 덧셈 사이에 `wait_to_read` 배리어를 삽입함으로써 동기 실행을 시뮬레이션합니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Python 프론트엔드 스레드와 C++ 백엔드 스레드 사이의 다소 단순화된 상호작용은 다음과 같이 요약할 수 있습니다.
1. 프론트엔드는 백엔드에게 계산 작업 `y = x + 1`을 큐에 넣도록 명령합니다.
1. 그 다음 백엔드는 큐에서 계산 작업을 받아 실제 계산을 수행합니다.
1. 그런 다음 백엔드는 계산 결과를 프론트엔드로 반환합니다.
이 세 단계의 소요 시간을 각각 $t_1, t_2$, $t_3$이라고 가정해 봅시다. 비동기 프로그래밍을 사용하지 않는다면, 10000번의 계산을 수행하는 데 걸리는 총 시간은 대략 $10000 (t_1+ t_2 + t_3)$입니다. 비동기 프로그래밍을 사용하면, 프론트엔드가 각 루프마다 백엔드가 계산 결과를 반환할 때까지 기다릴 필요가 없기 때문에, 10000번의 계산을 수행하는 데 걸리는 총 시간을 $t_1 + 10000 t_2 + t_3$($10000 t_2 > 9999t_1$이라고 가정)로 줄일 수 있습니다.
:end_tab:


## 요약


* 딥러닝 프레임워크는 Python 프론트엔드를 실행 백엔드로부터 분리할 수 있습니다. 이를 통해 백엔드로 명령을 빠르게 비동기로 삽입할 수 있고 그에 따른 병렬성을 얻을 수 있습니다.
* 비동기성은 상당히 반응성이 좋은 프론트엔드로 이어집니다. 그러나 작업 큐가 과도하게 채워지면 메모리 소비가 과도해질 수 있으므로 주의하십시오. 프론트엔드와 백엔드를 대략 동기화된 상태로 유지하기 위해 각 미니배치마다 동기화하는 것이 권장됩니다.
* 칩 벤더는 딥러닝의 효율성에 대한 훨씬 더 세분화된 통찰을 얻기 위한 정교한 성능 분석 도구를 제공합니다.

:begin_tab:`mxnet`
* MXNet의 메모리 관리에서 Python으로의 변환은 특정 변수가 준비될 때까지 백엔드를 대기시킨다는 사실에 유의하십시오. `print`, `asnumpy`, `item` 같은 함수들은 모두 이런 효과를 가집니다. 이는 바람직할 수도 있지만 동기화의 부주의한 사용은 성능을 망칠 수 있습니다.
:end_tab:


## 연습문제

:begin_tab:`mxnet`
1. 저희는 위에서 비동기 계산을 사용하면 10000번의 계산을 수행하는 데 필요한 총 시간을 $t_1 + 10000 t_2 + t_3$으로 줄일 수 있다고 언급했습니다. 여기서 왜 $10000 t_2 > 9999 t_1$을 가정해야 합니까?
1. `waitall`과 `wait_to_read`의 차이를 측정해 보십시오. 힌트: 여러 명령을 수행하고 중간 결과에 대해 동기화하십시오.
:end_tab:

:begin_tab:`pytorch`
1. CPU에서 이 절의 동일한 행렬 곱셈 연산을 벤치마크해 보십시오. 백엔드를 통한 비동기성을 여전히 관찰할 수 있습니까?
:end_tab:

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/2564)
:end_tab:
