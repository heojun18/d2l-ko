# 미니배치 확률적 경사 하강법(Minibatch Stochastic Gradient Descent)
:label:`sec_minibatch_sgd`

지금까지 저희는 경사 기반 학습 접근법에서 두 가지 극단을 만났습니다. :numref:`sec_gd`는 한 번에 한 번 패스로 경사도를 계산하고 파라미터를 업데이트하기 위해 전체 데이터셋을 사용합니다. 반대로 :numref:`sec_sgd`는 진전을 이루기 위해 한 번에 하나의 학습 예제를 처리합니다.
각각은 그들 자신의 단점이 있습니다.
경사 하강법은 데이터가 매우 유사할 때마다 특별히 *데이터 효율적*이지 않습니다.
확률적 경사 하강법은 CPU와 GPU가 벡터화의 완전한 힘을 활용할 수 없기 때문에 특별히 *계산 효율적*이지 않습니다.
이는 그 사이에 어떤 것이 있을 수 있음을 시사하고,
실제로 그것이 지금까지 저희가 논의한 예제에서 사용해 온 것입니다.

## 벡터화와 캐시

미니배치를 사용하기로 한 결정의 핵심은 계산 효율성입니다. 이는 여러 GPU와 여러 서버로의 병렬화를 고려할 때 가장 쉽게 이해됩니다. 이 경우 저희는 적어도 한 이미지를 각 GPU에 보내야 합니다. 서버당 8개의 GPU와 16개의 서버가 있으면 저희는 이미 128보다 작지 않은 미니배치 크기에 도달합니다.

단일 GPU나 심지어 CPU에 대해서는 사정이 좀 더 미묘합니다. 이러한 장치들은 여러 유형의 메모리, 종종 여러 유형의 계산 단위, 그리고 그것들 사이의 서로 다른 대역폭 제약 조건을 가집니다.
예를 들어, CPU는 적은 수의 레지스터와 그 다음 L1, L2, 그리고 어떤 경우에는 L3 캐시(서로 다른 프로세서 코어 간에 공유됨)를 가집니다.
이러한 캐시들은 크기와 지연 시간이 증가하는(동시에 대역폭은 감소하는) 형태입니다.
프로세서가 메인 메모리 인터페이스가 제공할 수 있는 것보다 훨씬 더 많은 작업을 수행할 수 있다는 점은 충분히 말할 수 있습니다.

첫째, AVX-512 벡터화를 가진 16코어의 2GHz CPU는 초당 최대 $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ 바이트를 처리할 수 있습니다. GPU의 능력은 이 숫자를 100배 정도 쉽게 초과합니다. 반면, 중급 서버 프로세서는 100 GB/s를 훨씬 넘지 않는 대역폭을 가질 수 있습니다. 즉, 프로세서를 공급하는 데 필요한 양의 10분의 1도 안 됩니다. 더 안 좋게도, 모든 메모리 접근이 동등하게 만들어진 것은 아닙니다. 메모리 인터페이스는 일반적으로 64비트 폭이나 그 이상(예: GPU에서는 최대 384비트)이므로, 단일 바이트를 읽는 것은 훨씬 더 넓은 접근의 비용을 발생시킵니다.

둘째, 첫 번째 접근에 대해 상당한 오버헤드가 있는 반면 순차적 접근은 비교적 저렴합니다(이는 종종 버스트 읽기라고 불립니다). 여러 소켓, 칩렛, 다른 구조가 있을 때의 캐싱 등 염두에 두어야 할 사항이 훨씬 더 많습니다.
더 깊이 있는 논의는 이 [위키피디아 글](https://en.wikipedia.org/wiki/Cache_hierarchy)을
참조하세요.

이러한 제약을 완화하는 방법은 실제로 프로세서에 데이터를 공급할 만큼 충분히 빠른 CPU 캐시의 계층 구조를 사용하는 것입니다. 이것이 딥러닝의 배칭 뒤에 *바로* 그 추진력입니다. 사정을 단순하게 유지하기 위해, 행렬-행렬 곱셈, 예를 들어 $\mathbf{A} = \mathbf{B}\mathbf{C}$를 생각해 봅시다. 저희는 $\mathbf{A}$를 계산하기 위해 여러 가지 선택지를 가집니다. 예를 들어, 저희는 다음을 시도해 볼 수 있습니다.

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$를 계산할 수 있습니다. 즉, 내적을 통해 요소별로 계산할 수 있습니다.
1. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$를 계산할 수 있습니다. 즉, 한 번에 한 열씩 계산할 수 있습니다. 마찬가지로 $\mathbf{A}$를 한 번에 한 행 $\mathbf{A}_{i,:}$씩 계산할 수 있습니다.
1. 단순히 $\mathbf{A} = \mathbf{B} \mathbf{C}$를 계산할 수 있습니다.
1. $\mathbf{B}$와 $\mathbf{C}$를 더 작은 블록 행렬로 나누고 $\mathbf{A}$를 한 번에 한 블록씩 계산할 수 있습니다.

첫 번째 옵션을 따른다면, 요소 $\mathbf{A}_{ij}$를 계산하고자 할 때마다 한 행 벡터와 한 열 벡터를 CPU에 복사해야 합니다. 더 안 좋은 것은, 행렬 요소가 순차적으로 정렬되어 있다는 사실 때문에 저희가 메모리에서 두 벡터 중 하나를 읽을 때 많은 분리된 위치에 접근해야 한다는 점입니다. 두 번째 옵션이 훨씬 더 유리합니다. 그 안에서, 저희는 $\mathbf{B}$를 계속 순회하는 동안 열 벡터 $\mathbf{C}_{:,j}$를 CPU 캐시에 유지할 수 있습니다. 이는 그에 상응하여 더 빠른 접근으로 메모리 대역폭 요구 사항을 절반으로 줄입니다. 물론, 옵션 3이 가장 바람직합니다. 안타깝게도, 대부분의 행렬은 캐시에 완전히 들어가지 않을 수 있습니다(결국 이것이 저희가 논의하고 있는 것입니다). 그러나 옵션 4는 실용적으로 유용한 대안을 제공합니다. 저희는 행렬의 블록을 캐시로 이동시키고 그것들을 지역적으로 곱할 수 있습니다. 최적화된 라이브러리가 이를 처리해 줍니다. 이러한 작업이 실제로 얼마나 효율적인지 살펴봅시다.

계산 효율성 너머로, Python과 딥러닝 프레임워크 자체에 의해 도입된 오버헤드는 상당합니다. 저희가 명령을 실행할 때마다 Python 인터프리터는 MXNet 엔진에 명령을 보내며, 이 엔진은 그것을 계산 그래프에 삽입하고 스케줄링 중에 처리해야 한다는 점을 기억하세요. 그러한 오버헤드는 꽤 해로울 수 있습니다. 요컨대, 가능할 때마다 벡터화(그리고 행렬)를 사용하는 것이 매우 권장됩니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

이 책의 나머지에서 저희가 자주 실행 시간을 벤치마크할 것이므로, 타이머를 정의해 봅시다.

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

요소별 할당은 단순히 $\mathbf{B}$와 $\mathbf{C}$의 모든 행과 열을 각각 반복하여 $\mathbf{A}$에 값을 할당합니다.

```{.python .input}
#@tab mxnet
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

더 빠른 전략은 열별 할당을 수행하는 것입니다.

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

마지막으로, 가장 효과적인 방식은 전체 작업을 하나의 블록으로 수행하는 것입니다. 
임의의 두 행렬 $\mathbf{B} \in \mathbb{R}^{m \times n}$와 $\mathbf{C} \in \mathbb{R}^{n \times p}$를 곱하는 것은 스칼라 곱셈과 덧셈을 별개의 작업으로 셀 때(실제로는 융합됨) 대략 $2mnp$ 부동소수점 연산이 걸린다는 점에 유의하세요.
따라서, 두 $256 \times 256$ 행렬을 곱하면
$0.03$ 십억 부동소수점 연산이 걸립니다.
각 작업의 속도가 어떻게 되는지 봅시다.

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## 미니배치

:label:`sec_minibatches`

과거 저희는 파라미터를 업데이트하기 위해 단일 관측값보다는 데이터의 *미니배치*를 읽는다는 것을 당연하게 여겼습니다. 저희는 이제 그것에 대한 간단한 정당화를 제공합니다. 단일 관측값을 처리하려면 저희는 많은 단일 행렬-벡터(또는 심지어 벡터-벡터) 곱셈을 수행해야 하며, 이는 꽤 비싸고 기본 딥러닝 프레임워크에 의해 상당한 오버헤드를 발생시킵니다. 이는 데이터에 적용될 때(종종 추론이라고 불립니다) 네트워크를 평가하는 것과 파라미터를 업데이트하기 위해 경사도를 계산할 때 모두에 적용됩니다. 즉, 이는 저희가 $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$를 수행할 때마다 적용됩니다. 여기서

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

입니다. 저희는 이 작업의 *계산* 효율성을 한 번에 관측값의 미니배치에 적용함으로써 증가시킬 수 있습니다. 즉, 단일 관측값에 대한 경사도 $\mathbf{g}_t$를 작은 배치에 대한 것으로 대체합니다.

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

이것이 $\mathbf{g}_t$의 통계적 특성에 어떤 영향을 미치는지 봅시다. $\mathbf{x}_t$와 미니배치 $\mathcal{B}_t$의 모든 요소가 학습 세트에서 균등하게 무작위로 추출되므로, 경사도의 기댓값은 변하지 않습니다. 반면, 분산은 크게 감소합니다. 미니배치 경사도는 평균화되고 있는 $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$개의 독립적 경사도로 구성되므로, 그것의 표준 편차는 $b^{-\frac{1}{2}}$의 인수만큼 감소합니다. 이는 그 자체로 좋은 일인데, 업데이트가 전체 경사도와 더 안정적으로 정렬됨을 의미하기 때문입니다.

순진하게 이는 큰 미니배치 $\mathcal{B}_t$를 선택하는 것이 보편적으로 바람직함을 나타내는 것 같습니다. 안타깝게도, 어느 시점 이후에는 계산 비용의 선형 증가에 비해 표준 편차의 추가 감소가 미미합니다. 실제로 저희는 GPU의 메모리에 여전히 들어가면서도 좋은 계산 효율성을 제공할 만큼 충분히 큰 미니배치를 선택합니다. 절약을 설명하기 위해 일부 코드를 살펴봅시다. 그 안에서 저희는 동일한 행렬-행렬 곱셈을 수행하지만, 이번에는 한 번에 64열의 "미니배치"로 나누어 수행합니다.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

보시다시피, 미니배치에 대한 계산은 본질적으로 전체 행렬에 대한 것만큼 효율적입니다. 주의의 한마디가 필요합니다. :numref:`sec_batch_norm`에서 저희는 미니배치의 분산 양에 크게 의존하는 일종의 정규화를 사용했습니다. 후자를 증가시킴에 따라, 분산은 감소하고 그와 함께 배치 정규화로 인한 노이즈 주입의 이점도 감소합니다. 적절한 항을 재조정하고 계산하는 방법에 대한 자세한 내용은 예를 들어 :citet:`Ioffe.2017`를 참조하세요.

## 데이터셋 읽기

미니배치가 데이터에서 어떻게 효율적으로 생성되는지 살펴봅시다. 다음에서 저희는 이러한 최적화 알고리즘을 비교하기 위해 NASA가 다양한 [항공기 날개 소음](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise)을 테스트하기 위해 개발한 데이터셋을 사용합니다. 편의를 위해 저희는 처음 $1,500$개의 예제만 사용합니다. 데이터는 전처리를 위해 백색화됩니다. 즉, 저희는 평균을 제거하고 좌표당 분산을 $1$로 재조정합니다.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## 처음부터 구현하기

:numref:`sec_linear_scratch`의 미니배치 확률적 경사 하강법 구현을 떠올려 봅시다. 다음에서 저희는 약간 더 일반적인 구현을 제공합니다. 편의를 위해 그것은 이 장 후반부에 소개될 다른 최적화 알고리즘과 동일한 호출 시그니처를 가집니다. 구체적으로, 저희는 상태 입력 `states`를 추가하고 하이퍼파라미터를 사전 `hyperparams`에 둡니다. 추가로,
저희는 학습 함수에서 각 미니배치 예제의 손실을 평균낼 것이므로, 최적화 알고리즘의 경사도는 배치 크기로
나눌 필요가 없습니다.

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

다음으로, 이 장 후반부에 소개될 다른 최적화 알고리즘의 사용을 용이하게 하기 위해 일반적인 학습 함수를 구현합니다. 이는 선형 회귀 모델을 초기화하고 미니배치 확률적 경사 하강법과 후속적으로 소개되는 다른 알고리즘으로 모델을 학습시키는 데 사용할 수 있습니다.

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

배치 경사 하강법에 대해 최적화가 어떻게 진행되는지 봅시다. 이는 미니배치 크기를 1500(즉, 예제의 총 개수)으로 설정함으로써 달성할 수 있습니다. 그 결과 모델 파라미터는 에포크당 한 번만 업데이트됩니다. 진전이 거의 없습니다. 실제로, 6단계 이후 진전이 멈춥니다.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

배치 크기가 1과 같을 때, 저희는 최적화를 위해 확률적 경사 하강법을 사용합니다. 구현의 단순성을 위해 저희는 상수(작지만) 학습률을 선택했습니다. 확률적 경사 하강법에서, 모델 파라미터는 예제가 처리될 때마다 업데이트됩니다. 저희의 경우 이는 에포크당 1500번의 업데이트에 해당합니다. 보시다시피, 한 에포크 후 목적 함수의 값의 감소가 느려집니다. 두 절차 모두 한 에포크 안에 1500개의 예제를 처리했지만, 확률적 경사 하강법은 저희의 실험에서 경사 하강법보다 더 많은 시간을 소비합니다. 이는 확률적 경사 하강법이 파라미터를 더 자주 업데이트했고 단일 관측값을 한 번에 하나씩 처리하는 것이 덜 효율적이기 때문입니다.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

마지막으로, 배치 크기가 100과 같을 때, 저희는 최적화를 위해 미니배치 확률적 경사 하강법을 사용합니다. 에포크당 필요한 시간은 확률적 경사 하강법에 필요한 시간과 배치 경사 하강법에 대한 시간보다 짧습니다.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

배치 크기를 10으로 줄이면, 각 배치의 작업 부하가 실행하기 덜 효율적이기 때문에 각 에포크에 대한 시간이 증가합니다.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

이제 저희는 이전 네 가지 실험에 대한 시간 대 손실을 비교할 수 있습니다. 보시다시피, 확률적 경사 하강법은 처리된 예제 수 측면에서는 GD보다 더 빠르게 수렴하지만, 예제별로 경사도를 계산하는 것이 효율적이지 않기 때문에 동일한 손실에 도달하는 데 GD보다 더 많은 시간을 사용합니다. 미니배치 확률적 경사 하강법은 수렴 속도와 계산 효율성을 절충할 수 있습니다. 10의 미니배치 크기는 확률적 경사 하강법보다 더 효율적입니다. 100의 미니배치 크기는 실행 시간 측면에서 GD보다도 우수합니다.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 간결한 구현

Gluon에서, 저희는 최적화 알고리즘을 호출하기 위해 `Trainer` 클래스를 사용할 수 있습니다. 이는 일반적인 학습 함수를 구현하는 데 사용됩니다. 저희는 현재 장 전체에서 이를 사용할 것입니다.

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Gluon을 사용해 마지막 실험을 반복하면 동일한 동작을 보입니다.

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## 요약

* 벡터화는 딥러닝 프레임워크로부터 발생하는 오버헤드 감소와 CPU와 GPU의 더 나은 메모리 지역성과 캐싱으로 인해 코드를 더 효율적으로 만듭니다.
* 확률적 경사 하강법에서 발생하는 통계적 효율성과 한 번에 큰 데이터 배치를 처리하는 데서 발생하는 계산 효율성 사이에는 절충이 있습니다.
* 미니배치 확률적 경사 하강법은 양쪽 모두의 장점을 제공합니다. 계산적 및 통계적 효율성입니다.
* 미니배치 확률적 경사 하강법에서 저희는 학습 데이터의 무작위 순열로 얻은 데이터 배치를 처리합니다(즉, 각 관측값은 무작위 순서이긴 하지만 에포크당 한 번만 처리됩니다).
* 학습 중에 학습률을 감쇠시키는 것이 권장됩니다.
* 일반적으로, 시계 시간으로 측정할 때 더 작은 위험으로의 수렴을 위해 미니배치 확률적 경사 하강법이 확률적 경사 하강법과 경사 하강법보다 더 빠릅니다.

## 연습문제

1. 배치 크기와 학습률을 수정하고 각 에포크에서 목적 함수의 값의 감소율과 소비되는 시간을 관찰해 보세요.
1. MXNet 문서를 읽고 `Trainer` 클래스 `set_learning_rate` 함수를 사용해 각 에포크 후 미니배치 확률적 경사 하강법의 학습률을 이전 값의 1/10으로 줄여 보세요.
1. 미니배치 확률적 경사 하강법을 학습 세트에서 실제로 *복원 추출*하는 변종과 비교해 보세요. 어떤 일이 일어나나요?
1. 사악한 지니가 모르게 데이터셋을 복제합니다(즉, 각 관측값이 두 번 발생하고 데이터셋이 원래 크기의 두 배로 늘어나지만 아무도 말해주지 않았습니다). 확률적 경사 하강법, 미니배치 확률적 경사 하강법, 그리고 경사 하강법의 동작은 어떻게 변하나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
