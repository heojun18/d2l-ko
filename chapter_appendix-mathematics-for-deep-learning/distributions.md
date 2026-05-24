# 분포
:label:`sec_distributions`

이제 이산과 연속 환경 모두에서 확률로 작업하는 방법을 배웠으므로, 마주치는 몇 가지 일반적인 분포에 대해 알아보겠습니다. 머신러닝의 영역에 따라, 저희는 이것들 중 훨씬 더 많은 것에 친숙해져야 할 수도 있고, 딥러닝의 일부 영역에 대해서는 잠재적으로 전혀 친숙해질 필요가 없을 수도 있습니다. 그러나, 이것은 친숙해질 좋은 기본 리스트입니다. 먼저 몇 가지 일반적인 라이브러리를 임포트해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## 베르누이

이는 보통 마주치는 가장 단순한 확률 변수입니다. 이 확률 변수는 확률 $p$로 $1$이 나오고 확률 $1-p$로 $0$이 나오는 동전 던지기를 인코딩합니다. 만약 이 분포를 가진 확률 변수 $X$가 있다면, 저희는 다음과 같이 쓸 것입니다.

$$
X \sim \textrm{Bernoulli}(p).
$$

누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

확률 질량 함수는 아래에 플롯되어 있습니다.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제, 누적 분포 함수 :eqref:`eq_bernoulli_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

만약 $X \sim \textrm{Bernoulli}(p)$이면:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

저희는 다음과 같이 베르누이 확률 변수에서 임의 모양의 배열을 샘플링할 수 있습니다.

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 이산 균등

다음으로 일반적으로 마주치는 확률 변수는 이산 균등입니다. 여기서의 저희의 논의를 위해, 정수 $\{1, 2, \ldots, n\}$에서 지원된다고 가정할 것이지만, 다른 어떤 값 집합도 자유롭게 선택될 수 있습니다. 이 맥락에서 *균등*이라는 단어의 의미는 모든 가능한 값이 동등하게 가능성이 있다는 것입니다. 각 값 $i \in \{1, 2, 3, \ldots, n\}$에 대한 확률은 $p_i = \frac{1}{n}$입니다. 저희는 이 분포를 가진 확률 변수 $X$를 다음과 같이 표기할 것입니다.

$$
X \sim U(n).
$$

누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

먼저 확률 질량 함수를 플롯해 봅시다.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제, 누적 분포 함수 :eqref:`eq_discrete_uniform_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

만약 $X \sim U(n)$이면:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

저희는 다음과 같이 이산 균등 확률 변수에서 임의 모양의 배열을 샘플링할 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 연속 균등

다음으로, 연속 균등 분포를 논의해 봅시다. 이 확률 변수 뒤에 있는 아이디어는 이산 균등 분포에서 $n$을 증가시키고 구간 $[a, b]$ 내에 맞도록 스케일링하면, $[a, b]$의 임의의 값을 모두 동등한 확률로 그냥 선택하는 연속 확률 변수에 접근할 것이라는 것입니다. 저희는 이 분포를 다음과 같이 표기할 것입니다.

$$
X \sim U(a, b).
$$

확률 밀도 함수는 다음과 같습니다.

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

먼저 확률 밀도 함수 :eqref:`eq_cont_uniform_pdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

이제, 누적 분포 함수 :eqref:`eq_cont_uniform_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

만약 $X \sim U(a, b)$이면:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

저희는 다음과 같이 균등 확률 변수에서 임의 모양의 배열을 샘플링할 수 있습니다. 기본적으로 $U(0,1)$에서 샘플링하므로, 다른 범위를 원한다면 스케일링해야 한다는 점에 유의하십시오.

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 이항

상황을 약간 더 복잡하게 만들어 *이항* 확률 변수를 검토해 봅시다. 이 확률 변수는 각각 성공할 확률 $p$를 가진 $n$개의 독립 실험의 시퀀스를 수행하고, 얼마나 많은 성공을 볼 것으로 기대하는지 묻는 것에서 유래합니다.

이를 수학적으로 표현해 봅시다. 각 실험은 독립 확률 변수 $X_i$이며, 여기서 저희는 성공을 인코딩하기 위해 $1$을, 실패를 인코딩하기 위해 $0$을 사용할 것입니다. 각각은 확률 $p$로 성공하는 독립 동전 던지기이므로, 저희는 $X_i \sim \textrm{Bernoulli}(p)$라고 말할 수 있습니다. 그러면, 이항 확률 변수는 다음과 같습니다.

$$
X = \sum_{i=1}^n X_i.
$$

이 경우, 저희는 다음과 같이 쓸 것입니다.

$$
X \sim \textrm{Binomial}(n, p).
$$

누적 분포 함수를 얻기 위해서는, 정확히 $k$개의 성공을 얻는 것은 $\binom{n}{k} = \frac{n!}{k!(n-k)!}$가지 방법으로 발생할 수 있으며, 각각은 발생할 확률 $p^k(1-p)^{n-k}$를 가진다는 점에 유의해야 합니다. 따라서 누적 분포 함수는 다음과 같습니다.

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

먼저 확률 질량 함수를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제, 누적 분포 함수 :eqref:`eq_binomial_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

만약 $X \sim \textrm{Binomial}(n, p)$이면:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

이는 $n$개의 베르누이 확률 변수의 합에 대한 기댓값의 선형성과, 독립 확률 변수의 합의 분산이 분산의 합이라는 사실로부터 따릅니다. 이는 다음과 같이 샘플링될 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## 푸아송
이제 사고 실험을 수행해 봅시다. 저희는 버스 정류장에 서 있고 다음 1분 동안 몇 대의 버스가 도착할지 알고 싶습니다. 1분 창 내에 버스가 도착할 확률에 불과한 $X^{(1)} \sim \textrm{Bernoulli}(p)$를 고려하는 것부터 시작해 봅시다. 도시 중심에서 멀리 떨어진 버스 정류장의 경우, 이는 꽤 좋은 근사일 수 있습니다. 저희는 1분 안에 한 대 이상의 버스를 결코 보지 못할 수 있습니다.

그러나, 만약 바쁜 지역에 있다면, 두 대의 버스가 도착할 가능성이 있거나 심지어 가능성이 높습니다. 저희는 처음 30초 또는 두 번째 30초에 대해 저희의 확률 변수를 두 부분으로 분할하여 이를 모델링할 수 있습니다. 이 경우 저희는 다음과 같이 쓸 수 있습니다.

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

여기서 $X^{(2)}$는 총합이고, $X^{(2)}_i \sim \textrm{Bernoulli}(p/2)$입니다. 그러면 총 분포는 $X^{(2)} \sim \textrm{Binomial}(2, p/2)$입니다.

여기서 멈출 이유가 무엇이겠습니까? 저희는 그 1분을 $n$ 부분으로 계속 분할해 봅시다. 위와 같은 추론에 의해, 저희는 다음을 봅니다.

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

이러한 확률 변수를 고려해 보십시오. 이전 절에 의해, 저희는 :eqref:`eq_eq_poisson_approx`가 평균 $\mu_{X^{(n)}} = n(p/n) = p$, 그리고 분산 $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$을 가짐을 압니다. 만약 $n \rightarrow \infty$를 취하면, 이 숫자들이 $\mu_{X^{(\infty)}} = p$로 안정화되고 분산이 $\sigma_{X^{(\infty)}}^2 = p$로 안정화됨을 볼 수 있습니다. 이는 이 무한 세분화 극한에서 저희가 정의할 수 있는 어떤 확률 변수가 *존재할 수 있음*을 나타냅니다.

이는 너무 큰 놀라움으로 다가오지 않아야 합니다. 왜냐하면 실제 세계에서 저희는 단지 버스 도착의 수를 셀 수 있기 때문입니다. 그러나 저희의 수학적 모델이 잘 정의되어 있음을 보는 것은 좋습니다. 이 논의는 *희귀 사건의 법칙*으로 형식화될 수 있습니다.

이 추론을 신중하게 따라가면, 저희는 다음 모델에 도달할 수 있습니다. 만약 확률로 값 $\{0,1,2, \ldots\}$를 취하는 확률 변수라면 저희는 $X \sim \textrm{Poisson}(\lambda)$라고 말할 것입니다.

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

값 $\lambda > 0$는 *비율*(또는 *모양* 매개변수)로 알려져 있으며, 저희가 시간 한 단위에서 기대하는 평균 도착 수를 나타냅니다.

저희는 이 확률 질량 함수를 합산하여 누적 분포 함수를 얻을 수 있습니다.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

먼저 확률 질량 함수 :eqref:`eq_poisson_mass`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

이제, 누적 분포 함수 :eqref:`eq_poisson_cdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

위에서 본 것처럼, 평균과 분산은 특히 간결합니다. 만약 $X \sim \textrm{Poisson}(\lambda)$이면:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

이는 다음과 같이 샘플링될 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## 가우시안
이제 다른, 그러나 관련된 실험을 시도해 봅시다. 다시 $n$개의 독립 $\textrm{Bernoulli}(p)$ 측정 $X_i$를 수행하고 있다고 합시다. 이들의 합의 분포는 $X^{(n)} \sim \textrm{Binomial}(n, p)$입니다. $n$이 증가하고 $p$가 감소함에 따라 극한을 취하는 대신, $p$를 고정한 다음 $n \rightarrow \infty$를 보내봅시다. 이 경우 $\mu_{X^{(n)}} = np \rightarrow \infty$이고 $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$이므로, 이 극한이 잘 정의되어야 한다고 생각할 이유가 없습니다.

그러나, 모든 희망이 사라진 것은 아닙니다! 다음과 같이 정의함으로써 평균과 분산이 잘 동작하도록 만듭시다.

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

이는 평균 0과 분산 1을 가지는 것으로 볼 수 있으며, 따라서 어떤 극한 분포로 수렴할 것이라고 믿는 것이 그럴듯합니다. 만약 이 분포들이 어떻게 보이는지 플롯하면, 저희는 그것이 작동할 것이라고 더욱 확신하게 될 것입니다.

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

한 가지 주목할 점은, 푸아송의 경우와 비교하여, 저희는 이제 표준 편차로 나누고 있는데, 이는 저희가 가능한 결과를 점점 더 작은 영역으로 쥐어짜고 있음을 의미합니다. 이는 저희의 극한이 더 이상 이산이 아니라 오히려 연속이 될 것이라는 표시입니다.

발생하는 것의 유도는 이 문서의 범위를 벗어나지만, *중심 극한 정리*는 $n \rightarrow \infty$일 때 이것이 가우시안 분포(또는 때때로 정규 분포)를 산출할 것이라고 진술합니다. 더 명시적으로, 어떤 $a, b$에 대해서도:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

여기서 저희는 확률 변수가 다음 밀도를 가진다면 주어진 평균 $\mu$와 분산 $\sigma^2$로 정규 분포되어 있다고 말하며, $X \sim \mathcal{N}(\mu, \sigma^2)$로 작성됩니다.

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

먼저 확률 밀도 함수 :eqref:`eq_gaussian_pdf`를 플롯해 봅시다.

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

이제, 누적 분포 함수를 플롯해 봅시다. 이는 이 부록의 범위를 벗어나지만, 가우시안 c.d.f.는 더 기본적인 함수로 닫힌 형식 공식을 가지지 않습니다. 저희는 이 적분을 수치적으로 계산하는 방법을 제공하는 `erf`를 사용할 것입니다.

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

날카로운 눈을 가진 독자들은 이러한 항 중 일부를 인식할 것입니다. 사실, 저희는 :numref:`sec_integral_calculus`에서 이 적분을 만났습니다. 사실 저희는 이 $p_X(x)$가 총 면적 1을 가지고 따라서 유효한 밀도임을 보기 위해 정확히 그 계산이 필요합니다.

저희가 동전 던지기로 작업하기로 선택한 것은 계산을 더 짧게 만들었지만, 그 선택에 대해 근본적인 것은 아무것도 없었습니다. 사실, 어떤 독립적이고 동일하게 분포된 확률 변수 $X_i$의 어떤 모음이든 취하여, 다음을 형성한다면

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

그러면

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

는 대략 가우시안일 것입니다. 그것이 작동하도록 만들기 위해 필요한 추가 요건이 있으며, 가장 일반적으로 $E[X^4] < \infty$이지만, 철학은 분명합니다.

중심 극한 정리는 가우시안이 확률, 통계, 그리고 머신러닝의 기본인 이유입니다. 저희가 측정한 것이 많은 작은 독립 기여의 합이라고 말할 수 있을 때마다, 저희는 측정되는 것이 가우시안에 가까울 것이라고 가정할 수 있습니다.

가우시안에는 훨씬 더 매혹적인 속성이 많이 있으며, 저희는 여기서 하나 더 논의하고 싶습니다. 가우시안은 *최대 엔트로피 분포*로 알려진 것입니다. 저희는 :numref:`sec_information_theory`에서 엔트로피에 대해 더 깊이 들어갈 것이지만, 이 시점에서 알아야 할 모든 것은 그것이 무작위성의 측정값이라는 것입니다. 엄격한 수학적 의미에서, 저희는 가우시안을 고정된 평균과 분산을 가진 확률 변수의 *가장* 무작위적인 선택으로 생각할 수 있습니다. 따라서, 만약 저희의 확률 변수가 어떤 평균과 분산을 가진다는 것을 안다면, 가우시안은 어떤 의미에서 저희가 할 수 있는 분포의 가장 보수적인 선택입니다.

이 절을 마치기 위해, $X \sim \mathcal{N}(\mu, \sigma^2)$이면 다음을 떠올립시다.

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

저희는 아래와 같이 가우시안(또는 표준 정규) 분포에서 샘플링할 수 있습니다.

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 지수족
:label:`subsec_exponential_family`

위에 나열된 모든 분포의 한 가지 공통 속성은 그들 모두가 *지수족*으로 알려진 것에 속한다는 것입니다. 지수족은 밀도가 다음 형태로 표현될 수 있는 분포의 집합입니다.

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

이 정의가 약간 미묘할 수 있으므로, 자세히 검토해 봅시다.

첫째, $h(\mathbf{x})$는 *기저 측도* 또는 *기본 측도*로 알려져 있습니다. 이는 저희가 지수 가중치로 수정하고 있는 측도의 원래 선택으로 볼 수 있습니다.

둘째, *자연 매개변수* 또는 *정준 매개변수*라고 불리는 벡터 $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$가 있습니다. 이는 기본 측도가 어떻게 수정될지를 정의합니다. 자연 매개변수는 $\mathbf{x}= (x_1, x_2, ..., x_n) \in \mathbb{R}^n$의 어떤 함수 $T(\cdot)$에 대한 이러한 매개변수의 내적을 취하고 지수화함으로써 새로운 측도에 들어갑니다. 벡터 $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$는 $\boldsymbol{\eta}$에 대한 *충분 통계*라고 불립니다. 이 이름은 $T(\mathbf{x})$로 나타낸 정보가 확률 밀도를 계산하기에 충분하고 샘플 $\mathbf{x}$로부터의 다른 어떤 정보도 필요하지 않기 때문에 사용됩니다.

셋째, *큐물런트 함수*라고 불리는 $A(\boldsymbol{\eta})$가 있는데, 이는 위 분포 :eqref:`eq_exp_pdf`가 1로 적분되도록 보장합니다. 즉,

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

구체적으로, 가우시안을 고려해 봅시다. $\mathbf{x}$가 단변량 변수라고 가정하면, 저희는 그것이 다음의 밀도를 가짐을 보았습니다.

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

이는 다음과 함께 지수족의 정의와 일치합니다.

* *기저 측도*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *자연 매개변수*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *충분 통계*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, 그리고
* *큐물런트 함수*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

위 항 각각의 정확한 선택은 다소 임의적이라는 점에 주목할 가치가 있습니다. 사실, 중요한 특징은 분포가 정확한 형태 자체가 아니라 이 형태로 표현될 수 있다는 것입니다.

:numref:`subsec_softmax_and_derivatives`에서 암시한 것처럼, 널리 사용되는 기법은 최종 출력 $\mathbf{y}$가 지수족 분포를 따른다고 가정하는 것입니다. 지수족은 머신러닝에서 자주 마주치는 일반적이고 강력한 분포 가족입니다.


## 요약
* 베르누이 확률 변수는 예/아니오 결과를 가진 사건을 모델링하는 데 사용될 수 있습니다.
* 이산 균등 분포는 유한한 가능성 집합에서의 선택을 모델링합니다.
* 연속 균등 분포는 구간에서 선택합니다.
* 이항 분포는 일련의 베르누이 확률 변수를 모델링하고, 성공의 수를 셉니다.
* 푸아송 확률 변수는 희귀 사건의 도착을 모델링합니다.
* 가우시안 확률 변수는 많은 수의 독립 확률 변수를 함께 더한 결과를 모델링합니다.
* 위의 모든 분포는 지수족에 속합니다.

## 연습문제

1. 두 독립 이항 확률 변수 $X, Y \sim \textrm{Binomial}(16, 1/2)$의 차이 $X-Y$인 확률 변수의 표준 편차는 무엇입니까?
2. 만약 푸아송 확률 변수 $X \sim \textrm{Poisson}(\lambda)$를 취하고 $\lambda \rightarrow \infty$일 때 $(X - \lambda)/\sqrt{\lambda}$를 고려하면, 이것이 대략 가우시안이 됨을 보일 수 있습니다. 이것이 왜 의미가 있습니까?
3. $n$ 원소에 대한 두 이산 균등 확률 변수의 합에 대한 확률 질량 함수는 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1099)
:end_tab:
