# 유사도에 의한 어텐션 풀링

:label:`sec_attention-pooling`

이제 어텐션 메커니즘의 주요 구성 요소를 소개했으니, 그것들을 다소 고전적인 환경, 즉 커널 밀도 추정 :cite:`Nadaraya.1964,Watson.1964` 을 통한 회귀와 분류에 사용해 봅시다. 이 우회 경로는 단순히 추가 배경을 제공할 뿐입니다: 완전히 선택 사항이며 필요하면 건너뛸 수 있습니다.
본질적으로 Nadaraya-Watson 추정기는 쿼리 $\mathbf{q}$ 를 키 $\mathbf{k}$ 와 연관시키는 어떤 유사도 커널 $\alpha(\mathbf{q}, \mathbf{k})$ 에 의존합니다. 일반적인 커널 몇 가지는 다음과 같습니다.

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{가우시안;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

저희가 선택할 수 있는 더 많은 옵션이 있습니다. 보다 광범위한 검토와 커널의 선택이 때때로 *Parzen Windows* :cite:`parzen1957consistent` 라고도 불리는 커널 밀도 추정과 어떻게 관련되는지에 대해서는 [위키피디아 문서](https://en.wikipedia.org/wiki/Kernel_(statistics)) 를 참조하십시오. 모든 커널은 휴리스틱이며 튜닝될 수 있습니다. 예를 들어 저희는 전역 기준뿐만 아니라 좌표별 기준으로도 너비를 조정할 수 있습니다. 그럼에도 불구하고 그 모두는 회귀와 분류에서 동일하게 다음 방정식으로 이어집니다.

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

특성과 레이블에 대해 각각 관측치 $(\mathbf{x}_i, y_i)$ 를 가진 (스칼라) 회귀의 경우, $\mathbf{v}_i = y_i$ 는 스칼라이고, $\mathbf{k}_i = \mathbf{x}_i$ 는 벡터이며, 쿼리 $\mathbf{q}$ 는 $f$ 가 평가되어야 할 새로운 위치를 나타냅니다. (다중 클래스) 분류의 경우, 저희는 $\mathbf{v}_i$ 를 얻기 위해 $y_i$ 의 원-핫 인코딩을 사용합니다. 이 추정기의 편리한 속성 중 하나는 학습이 필요하지 않다는 것입니다. 더 나아가, 데이터양이 증가함에 따라 적절히 커널을 좁히면 이 접근법은 일관적입니다 :cite:`mack1982weak`. 즉, 어떤 통계적으로 최적의 해로 수렴할 것입니다. 몇 가지 커널을 살펴보는 것으로 시작해 봅시다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**커널과 데이터**]

이 절에 정의된 모든 커널 $\alpha(\mathbf{k}, \mathbf{q})$ 은 *평행이동 및 회전 불변(translation and rotation invariant)* 입니다. 즉, 저희가 $\mathbf{k}$ 와 $\mathbf{q}$ 를 같은 방식으로 이동하고 회전시키면 $\alpha$ 의 값은 변하지 않습니다. 단순화를 위해 저희는 스칼라 인수 $k, q \in \mathbb{R}$ 을 선택하고 키 $k = 0$ 을 원점으로 선택합니다. 이는 다음을 산출합니다.

```{.python .input}
%%tab all
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

서로 다른 커널은 서로 다른 범위와 매끄러움의 개념에 해당합니다. 예를 들어 Boxcar 커널은 $1$ (또는 다르게 정의된 어떤 초매개변수) 거리 이내의 관측치에만 주의를 기울이며 그것을 무차별적으로 합니다.

Nadaraya-Watson 추정을 실제로 보기 위해, 몇 가지 학습 데이터를 정의해 봅시다. 다음에서 저희는 다음과 같은 의존성을 사용합니다.

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

여기서 $\epsilon$ 은 평균이 0이고 분산이 1인 정규 분포에서 추출됩니다. 저희는 40개의 학습 예시를 추출합니다.

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Nadaraya-Watson 회귀를 통한 어텐션 풀링**]

이제 저희에게 데이터와 커널이 있으므로, 필요한 것은 커널 회귀 추정치를 계산하는 함수입니다. 저희는 또한 약간의 진단을 수행하기 위해 상대적 커널 가중치를 얻고자 한다는 점에 유의하십시오. 따라서 저희는 먼저 모든 학습 특성(공변량) `x_train` 과 모든 검증 특성 `x_val` 사이의 커널을 계산합니다. 이는 행렬을 산출하고, 저희는 이어서 그것을 정규화합니다. 학습 레이블 `y_train` 과 곱했을 때 저희는 추정치를 얻습니다.

:eqref:`eq_attention_pooling` 의 어텐션 풀링을 떠올려 보십시오. 각 검증 특성을 쿼리로, 각 학습 특성-레이블 쌍을 키-값 쌍으로 두십시오. 결과적으로 정규화된 상대적 커널 가중치(아래의 `attention_w`)는 *어텐션 가중치* 입니다.

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalization over keys for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

다양한 커널이 어떤 종류의 추정치를 생성하는지 살펴봅시다.

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

가장 먼저 두드러지는 것은 세 가지 비자명한 커널 모두(가우시안, Boxcar, Epanechikov)가 참 함수에서 그리 멀지 않은 상당히 실용적인 추정치를 생성한다는 점입니다. 자명한 추정치 $f(x) = \frac{1}{n} \sum_i y_i$ 로 이어지는 상수 커널만이 다소 비현실적인 결과를 생성합니다. 어텐션 가중치를 좀 더 자세히 살펴봅시다.

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

이 시각화는 가우시안, Boxcar, Epanechikov의 추정치가 왜 매우 유사한지 분명하게 보여줍니다: 결국 커널의 함수 형태가 다름에도 불구하고 그것들은 매우 유사한 어텐션 가중치로부터 도출됩니다. 이는 항상 그러한지에 대한 질문을 제기합니다.

## [**어텐션 풀링 적응**]

저희는 가우시안 커널을 다른 너비의 커널로 대체할 수 있습니다. 즉, 저희는
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$ 를 사용할 수 있는데, 여기서 $\sigma^2$ 는 커널의 너비를 결정합니다. 이것이 결과에 영향을 주는지 살펴봅시다.

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

분명히 커널이 좁을수록 추정치는 덜 매끄럽습니다. 동시에 그것은 국소적인 변동에 더 잘 적응합니다. 이에 해당하는 어텐션 가중치를 살펴봅시다.

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

예상대로 커널이 좁을수록 큰 어텐션 가중치의 범위가 더 좁습니다. 또한 같은 너비를 선택하는 것이 이상적이지 않을 수 있다는 점도 분명합니다. 실제로 :citet:`Silverman86` 은 국소 밀도에 의존하는 휴리스틱을 제안했습니다. 훨씬 더 많은 그러한 "기법들"이 제안되어 왔습니다. 예를 들어 :citet:`norelli2022asif` 는 교차 모달 이미지와 텍스트 표현을 설계하기 위해 유사한 최근접 이웃 보간 기법을 사용했습니다.

명민한 독자라면 저희가 왜 반세기가 넘은 방법에 대해 이렇게 깊이 다루는지 의아해할 수 있습니다. 첫째, 이는 현대 어텐션 메커니즘의 가장 초기 선구자 중 하나입니다. 둘째, 시각화에 훌륭합니다. 셋째, 그리고 마찬가지로 중요하게, 수작업으로 설계된 어텐션 메커니즘의 한계를 보여줍니다. 훨씬 더 나은 전략은 쿼리와 키에 대한 표현을 학습함으로써 이 메커니즘을 *학습* 하는 것입니다. 이것이 저희가 다음 절에서 시작할 일입니다.


## 요약

Nadaraya-Watson 커널 회귀는 현재 어텐션 메커니즘의 초기 선구자입니다.
이는 분류든 회귀든 학습이나 튜닝이 거의 또는 전혀 없이 직접 사용될 수 있습니다.
어텐션 가중치는 쿼리와 키 간의 유사도(또는 거리)에 따라, 그리고 유사한 관측치가 얼마나 많이 사용 가능한지에 따라 할당됩니다.

## 연습문제

1. Parzen 윈도우 밀도 추정치는 $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$ 로 주어집니다. 이진 분류에 대해 Parzen 윈도우로 얻은 함수 $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$ 이 Nadaraya-Watson 분류와 동등함을 증명하십시오.
1. Nadaraya-Watson 회귀에서 커널 너비에 대한 좋은 값을 학습하기 위해 확률적 경사 하강법을 구현하십시오.
    1. 위의 추정치를 그대로 사용해 $(f(\mathbf{x_i}) - y_i)^2$ 를 직접 최소화하면 어떻게 됩니까? 힌트: $y_i$ 는 $f$ 를 계산하는 데 사용되는 항의 일부입니다.
    1. $f(\mathbf{x}_i)$ 의 추정치에서 $(\mathbf{x}_i, y_i)$ 를 제거하고 커널 너비에 대해 최적화하십시오. 여전히 과적합이 관찰됩니까?
1. 모든 $\mathbf{x}$ 가 단위 구면 위에 있다고 가정합시다. 즉, 모두 $\|\mathbf{x}\| = 1$ 을 만족합니다. 지수 안의 $\|\mathbf{x} - \mathbf{x}_i\|^2$ 항을 단순화할 수 있습니까? 힌트: 나중에 보겠지만 이것은 내적 어텐션과 매우 밀접하게 관련되어 있습니다.
1. :citet:`mack1982weak` 가 Nadaraya-Watson 추정이 일관적임을 증명했다는 것을 기억하십시오. 더 많은 데이터를 얻을수록 어텐션 메커니즘의 척도를 얼마나 빠르게 줄여야 합니까? 답에 대한 어떤 직관을 제시하십시오. 그것이 데이터의 차원에 의존합니까? 어떻게 그렇습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3866)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18026)
:end_tab:
