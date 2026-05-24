# 모멘텀(Momentum)
:label:`sec_momentum`

:numref:`sec_sgd`에서 저희는 확률적 경사 하강법을 수행할 때, 즉 경사도의 잡음이 있는 변종만 사용할 수 있을 때 어떤 일이 일어나는지 검토했습니다. 특히, 잡음이 있는 경사도의 경우 잡음에 직면하여 학습률을 선택할 때 추가로 주의를 기울여야 한다는 점을 알았습니다. 너무 빠르게 감소시키면, 수렴이 멈춥니다. 너무 관대하면, 잡음이 저희를 최적성에서 계속 멀어지게 하기 때문에 충분히 좋은 해로 수렴하지 못합니다.

## 기본

이 절에서는 더 효과적인 최적화 알고리즘, 특히 실제로 흔한 특정 유형의 최적화 문제에 대해 탐구할 것입니다.


### 새는 평균(Leaky Averages)

이전 절은 저희가 미니배치 SGD를 계산을 가속화하는 수단으로 논의하는 것을 보았습니다. 또한 경사도를 평균화하면 분산의 양이 줄어든다는 좋은 부수 효과가 있었습니다. 미니배치 확률적 경사 하강법은 다음과 같이 계산할 수 있습니다.

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

표기를 단순하게 유지하기 위해, 여기서 저희는 $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$를 시간 $t-1$에서 업데이트된 가중치를 사용한 표본 $i$에 대한 확률적 경사 하강법으로 사용했습니다.
미니배치에서 경사도를 평균화하는 것 이상으로 분산 감소의 효과로부터 이익을 얻을 수 있다면 좋을 것입니다. 이 작업을 수행하는 한 가지 옵션은 경사도 계산을 "새는 평균"으로 대체하는 것입니다.

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

여기서 $\beta \in (0, 1)$입니다. 이는 사실상 순간 경사도를 여러 *과거* 경사도에 걸쳐 평균화된 것으로 대체합니다. $\mathbf{v}$는 *속도(velocity)*라고 부릅니다. 이는 목적 함수 환경을 내려가는 무거운 공이 과거 힘들을 적분하는 방식과 유사하게 과거 경사도들을 축적합니다. 일어나고 있는 일을 더 자세히 보기 위해 $\mathbf{v}_t$를 재귀적으로 다음과 같이 확장해 봅시다.

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

큰 $\beta$는 장기 평균에 해당하고, 작은 $\beta$는 경사도 방법에 비해 약간의 보정에 해당합니다. 새로운 경사도 대체는 더 이상 특정 인스턴스에서 가장 가파른 하강 방향을 가리키지 않고 오히려 과거 경사도들의 가중 평균 방향을 가리킵니다. 이는 실제로 경사도를 계산하는 비용 없이 배치에 걸친 평균화의 이점 대부분을 실현할 수 있게 합니다. 저희는 이 평균화 절차를 나중에 더 자세히 다시 살펴볼 것입니다.

위의 추론은 이제 모멘텀이 포함된 경사도와 같은 *가속화된* 경사 방법으로 알려진 것의 기초를 형성했습니다. 그것들은 최적화 문제가 잘 조건화되어 있지 않을 때(즉, 어떤 방향에서는 다른 방향보다 진전이 훨씬 더 느린, 좁은 협곡과 같은 곳) 훨씬 더 효과적이라는 추가적인 이점을 누립니다. 더욱이, 그것들은 더 안정적인 하강 방향을 얻기 위해 후속 경사도에 걸쳐 평균을 낼 수 있게 합니다. 실제로, 잡음 없는 볼록 문제에서도 가속화의 측면은 모멘텀이 작동하는 그리고 그것이 그렇게 잘 작동하는 핵심 이유 중 하나입니다.

예상한 대로, 그 효능 덕분에 모멘텀은 딥러닝과 그 너머의 최적화에서 잘 연구된 주제입니다. 깊이 있는 분석과 인터랙티브 애니메이션은 예를 들어 :citet:`Goh.2017`의 아름다운 [해설 글](https://distill.pub/2017/momentum/)을 참조하세요. 이것은 :citet:`Polyak.1964`에 의해 제안되었습니다. :citet:`Nesterov.2018`는 볼록 최적화의 맥락에서 자세한 이론적 논의를 가지고 있습니다. 딥러닝에서 모멘텀은 오랫동안 유익한 것으로 알려져 왔습니다. 자세한 내용은 예를 들어 :citet:`Sutskever.Martens.Dahl.ea.2013`의 논의를 참조하세요.

### 잘 조건화되지 않은 문제

모멘텀 방법의 기하학적 특성을 더 잘 이해하기 위해, 상당히 덜 유쾌한 목적 함수로 경사 하강법을 다시 살펴봅니다. :numref:`sec_gd`에서 저희가 $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, 즉 적당히 왜곡된 타원체 목적을 사용했음을 기억하세요. 저희는 $x_1$ 방향으로 늘려서 이 함수를 더 왜곡합니다.

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

이전과 같이 $f$는 $(0, 0)$에서 최솟값을 가집니다. 이 함수는 $x_1$ 방향으로 *매우* 평평합니다. 이 새 함수에 대해 이전처럼 경사 하강법을 수행할 때 어떤 일이 일어나는지 봅시다. 저희는 $0.4$의 학습률을 선택합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

구성에 따라, $x_2$ 방향의 경사도는 수평 $x_1$ 방향보다 *훨씬* 더 높고 훨씬 더 빠르게 변합니다. 따라서 저희는 두 가지 바람직하지 않은 선택 사이에 갇혀 있습니다. 작은 학습률을 선택하면 해가 $x_2$ 방향에서 발산하지 않지만, $x_1$ 방향에서의 느린 수렴에 시달리게 됩니다. 반대로, 큰 학습률을 사용하면 $x_1$ 방향에서는 빠르게 진전하지만 $x_2$에서는 발산합니다. 아래 예는 학습률을 $0.4$에서 $0.6$으로 약간 증가시킨 후에도 어떤 일이 일어나는지 보여줍니다. $x_1$ 방향에서의 수렴은 개선되지만 전체 해의 품질은 훨씬 더 나쁩니다.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### 모멘텀 방법

모멘텀 방법은 위에 설명된 경사 하강법 문제를 풀 수 있게 해줍니다. 위의 최적화 추적을 보면, 과거에 걸쳐 경사도를 평균화하는 것이 잘 작동할 것이라는 직관을 얻을 수 있습니다. 결국, $x_1$ 방향에서 이는 잘 정렬된 경사도들을 집계할 것이므로 매 단계에서 저희가 커버하는 거리가 증가합니다. 반대로, 경사도가 진동하는 $x_2$ 방향에서, 집계된 경사도는 서로 상쇄되는 진동으로 인해 스텝 크기를 감소시킬 것입니다.
경사도 $\mathbf{g}_t$ 대신 $\mathbf{v}_t$를 사용하면 다음과 같은 업데이트 방정식이 나옵니다.

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

$\beta = 0$의 경우 일반적인 경사 하강법을 복구한다는 점에 유의하세요. 수학적 특성에 대해 더 깊이 들어가기 전에, 알고리즘이 실제로 어떻게 동작하는지 간단히 살펴봅시다.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

보시다시피, 이전에 저희가 사용한 것과 동일한 학습률에서도, 모멘텀은 여전히 잘 수렴합니다. 모멘텀 파라미터를 감소시키면 어떤 일이 일어나는지 봅시다. 이를 $\beta = 0.25$로 반감시키면 거의 수렴하지 않는 궤적이 나옵니다. 그럼에도 불구하고, 모멘텀 없이(해가 발산할 때)보다 훨씬 낫습니다.

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

저희는 모멘텀을 확률적 경사 하강법과 결합할 수 있고, 특히 미니배치 확률적 경사 하강법과 결합할 수 있다는 점에 유의하세요. 유일한 변경 사항은 그 경우 경사도 $\mathbf{g}_{t, t-1}$을 $\mathbf{g}_t$로 대체한다는 것입니다. 마지막으로, 편의를 위해 저희는 시간 $t=0$에서 $\mathbf{v}_0 = 0$로 초기화합니다. 새는 평균화가 실제로 업데이트에 무엇을 하는지 살펴봅시다.

### 효과적인 표본 가중치

$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$임을 기억하세요. 극한에서 항들은 $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$로 합쳐집니다. 다시 말해, 경사 하강법이나 확률적 경사 하강법에서 크기 $\eta$의 스텝을 밟는 대신, 저희는 크기 $\frac{\eta}{1-\beta}$의 스텝을 밟으면서 동시에 잠재적으로 훨씬 더 잘 동작하는 하강 방향을 다룹니다. 이는 하나에 두 가지 이점입니다. 다양한 $\beta$의 선택에 대해 가중치가 어떻게 동작하는지 설명하기 위해 아래 도표를 고려하세요.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 실용적인 실험

모멘텀이 실제로 어떻게 동작하는지, 즉 적절한 옵티마이저의 맥락 내에서 사용될 때 어떻게 동작하는지 봅시다. 이를 위해 좀 더 확장 가능한 구현이 필요합니다.

### 처음부터 구현하기

(미니배치) 확률적 경사 하강법과 비교했을 때, 모멘텀 방법은 보조 변수 집합, 즉 속도를 유지할 필요가 있습니다. 이는 경사도(그리고 최적화 문제의 변수)와 동일한 모양을 가집니다. 아래 구현에서 저희는 이러한 변수들을 `states`라고 부릅니다.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

이것이 실제로 어떻게 동작하는지 봅시다.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

모멘텀 하이퍼파라미터 `momentum`을 0.9로 증가시키면, 이는 $\frac{1}{1 - 0.9} = 10$의 상당히 더 큰 효과적인 표본 크기에 해당합니다. 사정을 통제하기 위해 학습률을 $0.01$로 약간 줄입니다.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

학습률을 더 줄이면 비평활 최적화 문제의 어떤 문제도 해결합니다. 그것을 $0.005$로 설정하면 좋은 수렴 특성이 나옵니다.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 간결한 구현

표준 `sgd` 솔버가 이미 모멘텀이 내장되어 있으므로 Gluon에서 할 일이 거의 없습니다. 매칭 파라미터를 설정하면 매우 유사한 궤적이 나옵니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 이론적 분석

지금까지 $f(x) = 0.1 x_1^2 + 2 x_2^2$의 2차원 예제는 다소 인위적인 것처럼 보였습니다. 저희는 이제 이것이 적어도 볼록 이차 목적 함수를 최소화하는 경우, 누군가가 만날 수 있는 문제 유형의 꽤 대표적인 것임을 볼 것입니다.

### 이차 볼록 함수

함수를 고려해 봅시다.

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

이는 일반적인 이차 함수입니다. 양의 정부호 행렬 $\mathbf{Q} \succ 0$의 경우, 즉 양의 고윳값을 가진 행렬의 경우, 이는 $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$에서 최솟값 $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$의 최소화자를 가집니다. 따라서 저희는 $h$를 다음과 같이 다시 쓸 수 있습니다.

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

경사도는 $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$로 주어집니다. 즉, 그것은 $\mathbf{x}$와 최소화자 사이의 거리에 $\mathbf{Q}$를 곱한 것으로 주어집니다. 따라서 속도도 $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ 항들의 선형 결합입니다.

$\mathbf{Q}$가 양의 정부호이므로, 직교(회전) 행렬 $\mathbf{O}$와 양의 고윳값의 대각 행렬 $\boldsymbol{\Lambda}$에 대해 $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$를 통해 고유 시스템으로 분해될 수 있습니다. 이를 통해 저희는 $\mathbf{x}$에서 $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$로의 변수 변경을 수행하여 훨씬 단순화된 표현을 얻을 수 있습니다.

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

여기서 $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$입니다. $\mathbf{O}$가 단지 직교 행렬이므로 이는 의미 있는 방식으로 경사도를 교란시키지 않습니다. $\mathbf{z}$로 표현하면 경사 하강법은 다음이 됩니다.

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

이 표현의 중요한 사실은 경사 하강법이 서로 다른 고유공간들 사이에서 *섞이지 않는다*는 것입니다. 즉, $\mathbf{Q}$의 고유 시스템 측면에서 표현될 때 최적화 문제는 좌표별 방식으로 진행됩니다. 이는 다음에 대해서도 성립합니다.

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

이렇게 함으로써 저희는 다음 정리를 방금 증명했습니다. 볼록 이차 함수에 대한 모멘텀이 있는 그리고 없는 경사 하강법은 이차 행렬의 고유벡터 방향으로의 좌표별 최적화로 분해됩니다.

### 스칼라 함수

위의 결과를 고려할 때 함수 $f(x) = \frac{\lambda}{2} x^2$를 최소화할 때 어떤 일이 일어나는지 봅시다. 경사 하강법의 경우 저희는

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

를 가집니다. $|1 - \eta \lambda| < 1$일 때마다 이 최적화는 $t$단계 후에 $x_t = (1 - \eta \lambda)^t x_0$이므로 지수적 속도로 수렴합니다. 이는 학습률 $\eta$를 $\eta \lambda = 1$까지 증가시킴에 따라 수렴 속도가 초기에 어떻게 개선되는지 보여줍니다. 그 이후로는 발산하고 $\eta \lambda > 2$의 경우 최적화 문제가 발산합니다.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

모멘텀의 경우 수렴을 분석하기 위해 저희는 업데이트 방정식을 두 개의 스칼라로 다시 쓰는 것으로 시작합니다. 하나는 $x$용이고 하나는 속도 $v$용입니다. 이는 다음과 같습니다.

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

저희는 수렴 동작을 지배하는 $2 \times 2$를 표시하기 위해 $\mathbf{R}$을 사용했습니다. $t$단계 후 초기 선택 $[v_0, x_0]$은 $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$이 됩니다. 따라서, 수렴 속도를 결정하는 것은 $\mathbf{R}$의 고윳값에 달려 있습니다. 훌륭한 애니메이션은 :citet:`Goh.2017`의 [Distill 포스트](https://distill.pub/2017/momentum/)를, 자세한 분석은 :citet:`Flammarion.Bach.2015`를 참조하세요. $0 < \eta \lambda < 2 + 2 \beta$일 때 속도가 수렴함을 보일 수 있습니다. 이는 경사 하강법에 대한 $0 < \eta \lambda < 2$와 비교했을 때 실현 가능한 파라미터의 더 큰 범위입니다. 또한 일반적으로 $\beta$의 큰 값이 바람직함을 시사합니다. 더 자세한 내용은 상당한 양의 기술적 세부사항을 필요로 하며 관심 있는 독자가 원본 출판물을 참조할 것을 제안합니다.

## 요약

* 모멘텀은 경사도를 과거 경사도에 걸친 새는 평균으로 대체합니다. 이는 수렴을 상당히 가속화합니다.
* 잡음 없는 경사 하강법과 (잡음이 있는) 확률적 경사 하강법 모두에 바람직합니다.
* 모멘텀은 확률적 경사 하강법에서 훨씬 더 발생하기 쉬운 최적화 과정의 멈춤을 방지합니다.
* 효과적인 경사도 수는 과거 데이터의 지수적 가중치 감쇠로 인해 $\frac{1}{1-\beta}$로 주어집니다.
* 볼록 이차 문제의 경우 이는 명시적으로 자세히 분석될 수 있습니다.
* 구현은 매우 간단하지만 추가 상태 벡터(속도 $\mathbf{v}$)를 저장해야 합니다.

## 연습문제

1. 모멘텀 하이퍼파라미터와 학습률의 다른 조합을 사용하고 다양한 실험 결과를 관찰하고 분석해 보세요.
1. 여러 고윳값을 가진 이차 문제, 즉 $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, 예를 들어 $\lambda_i = 2^{-i}$에 대해 경사 하강법과 모멘텀을 시도해 보세요. 초기화 $x_i = 1$에 대해 $x$의 값이 어떻게 감소하는지 그려 보세요.
1. $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$에 대한 최솟값과 최소화자를 유도해 보세요.
1. 모멘텀이 있는 확률적 경사 하강법을 수행할 때 무엇이 변하나요? 모멘텀이 있는 미니배치 확률적 경사 하강법을 사용할 때 어떤 일이 일어나나요? 파라미터로 실험해 보세요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
