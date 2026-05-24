# RMSProp
:label:`sec_rmsprop`


:numref:`sec_adagrad`의 핵심 문제 중 하나는 학습률이 사실상 $\mathcal{O}(t^{-\frac{1}{2}})$의 사전 정의된 스케줄로 감소한다는 것입니다. 이것이 일반적으로 볼록 문제에는 적절하지만, 딥러닝에서 마주치는 것과 같은 비볼록 문제에는 이상적이지 않을 수 있습니다. 그럼에도 불구하고, Adagrad의 좌표별 적응성은 사전 조건자로서 매우 바람직합니다.

:citet:`Tieleman.Hinton.2012`는 좌표 적응적 학습률에서 속도 스케줄링을 분리하기 위한 간단한 수정으로 RMSProp 알고리즘을 제안했습니다. 문제는 Adagrad가 경사도 $\mathbf{g}_t$의 제곱을 상태 벡터 $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$에 축적한다는 것입니다. 그 결과 $\mathbf{s}_t$는 정규화가 없기 때문에 끝없이 계속 자라며, 알고리즘이 수렴할 때 본질적으로 선형적으로 자랍니다.

이 문제를 고치는 한 가지 방법은 $\mathbf{s}_t / t$를 사용하는 것입니다. $\mathbf{g}_t$의 합리적인 분포에 대해 이는 수렴할 것입니다. 안타깝게도, 절차가 값들의 전체 궤적을 기억하기 때문에 극한 동작이 중요해지기 시작할 때까지 매우 긴 시간이 걸릴 수 있습니다. 대안은 저희가 모멘텀 방법에서 사용한 것과 같은 방식으로 새는 평균을 사용하는 것입니다. 즉, 어떤 파라미터 $\gamma > 0$에 대해 $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$입니다. 다른 모든 부분을 변경하지 않고 유지하면 RMSProp이 나옵니다.

## 알고리즘

방정식을 자세히 적어 봅시다.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

상수 $\epsilon > 0$은 일반적으로 $10^{-6}$로 설정되어 0으로 나누거나 지나치게 큰 스텝 크기로부터 고통을 받지 않도록 보장합니다. 이 확장이 주어지면 저희는 이제 좌표당 기준으로 적용되는 스케일링과 독립적으로 학습률 $\eta$를 제어할 자유가 있습니다. 새는 평균의 측면에서, 저희는 모멘텀 방법의 경우에 이전에 적용했던 것과 동일한 추론을 적용할 수 있습니다. $\mathbf{s}_t$의 정의를 확장하면 다음이 나옵니다.

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

:numref:`sec_momentum`에서 이전과 같이 저희는 $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$를 사용합니다. 따라서 가중치의 합은 관측의 반감기 $\gamma^{-1}$로 $1$로 정규화됩니다. 다양한 $\gamma$ 선택에 대해 지난 40 시간 단계의 가중치를 시각화해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## 처음부터 구현하기

이전처럼 저희는 RMSProp의 궤적을 관찰하기 위해 이차 함수 $f(\mathbf{x})=0.1x_1^2+2x_2^2$을 사용합니다. :numref:`sec_adagrad`에서 저희가 0.4의 학습률로 Adagrad를 사용했을 때, 학습률이 너무 빠르게 감소했기 때문에 변수가 알고리즘의 후기 단계에서 매우 느리게만 움직였음을 기억하세요. $\eta$가 별도로 제어되므로 RMSProp에서는 이 일이 일어나지 않습니다.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

다음으로, 딥 네트워크에서 사용될 RMSProp을 구현합니다. 이것은 똑같이 간단합니다.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

저희는 초기 학습률을 0.01로, 가중치 항 $\gamma$를 0.9로 설정합니다. 즉, $\mathbf{s}$는 평균적으로 지난 $1/(1-\gamma) = 10$개의 제곱 경사도 관측값에 걸쳐 집계됩니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 간결한 구현

RMSProp은 다소 인기 있는 알고리즘이므로 `Trainer` 인스턴스에도 사용할 수 있습니다. 저희가 해야 할 일은 단지 `rmsprop`라는 이름의 알고리즘을 사용해 그것을 인스턴스화하고, $\gamma$를 `gamma1` 파라미터에 할당하는 것뿐입니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## 요약

* RMSProp은 둘 다 계수를 스케일링하기 위해 경사도의 제곱을 사용한다는 점에서 Adagrad와 매우 유사합니다.
* RMSProp은 모멘텀과 새는 평균화를 공유합니다. 그러나 RMSProp은 계수별 사전 조건자를 조정하기 위해 이 기법을 사용합니다.
* 학습률은 실제로 실험자가 스케줄링해야 합니다.
* 계수 $\gamma$는 좌표당 스케일을 조정할 때 이력이 얼마나 긴지를 결정합니다.

## 연습문제

1. $\gamma = 1$로 설정하면 실험적으로 어떤 일이 일어나나요? 왜 그렇습니까?
1. $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$를 최소화하도록 최적화 문제를 회전시키세요. 수렴에 어떤 일이 일어나나요?
1. Fashion-MNIST에서의 학습과 같은 실제 머신러닝 문제에서 RMSProp에 어떤 일이 일어나는지 시도해 보세요. 학습률을 조정하는 다양한 선택으로 실험해 보세요.
1. 최적화가 진행됨에 따라 $\gamma$를 조정하고 싶을까요? RMSProp은 이에 얼마나 민감합니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:
