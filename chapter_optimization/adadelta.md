# Adadelta
:label:`sec_adadelta`

Adadelta는 AdaGrad의 또 다른 변종입니다(:numref:`sec_adagrad`). 주된 차이는 학습률이 좌표에 적응하는 양을 감소시킨다는 사실에 있습니다. 더욱이, 전통적으로 미래 변화에 대한 보정으로 변화의 양 자체를 사용하기 때문에 학습률이 없는 것으로 언급되었습니다. 이 알고리즘은 :citet:`Zeiler.2012`에서 제안되었습니다. 지금까지의 이전 알고리즘들에 대한 논의를 고려할 때 이것은 꽤 직관적입니다.

## 알고리즘

요약하자면, Adadelta는 두 개의 상태 변수를 사용합니다. 경사도의 2차 모멘트의 새는 평균을 저장하는 $\mathbf{s}_t$와 모델 자체에서 파라미터 변화의 2차 모멘트의 새는 평균을 저장하는 $\Delta\mathbf{x}_t$입니다. 다른 출판물과 구현과의 호환성을 위해 저자의 원래 표기와 명명을 사용한다는 점에 유의하세요(모멘텀, Adagrad, RMSProp, Adadelta에서 동일한 목적을 수행하는 파라미터를 나타내기 위해 다른 그리스 변수를 사용해야 하는 다른 실제 이유는 없습니다).

다음은 Adadelta의 기술적 세부사항입니다. 이번에 적용되는 파라미터가 $\rho$라고 가정하면, 저희는 :numref:`sec_rmsprop`과 유사하게 다음의 새는 업데이트를 얻습니다.

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop`과의 차이는 저희가 재조정된 경사도 $\mathbf{g}_t'$로 업데이트를 수행한다는 것입니다. 즉,

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

그렇다면 재조정된 경사도 $\mathbf{g}_t'$는 무엇입니까? 저희는 다음과 같이 계산할 수 있습니다.

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

여기서 $\Delta \mathbf{x}_{t-1}$는 제곱된 재조정 경사도 $\mathbf{g}_t'$의 새는 평균입니다. 저희는 $\Delta \mathbf{x}_{0}$를 $0$로 초기화하고 각 단계에서 $\mathbf{g}_t'$로 업데이트합니다. 즉,

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

이고 $\epsilon$ ($10^{-5}$와 같은 작은 값)이 수치적 안정성을 유지하기 위해 추가됩니다.



## 구현

Adadelta는 각 변수에 대해 두 개의 상태 변수, $\mathbf{s}_t$와 $\Delta\mathbf{x}_t$를 유지해야 합니다. 이는 다음 구현으로 이어집니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

$\rho = 0.9$를 선택하는 것은 각 파라미터 업데이트에 대해 10의 반감기에 해당합니다. 이는 꽤 잘 작동하는 경향이 있습니다. 저희는 다음 동작을 얻습니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

간결한 구현을 위해 저희는 단순히 고수준 API의 Adadelta 알고리즘을 사용합니다. 이는 훨씬 더 간결한 호출을 위해 다음과 같은 한 줄짜리 코드를 만들어 줍니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it is converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## 요약

* Adadelta는 학습률 파라미터가 없습니다. 대신, 학습률을 적응시키기 위해 파라미터 자체의 변화율을 사용합니다.
* Adadelta는 경사도의 2차 모멘트와 파라미터의 변화를 저장하기 위해 두 개의 상태 변수를 필요로 합니다.
* Adadelta는 적절한 통계의 실행 추정값을 유지하기 위해 새는 평균을 사용합니다.

## 연습문제

1. $\rho$의 값을 조정해 보세요. 어떤 일이 일어나나요?
1. $\mathbf{g}_t'$의 사용 없이 알고리즘을 어떻게 구현할 수 있는지 보이세요. 왜 이것이 좋은 아이디어일 수 있을까요?
1. Adadelta는 정말로 학습률이 없습니까? Adadelta를 깨뜨릴 수 있는 최적화 문제를 찾을 수 있나요?
1. 수렴 동작을 논의하기 위해 Adadelta를 Adagrad와 RMS prop과 비교해 보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
