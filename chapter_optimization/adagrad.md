# Adagrad
:label:`sec_adagrad`

자주 발생하지 않는 특성을 가진 학습 문제부터 살펴봅시다.


## 희소 특성과 학습률

언어 모델을 학습시키고 있다고 상상해 봅시다. 좋은 정확도를 얻기 위해, 저희는 일반적으로 학습을 계속할수록 학습률을 감소시키고 싶어 합니다. 보통 $\mathcal{O}(t^{-\frac{1}{2}})$의 속도 혹은 그보다 느린 속도로요. 이제 희소 특성, 즉 자주 발생하지 않는 특성으로 학습하는 모델을 생각해 봅시다. 이는 자연어에서 흔합니다. 예를 들어, *preconditioning*이라는 단어를 보는 것이 *learning*보다 훨씬 덜 가능성이 큽니다. 그러나 이는 계산 광고와 개인화된 협업 필터링 같은 다른 분야에서도 흔합니다. 결국, 소수의 사람들에게만 관심 있는 많은 것들이 있습니다.

자주 발생하지 않는 특성과 관련된 파라미터들은 이러한 특성들이 발생할 때마다만 의미 있는 업데이트를 받습니다. 감소하는 학습률을 고려할 때, 흔한 특성들에 대한 파라미터는 그들의 최적 값으로 꽤 빠르게 수렴하는 반면, 자주 발생하지 않는 특성들에 대해서는 그들의 최적 값을 결정할 수 있기 전에 충분히 자주 관찰하지 못하는 상황에 처할 수 있습니다. 다시 말해, 학습률이 빈번한 특성에 대해서는 너무 천천히 감소하거나 자주 발생하지 않는 특성에 대해서는 너무 빠르게 감소합니다.

이 문제를 해결하기 위한 가능한 임시 방편은 특정 특성을 본 횟수를 세고 이를 학습률을 조정하기 위한 시계로 사용하는 것입니다. 즉, $\eta = \frac{\eta_0}{\sqrt{t + c}}$ 형식의 학습률을 선택하는 대신 $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$를 사용할 수 있습니다. 여기서 $s(i, t)$는 시간 $t$까지 저희가 관찰한 특성 $i$의 0이 아닌 항목의 수를 셉니다. 이는 의미 있는 오버헤드 없이 사실 꽤 구현하기 쉽습니다. 그러나 희소성이 정확히 있는 것이 아니라 경사도가 종종 매우 작고 드물게만 큰 데이터의 경우 실패합니다. 결국, 관찰된 특성으로 자격이 있는 것과 그렇지 않은 것 사이에 어디에 선을 그어야 할지 불분명합니다.

:citet:`Duchi.Hazan.Singer.2011`의 Adagrad는 다소 조잡한 카운터 $s(i, t)$를 이전에 관찰된 경사도의 제곱들의 집계로 대체함으로써 이 문제를 다룹니다. 특히, 학습률을 조정하는 수단으로 $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$를 사용합니다. 이는 두 가지 이점이 있습니다. 첫째, 저희는 더 이상 언제 경사도가 충분히 큰지 결정할 필요가 없습니다. 둘째, 경사도의 크기에 따라 자동으로 스케일링됩니다. 일상적으로 큰 경사도에 해당하는 좌표는 상당히 축소되는 반면, 작은 경사도를 가진 다른 좌표는 훨씬 더 부드러운 처리를 받습니다. 실제로 이는 계산 광고와 관련 문제에 매우 효과적인 최적화 절차로 이어집니다. 그러나 이는 사전 조건화의 맥락에서 가장 잘 이해되는 Adagrad에 내재된 추가적인 이점의 일부를 가립니다.


## 사전 조건화(Preconditioning)

볼록 최적화 문제는 알고리즘의 특성을 분석하는 데 좋습니다. 결국, 대부분의 비볼록 문제에 대해 의미 있는 이론적 보장을 유도하기는 어렵지만, *직관*과 *통찰*은 종종 이어집니다. $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$를 최소화하는 문제를 살펴봅시다.

:numref:`sec_momentum`에서 보았듯이, 이 문제를 그것의 고유 분해 $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$를 사용해 다시 쓰면 각 좌표를 개별적으로 풀 수 있는 훨씬 단순화된 문제에 도달할 수 있습니다.

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

여기서 저희는 $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$를 사용했고 따라서 $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$입니다. 수정된 문제는 최소화자로 $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$를 가지며 최솟값 $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$를 가집니다. $\boldsymbol{\Lambda}$가 $\mathbf{Q}$의 고윳값을 포함하는 대각 행렬이므로 이것은 계산하기 훨씬 쉽습니다.

$\mathbf{c}$를 약간 교란시키면 $f$의 최소화자에서도 작은 변화만을 발견하기를 바랄 것입니다. 안타깝게도 그렇지 않습니다. $\mathbf{c}$의 약간의 변화는 $\bar{\mathbf{c}}$에서도 똑같이 약간의 변화로 이어지지만, 이는 $f$(그리고 각각 $\bar{f}$)의 최소화자의 경우에는 그렇지 않습니다. 고윳값 $\boldsymbol{\Lambda}_i$가 클 때마다 저희는 $\bar{x}_i$와 $\bar{f}$의 최솟값에서 작은 변화만 볼 것입니다. 반대로, 작은 $\boldsymbol{\Lambda}_i$의 경우 $\bar{x}_i$의 변화는 극적일 수 있습니다. 가장 큰 고윳값과 가장 작은 고윳값 사이의 비율을 최적화 문제의 조건수라고 부릅니다.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

조건수 $\kappa$가 크면 최적화 문제를 정확하게 풀기 어렵습니다. 저희는 값의 큰 동적 범위를 올바르게 얻기 위해 주의해야 합니다. 저희의 분석은 다소 순진하지만 명백한 질문으로 이어집니다. 모든 고윳값이 $1$이 되도록 공간을 왜곡함으로써 문제를 단순히 "고칠" 수는 없을까요? 이론적으로 이는 꽤 쉽습니다. 저희는 단지 문제를 $\mathbf{x}$에서 $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$의 것으로 재조정하기 위해 $\mathbf{Q}$의 고윳값과 고유벡터가 필요합니다. 새 좌표계에서 $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$는 $\|\mathbf{z}\|^2$로 단순화될 수 있습니다. 안타깝게도, 이는 다소 비실용적인 제안입니다. 고윳값과 고유벡터를 계산하는 것은 실제 문제를 푸는 것보다 일반적으로 *훨씬 더* 비쌉니다.

고윳값을 정확하게 계산하는 것은 비쌀 수 있지만, 그것들을 추측하고 다소 근사적으로 계산하는 것도 아무것도 하지 않는 것보다는 이미 훨씬 나을 수 있습니다. 특히, 저희는 $\mathbf{Q}$의 대각 항목을 사용하고 그에 따라 재조정할 수 있습니다. 이것은 고윳값을 계산하는 것보다 *훨씬* 저렴합니다.

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

이 경우 저희는 $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$를 가지며 구체적으로 모든 $i$에 대해 $\tilde{\mathbf{Q}}_{ii} = 1$입니다. 대부분의 경우 이는 조건수를 상당히 단순화합니다. 예를 들어, 저희가 이전에 논의한 경우, 문제가 축 정렬되어 있으므로 이것은 문제를 완전히 제거할 것입니다.

안타깝게도 저희는 또 다른 문제에 직면합니다. 딥러닝에서 저희는 일반적으로 목적 함수의 2차 도함수에 접근조차 할 수 없습니다. $\mathbf{x} \in \mathbb{R}^d$의 경우 미니배치에서조차 2차 도함수는 $\mathcal{O}(d^2)$의 공간과 작업을 계산하는 데 필요할 수 있으므로 실용적으로 불가능합니다. Adagrad의 독창적인 아이디어는 헤시안의 잡기 힘든 대각에 대한 대리로 비교적 계산하기 저렴하고 효과적인 것(경사도 자체의 크기)을 사용하는 것입니다.

이것이 왜 작동하는지 보려면 $\bar{f}(\bar{\mathbf{x}})$를 봅시다. 저희는

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

를 가지는데, 여기서 $\bar{\mathbf{x}}_0$는 $\bar{f}$의 최소화자입니다. 따라서 경사도의 크기는 $\boldsymbol{\Lambda}$와 최적성으로부터의 거리에 모두 의존합니다. $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$가 변하지 않는다면, 이것이 필요한 전부일 것입니다. 결국, 이 경우 경사도 $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$의 크기로 충분합니다. AdaGrad가 확률적 경사 하강법 알고리즘이므로, 저희는 최적성에서도 0이 아닌 분산을 가진 경사도를 볼 것입니다. 그 결과 저희는 경사도의 분산을 헤시안의 척도에 대한 저렴한 대리로 안전하게 사용할 수 있습니다. 철저한 분석은 이 절의 범위를 벗어납니다(수 페이지가 될 것입니다). 자세한 내용은 :cite:`Duchi.Hazan.Singer.2011`을 독자에게 참조합니다.

## 알고리즘

위로부터의 논의를 형식화해 봅시다. 저희는 다음과 같이 과거 경사도 분산을 축적하기 위해 변수 $\mathbf{s}_t$를 사용합니다.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

여기서 연산은 좌표별로 적용됩니다. 즉, $\mathbf{v}^2$는 항목 $v_i^2$를 가집니다. 마찬가지로 $\frac{1}{\sqrt{v}}$는 항목 $\frac{1}{\sqrt{v_i}}$를 가지고 $\mathbf{u} \cdot \mathbf{v}$는 항목 $u_i v_i$를 가집니다. 이전과 같이 $\eta$는 학습률이고 $\epsilon$은 0으로 나누지 않도록 보장하는 가산 상수입니다. 마지막으로, 저희는 $\mathbf{s}_0 = \mathbf{0}$로 초기화합니다.

모멘텀의 경우와 마찬가지로, 저희는 보조 변수를 추적해야 합니다. 이 경우 좌표당 개별 학습률을 허용하기 위해서입니다. 이는 주요 비용이 일반적으로 $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$와 그것의 도함수를 계산하는 것이므로 SGD에 비해 Adagrad의 비용을 크게 증가시키지 않습니다.

$\mathbf{s}_t$에 제곱 경사도를 축적하는 것은 $\mathbf{s}_t$가 본질적으로 선형 속도로 자라는 것을 의미한다는 점에 유의하세요(실제로는 경사도가 초기에 감소하므로 선형보다 다소 느리게). 이는 좌표별 기반으로 조정되긴 하지만 $\mathcal{O}(t^{-\frac{1}{2}})$ 학습률로 이어집니다. 볼록 문제의 경우 이는 완벽하게 적절합니다. 그러나 딥러닝에서는 학습률을 다소 더 천천히 감소시키고 싶을 수 있습니다. 이는 후속 장에서 논의할 여러 Adagrad 변종으로 이어졌습니다. 지금은 이차 볼록 문제에서 어떻게 동작하는지 봅시다. 저희는 이전과 같은 문제를 사용합니다.

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

저희는 이전과 같은 학습률, 즉 $\eta = 0.4$를 사용해 Adagrad를 구현할 것입니다. 보시다시피, 독립 변수의 반복적 궤적은 더 평활합니다. 그러나 $\boldsymbol{s}_t$의 누적 효과로 인해, 학습률은 지속적으로 감쇠하므로, 독립 변수는 반복의 후기 단계에서 그렇게 많이 움직이지 않습니다.

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
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

학습률을 $2$로 증가시키면 훨씬 더 나은 동작을 봅니다. 이는 잡음 없는 경우에서조차 학습률의 감소가 다소 공격적일 수 있음을 이미 나타내며, 파라미터가 적절하게 수렴하도록 보장해야 함을 의미합니다.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## 처음부터 구현하기

모멘텀 방법과 마찬가지로, Adagrad는 파라미터와 동일한 모양의 상태 변수를 유지해야 합니다.

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

:numref:`sec_minibatch_sgd`의 실험과 비교했을 때 저희는 모델을 학습시키기 위해
더 큰 학습률을 사용합니다.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 간결한 구현

알고리즘 `adagrad`의 `Trainer` 인스턴스를 사용해, 저희는 Gluon에서 Adagrad 알고리즘을 호출할 수 있습니다.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## 요약

* Adagrad는 좌표당 기준으로 학습률을 동적으로 감소시킵니다.
* 진전이 얼마나 빠르게 달성되는지를 조정하는 수단으로 경사도의 크기를 사용합니다. 큰 경사도를 가진 좌표는 더 작은 학습률로 보상됩니다.
* 정확한 2차 도함수를 계산하는 것은 메모리와 계산 제약으로 인해 딥러닝 문제에서 일반적으로 실행 불가능합니다. 경사도는 유용한 대리일 수 있습니다.
* 최적화 문제가 다소 고르지 않은 구조를 가지면 Adagrad는 왜곡을 완화하는 데 도움이 될 수 있습니다.
* Adagrad는 자주 발생하지 않는 항에 대해 학습률이 더 천천히 감소해야 하는 희소 특성에 특히 효과적입니다.
* 딥러닝 문제에서 Adagrad는 때때로 학습률을 줄이는 데 너무 공격적일 수 있습니다. 저희는 :numref:`sec_adam`의 맥락에서 이를 완화하기 위한 전략을 논의할 것입니다.

## 연습문제

1. 직교 행렬 $\mathbf{U}$와 벡터 $\mathbf{c}$에 대해 다음이 성립함을 증명하세요: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. 이것이 왜 직교 변수 변경 후 교란의 크기가 변하지 않는다는 것을 의미하나요?
1. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$에 대해, 그리고 목적 함수가 45도 회전된 경우, 즉 $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$에 대해서도 Adagrad를 시도해 보세요. 다르게 동작합니까?
1. 행렬 $\mathbf{M}$의 고윳값 $\lambda_i$가 $j$의 적어도 한 가지 선택에 대해 $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$를 만족함을 명시하는 [Gerschgorin의 원 정리](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)를 증명하세요.
1. Gerschgorin의 정리는 대각 사전 조건화된 행렬 $\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$의 고윳값에 대해 무엇을 말해주나요?
1. 적절한 딥 네트워크, 예를 들어 Fashion-MNIST에 적용된 :numref:`sec_lenet`에 대해 Adagrad를 시도해 보세요.
1. 학습률의 덜 공격적인 감쇠를 달성하기 위해 Adagrad를 어떻게 수정해야 할까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
