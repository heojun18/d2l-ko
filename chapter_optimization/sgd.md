# 확률적 경사 하강법(Stochastic Gradient Descent)
:label:`sec_sgd`

앞선 장들에서는 학습 절차에서 확률적 경사 하강법을 계속 사용했지만, 그것이 왜 작동하는지 설명하지는 않았습니다.
이를 좀 더 명확히 하기 위해,
저희는 방금 :numref:`sec_gd`에서
경사 하강법의 기본 원리를 설명했습니다.
이번 절에서는 *확률적 경사 하강법*에 대해
더 자세히 다루겠습니다.

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

## 확률적 경사 업데이트

딥러닝에서 목적 함수는 보통 학습 데이터셋의 각 예제에 대한 손실 함수들의 평균입니다.
$n$개의 예제로 이루어진 학습 데이터셋이 주어졌을 때,
$f_i(\mathbf{x})$를 인덱스 $i$의 학습 예제에 대한 손실 함수라고 가정합니다.
여기서 $\mathbf{x}$는 파라미터 벡터입니다.
그러면 저희는 목적 함수에 도달합니다.

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$에서의 목적 함수의 경사도는 다음과 같이 계산됩니다.

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

경사 하강법을 사용한다면, 각 독립 변수 반복의 계산 비용은 $\mathcal{O}(n)$이며, 이는 $n$에 선형적으로 증가합니다. 따라서 학습 데이터셋이 더 클수록, 각 반복에 대한 경사 하강법의 비용은 더 높아질 것입니다.

확률적 경사 하강법(SGD)은 각 반복에서의 계산 비용을 줄입니다. 확률적 경사 하강법의 각 반복에서, 저희는 데이터 예제에 대한 인덱스 $i\in\{1,\ldots, n\}$를 균등하게 무작위로 샘플링하고, $\mathbf{x}$를 업데이트하기 위해 경사도 $\nabla f_i(\mathbf{x})$를 계산합니다.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

여기서 $\eta$는 학습률입니다. 각 반복의 계산 비용이 경사 하강법의 $\mathcal{O}(n)$에서 상수 $\mathcal{O}(1)$로 떨어지는 것을 볼 수 있습니다. 더욱이, 저희는 확률적 경사도 $\nabla f_i(\mathbf{x})$가 전체 경사도 $\nabla f(\mathbf{x})$의 비편향 추정량이라는 점을 강조하고 싶습니다. 왜냐하면

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

이기 때문입니다. 이는 평균적으로 확률적 경사도가 경사도의 좋은 추정값이라는 것을 의미합니다.

이제 저희는 확률적 경사 하강법을 시뮬레이션하기 위해 평균 0, 분산 1의 무작위 노이즈를 경사도에 더하는 방식으로 경사 하강법과 비교해 볼 것입니다.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

보시다시피, 확률적 경사 하강법에서 변수들의 궤적은 :numref:`sec_gd`의 경사 하강법에서 관찰한 것보다 훨씬 더 잡음이 많습니다. 이는 경사도의 확률적 특성 때문입니다. 즉, 최솟값 근처에 도달했을 때조차도, 저희는 여전히 $\eta \nabla f_i(\mathbf{x})$를 통해 순간 경사도가 주입하는 불확실성에 영향을 받습니다. 50번의 단계 이후에도 품질은 여전히 그다지 좋지 않습니다. 더욱 나쁜 것은, 추가 단계 이후에도 개선되지 않을 것입니다(이를 확인하기 위해 더 많은 단계로 실험해 보시기를 권합니다). 이로 인해 저희에게는 유일한 대안이 남습니다. 학습률 $\eta$를 변경하는 것입니다. 그러나 이를 너무 작게 선택하면 처음에는 의미 있는 진전을 이루지 못할 것입니다. 반면, 너무 크게 선택하면 위에서 본 것처럼 좋은 해를 얻지 못할 것입니다. 이러한 상충하는 목표를 해결하는 유일한 방법은 최적화가 진행됨에 따라 학습률을 *동적으로* 줄이는 것입니다.

이는 또한 `sgd` 스텝 함수에 학습률 함수 `lr`을 추가한 이유이기도 합니다. 위 예제에서는 관련된 `lr` 함수를 상수로 설정했으므로 학습률 스케줄링을 위한 모든 기능은 휴면 상태에 있습니다.

## 동적 학습률

$\eta$를 시간 의존적인 학습률 $\eta(t)$로 대체하면 최적화 알고리즘의 수렴을 제어하는 복잡성이 더해집니다. 특히, $\eta$가 얼마나 빠르게 감소해야 하는지를 알아내야 합니다. 너무 빠르면, 저희는 최적화를 너무 일찍 중단할 것입니다. 너무 느리게 감소시키면, 최적화에 너무 많은 시간을 낭비하게 됩니다. 다음은 시간에 따라 $\eta$를 조정하는 데 사용되는 몇 가지 기본 전략입니다(나중에 더 발전된 전략을 논의할 것입니다).

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

첫 번째 *조각별 상수(piecewise constant)* 시나리오에서는, 예를 들어 최적화의 진전이 멈출 때마다 학습률을 감소시킵니다. 이는 딥 네트워크 학습을 위한 일반적인 전략입니다. 또는 *지수적 감쇠(exponential decay)*에 의해 훨씬 더 공격적으로 감소시킬 수 있습니다. 안타깝게도 이는 종종 알고리즘이 수렴하기 전에 조기 중단으로 이어집니다. 인기 있는 선택은 $\alpha = 0.5$의 *다항 감쇠(polynomial decay)*입니다. 볼록 최적화의 경우 이 속도가 잘 동작함을 보여주는 여러 증명이 있습니다.

지수적 감쇠가 실제로 어떻게 보이는지 봅시다.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

예상한 대로, 파라미터의 분산은 상당히 감소했습니다. 그러나 이는 최적해 $\mathbf{x} = (0, 0)$로 수렴하지 못하는 대가로 옵니다. 1000번의 반복 단계 후에도 저희는 여전히 최적해에서 매우 멀리 있습니다. 실제로 알고리즘은 전혀 수렴하지 못합니다. 반면, 단계 수의 역제곱근으로 학습률이 감쇠하는 다항 감쇠를 사용하면, 50번의 단계만으로 수렴이 더 좋아집니다.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

학습률을 설정하는 방법에는 훨씬 더 많은 선택지가 있습니다. 예를 들어, 작은 속도로 시작해서 빠르게 끌어올린 다음 더 천천히 다시 감소시킬 수 있습니다. 더 작은 학습률과 더 큰 학습률 사이를 번갈아 사용할 수도 있습니다. 이러한 스케줄에는 매우 다양한 종류가 있습니다. 지금은 포괄적인 이론적 분석이 가능한 학습률 스케줄, 즉 볼록 환경에서의 학습률에 초점을 맞춥시다. 일반적인 비볼록 문제의 경우, 일반적으로 비선형 비볼록 문제를 최소화하는 것이 NP 하드이므로 의미 있는 수렴 보장을 얻기가 매우 어렵습니다. 개관을 위해서는 예를 들어 Tibshirani 2015의 훌륭한 [강의 노트](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf)를 참조하세요.



## 볼록 목적에 대한 수렴 분석

볼록 목적 함수에 대한 확률적 경사 하강법의 다음 수렴 분석은 선택 사항이며, 주로 문제에 대한 더 많은 직관을 전달하는 데 사용됩니다.
저희는 가장 단순한 증명 중 하나로 :cite:`Nesterov.Vial.2000`에 제한합니다.
훨씬 더 발전된 증명 기법들이 존재합니다. 예를 들어, 목적 함수가 특히 잘 동작할 때마다 그러합니다.


목적 함수 $f(\boldsymbol{\xi}, \mathbf{x})$가 모든 $\boldsymbol{\xi}$에 대해
$\mathbf{x}$에서 볼록이라고 가정해 봅시다.
더 구체적으로,
저희는 확률적 경사 하강법 업데이트를 고려합니다.

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

여기서 $f(\boldsymbol{\xi}_t, \mathbf{x})$는
단계 $t$에서 어떤 분포로부터 추출된
학습 예제 $\boldsymbol{\xi}_t$에 대한
목적 함수이고 $\mathbf{x}$는 모델 파라미터입니다.
다음을 표기합니다.

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

이는 기댓값 위험이고 $R^*$는 $\mathbf{x}$에 대한 그것의 최솟값입니다. 마지막으로 $\mathbf{x}^*$를 최소화자라고 합시다(저희는 그것이 $\mathbf{x}$가 정의된 영역 내에 존재한다고 가정합니다). 이 경우 저희는 시간 $t$에서의 현재 파라미터 $\mathbf{x}_t$와 위험 최소화자 $\mathbf{x}^*$ 사이의 거리를 추적하고 시간이 지남에 따라 개선되는지 볼 수 있습니다.

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

저희는 확률적 경사도 $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$의 $\ell_2$ 노름이 어떤 상수 $L$에 의해 한정된다고 가정하므로, 다음을 얻습니다.

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


저희는 $\mathbf{x}_t$와 $\mathbf{x}^*$ 사이의 거리가 *기댓값으로* 어떻게 변하는지에 주로 관심이 있습니다. 사실, 특정한 단계 시퀀스에 대해 저희가 어떤 $\boldsymbol{\xi}_t$를 만나느냐에 따라 거리는 증가할 수도 있습니다. 따라서 저희는 내적을 한정해야 합니다.
임의의 볼록 함수 $f$에 대해 모든 $\mathbf{x}$와 $\mathbf{y}$에 대해
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$가 성립하므로,
볼록성에 의해 저희는

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

를 얻습니다. :eqref:`eq_sgd-L`와 :eqref:`eq_sgd-f-xi-xstar`의 두 부등식을 :eqref:`eq_sgd-xt+1-xstar`에 대입하면, 시간 $t+1$에서 파라미터 사이의 거리에 대한 한계를 다음과 같이 얻습니다.

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

이는 현재 손실과 최적 손실의 차이가 $\eta_t L^2/2$를 능가하는 한 진전을 이룬다는 것을 의미합니다. 이 차이는 0으로 수렴하게 되어 있으므로 학습률 $\eta_t$ 또한 *사라져야* 한다는 것이 따릅니다.

다음으로 저희는 :eqref:`eqref_sgd-xt-diff`에 대해 기댓값을 취합니다. 이는

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

를 산출합니다. 마지막 단계는 $t \in \{1, \ldots, T\}$에 대한 부등식의 합을 구하는 것입니다. 합이 망원경적이고 더 낮은 항을 떨어뜨리면 다음을 얻습니다.

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

저희는 $\mathbf{x}_1$이 주어진 것이므로 기댓값을 떨어뜨릴 수 있음을 활용했습니다. 마지막으로 정의합니다.

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

다음

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

이므로, 옌센 부등식에 의해(:eqref:`eq_jensens-inequality`에서 $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$로 설정) 그리고 $R$의 볼록성에 의해 $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$가 성립하므로,

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

이를 부등식 :eqref:`eq_sgd-x1-xstar`에 대입하면 한계를 얻습니다.

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

여기서 $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$는 초기 파라미터 선택과 최종 결과 사이의 거리에 대한 한계입니다. 요약하면, 수렴 속도는 확률적 경사도의 노름이 어떻게 한정되어 있는지($L$)와 초기 파라미터 값이 최적성에서 얼마나 멀리 떨어져 있는지($r$)에 달려 있습니다. 한계가 $\mathbf{x}_T$가 아니라 $\bar{\mathbf{x}}$로 표현된다는 점에 유의하세요. 이는 $\bar{\mathbf{x}}$가 최적화 경로의 평활화된 버전이기 때문입니다.
$r$, $L$, $T$가 알려져 있을 때마다 저희는 학습률 $\eta = r/(L \sqrt{T})$를 선택할 수 있습니다. 이는 상한 $rL/\sqrt{T}$를 산출합니다. 즉, 저희는 $\mathcal{O}(1/\sqrt{T})$의 속도로 최적해로 수렴합니다.





## 확률적 경사도와 유한 표본

지금까지 저희는 확률적 경사 하강법에 대해 다소 무신경하게 이야기해 왔습니다. 저희는 어떤 분포 $p(x, y)$로부터 인스턴스 $x_i$(일반적으로 레이블 $y_i$와 함께)를 추출하고 이를 사용해 어떤 방식으로 모델 파라미터를 업데이트한다고 가정했습니다. 특히, 유한한 표본 크기의 경우, 저희는 단순히 어떤 함수 $\delta_{x_i}$와 $\delta_{y_i}$에 대한 이산 분포 $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$가
저희가 그것에 대해 확률적 경사 하강법을 수행할 수 있게 한다고 주장했습니다.

그러나, 이는 사실 저희가 한 일이 아닙니다. 현재 절의 장난감 예제에서 저희는 단순히 비확률적 경사도에 노이즈를 더했습니다. 즉, 쌍 $(x_i, y_i)$를 가지고 있는 척했습니다. 여기서는 이것이 정당화된다는 것이 밝혀집니다(자세한 논의는 연습문제 참조). 더 우려스러운 것은 이전의 모든 논의에서 저희가 분명히 이렇게 하지 않았다는 점입니다. 대신 저희는 모든 인스턴스를 *정확히 한 번* 반복했습니다. 이것이 왜 더 선호되는지를 보려면 그 반대를 생각해 봅시다. 즉, *복원 추출*로 이산 분포에서 $n$개의 관측값을 샘플링하는 것입니다. 무작위로 요소 $i$를 선택할 확률은 $1/n$입니다. 따라서 *적어도* 한 번 선택할 확률은

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

이 됩니다. 비슷한 추론에 따라 어떤 표본(즉, 학습 예제)을 *정확히 한 번* 뽑을 확률은

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

로 주어집니다. 복원 추출은 *비복원 추출*과 비교해 분산이 증가하고 데이터 효율성이 감소하게 됩니다. 따라서, 실제로는 후자를 수행합니다(그리고 이것이 이 책 전체의 기본 선택입니다). 마지막으로, 학습 데이터셋의 반복적 통과는 그것을 *서로 다른* 무작위 순서로 순회한다는 점에 유의하세요.


## 요약

* 볼록 문제의 경우 광범위한 학습률 선택에 대해 확률적 경사 하강법이 최적해로 수렴할 것임을 증명할 수 있습니다.
* 딥러닝의 경우 일반적으로 그렇지 않습니다. 그러나 볼록 문제의 분석은 최적화에 접근하는 방법, 즉 학습률을 점진적으로 줄이되 너무 빠르지 않게 하는 방법에 대한 유용한 통찰을 제공합니다.
* 학습률이 너무 작거나 너무 클 때 문제가 발생합니다. 실제로는 여러 번의 실험 후에야 적절한 학습률을 찾는 경우가 많습니다.
* 학습 데이터셋에 더 많은 예제가 있을 때, 경사 하강법의 각 반복을 계산하는 데 더 많은 비용이 들기 때문에 이러한 경우에는 확률적 경사 하강법이 선호됩니다.
* 확률적 경사 하강법의 최적성 보장은 일반적으로 비볼록 경우에는 사용할 수 없습니다. 왜냐하면 확인해야 할 지역 최솟값의 수가 지수적일 수 있기 때문입니다.




## 연습문제

1. 확률적 경사 하강법에 대해 다양한 학습률 스케줄과 다양한 반복 횟수로 실험해 보세요. 특히, 반복 횟수의 함수로 최적해 $(0, 0)$로부터의 거리를 그려 보세요.
1. 함수 $f(x_1, x_2) = x_1^2 + 2 x_2^2$에 대해 경사도에 정규 노이즈를 더하는 것이 손실 함수 $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$를 최소화하는 것과 동등함을 증명하세요. 여기서 $\mathbf{x}$는 정규분포에서 추출됩니다.
1. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$에서 복원 추출로 샘플링할 때와 비복원 추출로 샘플링할 때 확률적 경사 하강법의 수렴을 비교해 보세요.
1. 어떤 경사도(또는 그것과 관련된 어떤 좌표)가 다른 모든 경사도보다 지속적으로 클 때 확률적 경사 하강법 솔버를 어떻게 변경하시겠습니까?
1. $f(x) = x^2 (1 + \sin x)$라고 가정해 봅시다. $f$는 얼마나 많은 지역 최솟값을 가집니까? 그것을 최소화하기 위해 모든 지역 최솟값을 평가해야 하는 방식으로 $f$를 변경할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
