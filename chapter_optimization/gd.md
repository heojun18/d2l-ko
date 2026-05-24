# 경사 하강법(Gradient Descent)
:label:`sec_gd`

이 절에서는 *경사 하강법*의 기본 개념을 소개합니다.
딥러닝에서 직접 사용되는 경우는 드물지만, 경사 하강법을 이해하는 것은 확률적 경사 하강법 알고리즘을 이해하는 데 핵심적입니다.
예를 들어, 학습률이 지나치게 커서 최적화 문제가 발산할 수도 있습니다. 이 현상은 이미 경사 하강법에서 나타날 수 있습니다. 마찬가지로, 사전 조건화(preconditioning)는 경사 하강법에서 흔한 기법이며 더 발전된 알고리즘으로도 이어집니다.
간단한 특수 사례부터 시작해 봅시다.


## 1차원 경사 하강법

1차원에서의 경사 하강법은 경사 하강법 알고리즘이 왜 목적 함수의 값을 줄일 수 있는지 설명하는 훌륭한 예입니다. 어떤 연속 미분 가능한 실수값 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$를 생각해 봅시다. 테일러 전개를 사용하면 다음을 얻습니다.

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

즉, 1차 근사로 $f(x+\epsilon)$는 $x$에서의 함수 값 $f(x)$와 1차 도함수 $f'(x)$로 주어집니다. 작은 $\epsilon$에 대해 음의 경사도 방향으로 이동하면 $f$가 감소할 것이라고 가정하는 것은 무리가 아닙니다. 단순하게 유지하기 위해 고정된 스텝 크기 $\eta > 0$를 고르고 $\epsilon = -\eta f'(x)$를 선택합니다. 이것을 위의 테일러 전개에 대입하면 다음을 얻습니다.

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

도함수 $f'(x) \neq 0$이 사라지지 않으면 $\eta f'^2(x)>0$이므로 진전을 이루게 됩니다. 더욱이 고차항이 무의미해지도록 $\eta$를 항상 충분히 작게 선택할 수 있습니다. 따라서 저희는

$$f(x - \eta f'(x)) \lessapprox f(x).$$

에 도달합니다. 이는 만약 저희가

$$x \leftarrow x - \eta f'(x)$$

를 사용해 $x$를 반복하면, 함수 $f(x)$의 값이 감소할 수 있음을 의미합니다. 따라서 경사 하강법에서는 먼저 초기 값 $x$와 상수 $\eta > 0$를 선택한 다음, 정지 조건에 도달할 때까지 그것들을 사용해 $x$를 지속적으로 반복합니다. 예를 들어 경사도의 크기 $|f'(x)|$가 충분히 작아지거나 반복 횟수가 특정 값에 도달했을 때 그러합니다.

단순성을 위해 경사 하강법을 어떻게 구현하는지 설명하기 위해 목적 함수 $f(x)=x^2$을 선택합니다. $x=0$이 $f(x)$를 최소화하는 해임을 알고 있지만, $x$가 어떻게 변하는지 관찰하기 위해 이 단순한 함수를 여전히 사용합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

다음으로, $x=10$을 초기 값으로 사용하고 $\eta=0.2$로 가정합니다. 경사 하강법을 사용해 $x$를 10번 반복하면, 결국 $x$의 값이 최적해에 접근하는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$를 최적화하는 과정은 다음과 같이 그릴 수 있습니다.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 학습률
:label:`subsec_gd-learningrate`

학습률 $\eta$는 알고리즘 설계자가 설정할 수 있습니다. 너무 작은 학습률을 사용하면 $x$가 매우 느리게 업데이트되어 더 나은 해를 얻기 위해 더 많은 반복이 필요하게 됩니다. 이러한 경우 어떤 일이 일어나는지 보여주기 위해 $\eta = 0.05$에 대한 동일한 최적화 문제의 진전을 살펴봅시다. 보시다시피, 10번의 단계 이후에도 여전히 최적해에서 매우 멀리 떨어져 있습니다.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

반대로, 지나치게 높은 학습률을 사용하면 $\left|\eta f'(x)\right|$가 1차 테일러 전개 공식에 너무 클 수 있습니다. 즉, :eqref:`gd-taylor-2`에서 $\mathcal{O}(\eta^2 f'^2(x))$ 항이 유의미해질 수 있습니다. 이 경우, $x$의 반복이 $f(x)$의 값을 낮출 수 있다는 것을 보장할 수 없습니다. 예를 들어, 학습률을 $\eta=1.1$로 설정하면 $x$는 최적해 $x=0$을 지나쳐 점차 발산합니다.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 지역 최솟값

비볼록 함수의 경우 어떤 일이 일어나는지 설명하기 위해 어떤 상수 $c$에 대한 $f(x) = x \cdot \cos(cx)$의 경우를 생각해 봅시다. 이 함수는 무한히 많은 지역 최솟값을 가집니다. 저희가 선택한 학습률과 문제의 조건이 얼마나 좋은지에 따라, 많은 해들 중 하나로 귀결될 수 있습니다. 아래 예는 (비현실적으로) 높은 학습률이 어떻게 좋지 않은 지역 최솟값으로 이어지는지 보여줍니다.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 다변량 경사 하강법

이제 일변량 경우에 대한 더 나은 직관을 얻었으니, $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$인 상황을 생각해 봅시다. 즉, 목적 함수 $f: \mathbb{R}^d \to \mathbb{R}$는 벡터를 스칼라로 매핑합니다. 그에 대응하여 그것의 경사도도 다변량입니다. 이는 $d$개의 편도함수로 구성된 벡터입니다.

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

경사도의 각 편도함수 요소 $\partial f(\mathbf{x})/\partial x_i$는 입력 $x_i$에 대한 $\mathbf{x}$에서의 $f$의 변화율을 나타냅니다. 일변량 경우와 마찬가지로, 저희는 무엇을 해야 할지에 대한 아이디어를 얻기 위해 다변량 함수에 대한 대응되는 테일러 근사를 사용할 수 있습니다. 특히, 다음이 성립합니다.

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

다시 말해, $\boldsymbol{\epsilon}$에 대한 2차 항까지 보면, 가장 가파른 하강 방향은 음의 경사도 $-\nabla f(\mathbf{x})$로 주어집니다. 적절한 학습률 $\eta > 0$를 선택하면 전형적인 경사 하강법 알고리즘이 나옵니다.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

이 알고리즘이 실제로 어떻게 동작하는지 보기 위해, 2차원 벡터 $\mathbf{x} = [x_1, x_2]^\top$를 입력으로 받고 스칼라를 출력하는 목적 함수 $f(\mathbf{x})=x_1^2+2x_2^2$를 만들어 봅시다. 경사도는 $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$로 주어집니다. 초기 위치 $[-5, -2]$에서 경사 하강법에 의한 $\mathbf{x}$의 궤적을 관찰할 것입니다.

먼저, 두 개의 도우미 함수가 더 필요합니다. 첫 번째는 업데이트 함수를 사용하여 초기 값에 20번 적용합니다. 두 번째 도우미는 $\mathbf{x}$의 궤적을 시각화합니다.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

다음으로, 학습률 $\eta = 0.1$에 대해 최적화 변수 $\mathbf{x}$의 궤적을 관찰합니다. 20번의 단계 후 $\mathbf{x}$의 값이 $[0, 0]$의 최솟값에 접근하는 것을 볼 수 있습니다. 진전은 다소 느리지만 꽤 잘 동작합니다.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 적응적 방법(Adaptive Methods)

:numref:`subsec_gd-learningrate`에서 보았듯이, 학습률 $\eta$를 "딱 맞게" 설정하는 것은 까다롭습니다. 너무 작게 선택하면 거의 진전이 없습니다. 너무 크게 선택하면 해가 진동하며 최악의 경우 발산할 수도 있습니다. $\eta$를 자동으로 결정하거나 학습률을 선택할 필요 자체를 없앨 수 있다면 어떨까요?
목적 함수의 값과 경사도뿐 아니라 그것의 *곡률(curvature)*까지 살펴보는 2차 방법들이 이 경우 도움이 될 수 있습니다. 비록 이러한 방법들은 계산 비용 때문에 딥러닝에 직접 적용될 수는 없지만, 아래에 설명될 알고리즘들의 바람직한 특성을 많이 모방하는 발전된 최적화 알고리즘을 설계하는 방법에 대한 유용한 직관을 제공합니다.


### 뉴턴의 방법(Newton's Method)

어떤 함수 $f: \mathbb{R}^d \rightarrow \mathbb{R}$의 테일러 전개를 살펴볼 때, 첫 번째 항 이후에 멈출 필요는 없습니다. 사실 다음과 같이 쓸 수 있습니다.

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

번거로운 표기를 피하기 위해 $f$의 헤시안이라고 하는 $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$를 정의하는데, 이는 $d \times d$ 행렬입니다. 작은 $d$와 단순한 문제에 대해서는 $\mathbf{H}$를 계산하기 쉽습니다. 반면 딥 뉴럴 네트워크의 경우, $\mathcal{O}(d^2)$ 항목을 저장하는 비용 때문에 $\mathbf{H}$가 지나치게 클 수 있습니다. 더욱이 역전파를 통해 계산하기에 너무 비쌀 수 있습니다. 지금은 그러한 고려 사항을 무시하고 어떤 알고리즘을 얻게 되는지 살펴봅시다.

결국, $f$의 최솟값은 $\nabla f = 0$를 만족합니다.
:numref:`subsec_calculus-grad`의 미적분 규칙을 따라,
:eqref:`gd-hot-taylor`을 $\boldsymbol{\epsilon}$에 대해 미분하고 고차 항을 무시하면 다음에 도달합니다.

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

즉, 저희는 최적화 문제의 일부로 헤시안 $\mathbf{H}$를 역행렬화해야 합니다.

간단한 예로, $f(x) = \frac{1}{2} x^2$에 대해 $\nabla f(x) = x$이고 $\mathbf{H} = 1$입니다. 따라서 임의의 $x$에 대해 $\epsilon = -x$를 얻습니다. 다시 말해, 어떤 조정도 필요 없이 *단 한 번*의 단계로 완벽하게 수렴할 수 있습니다! 안타깝게도, 저희가 여기서 약간 운이 좋았습니다. $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$이므로 테일러 전개가 정확했습니다.

다른 문제에서는 어떻게 되는지 봅시다.
어떤 상수 $c$에 대한 볼록 쌍곡 코사인 함수 $f(x) = \cosh(cx)$가 주어졌을 때,
$x=0$에서의 전역 최솟값에
몇 번의 반복 후에 도달함을 볼 수 있습니다.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

이제 어떤 상수 $c$에 대한 $f(x) = x \cos(c x)$와 같은 *비볼록* 함수를 생각해 봅시다. 결국, 뉴턴의 방법에서는 헤시안으로 나누게 됩니다. 이는 2차 도함수가 *음수*이면 $f$의 값을 *증가시키는* 방향으로 걸어갈 수 있음을 의미합니다.
이는 알고리즘의 치명적인 결함입니다.
실제로 어떤 일이 일어나는지 봅시다.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

이는 극적으로 잘못되었습니다. 어떻게 고칠 수 있을까요? 한 가지 방법은 헤시안의 절댓값을 취해 "고치는" 것입니다. 또 다른 전략은 학습률을 다시 도입하는 것입니다. 이는 목적을 무산시키는 것처럼 보이지만, 꼭 그렇지는 않습니다. 2차 정보를 가지면 곡률이 클 때마다 신중해질 수 있고 목적 함수가 더 평평할 때마다 더 큰 걸음을 내디딜 수 있습니다.
약간 더 작은 학습률, 예를 들어 $\eta = 0.5$에서 이것이 어떻게 동작하는지 봅시다. 보시다시피, 꽤 효율적인 알고리즘을 얻습니다.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 수렴 분석

저희는 2차 도함수가 0이 아닌, 즉 $f'' > 0$인 어떤 볼록이고 세 번 미분 가능한 목적 함수 $f$에 대해 뉴턴 방법의 수렴 속도만 분석합니다. 다변량 증명은 아래 1차원 논증의 직접적인 확장이며, 직관 측면에서 크게 도움이 되지 않으므로 생략합니다.

$x^{(k)}$를 $k$번째 반복에서의 $x$의 값으로 표기하고 $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$를 $k$번째 반복에서의 최적성으로부터의 거리라 합시다. 테일러 전개에 의해, 조건 $f'(x^*) = 0$은 다음과 같이 쓸 수 있습니다.

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

이는 어떤 $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$에 대해 성립합니다. 위의 전개를 $f''(x^{(k)})$로 나누면 다음을 얻습니다.

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

저희가 업데이트 $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$를 가지고 있음을 기억하세요.
이 업데이트 방정식을 대입하고 양변에 절댓값을 취하면, 저희는

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

를 얻습니다. 따라서, 저희가 한정된 $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$의 영역에 있을 때마다, 저희는 이차적으로 감소하는 오차를 갖습니다.

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$


여담으로, 최적화 연구자들은 이것을 *선형(linear)* 수렴이라고 부르며, $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$와 같은 조건은 *상수(constant)* 수렴 속도라고 부릅니다.
이 분석에는 몇 가지 주의 사항이 따른다는 점에 유의하세요.
첫째, 저희는 빠른 수렴 영역에 언제 도달할지에 대해서는 사실 많은 보장을 가지지 못합니다. 대신, 일단 도달하면 수렴이 매우 빠를 것이라는 점만 알 수 있습니다. 둘째, 이 분석은 $f$가 고차 도함수까지 잘 동작해야 함을 요구합니다. 이는 $f$가 값을 변경하는 방식에 있어 "놀라운" 속성을 가지지 않도록 보장하는 것에 해당합니다.



### 사전 조건화(Preconditioning)

당연하게도 전체 헤시안을 계산하고 저장하는 것은 매우 비쌉니다. 따라서 대안을 찾는 것이 바람직합니다. 한 가지 개선 방법은 *사전 조건화*입니다. 헤시안을 전체적으로 계산하는 것을 피하고 *대각* 항목만 계산합니다. 이는 다음 형태의 업데이트 알고리즘으로 이어집니다.

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


이것이 완전한 뉴턴 방법만큼 좋지는 않지만, 사용하지 않는 것보다는 훨씬 낫습니다.
이것이 왜 좋은 아이디어일 수 있는지 보려면, 한 변수가 밀리미터 단위 높이를 나타내고 다른 변수가 킬로미터 단위 높이를 나타내는 상황을 생각해 보세요. 두 경우 모두 자연스러운 척도가 미터라고 가정하면, 매개변수화에 끔찍한 불일치가 있습니다. 다행히, 사전 조건화를 사용하면 이것이 제거됩니다. 사실상 경사 하강법에 사전 조건화를 적용하는 것은 각 변수(벡터 $\mathbf{x}$의 좌표)에 대해 다른 학습률을 선택하는 것에 해당합니다.
나중에 보겠지만, 사전 조건화는 확률적 경사 하강법 최적화 알고리즘의 일부 혁신을 이끕니다.


### 선 탐색이 포함된 경사 하강법

경사 하강법의 주요 문제 중 하나는 목표를 지나치거나 불충분한 진전을 이룰 수 있다는 점입니다. 이 문제에 대한 간단한 해결책은 경사 하강법과 함께 선 탐색을 사용하는 것입니다. 즉, $\nabla f(\mathbf{x})$로 주어진 방향을 사용한 다음 어떤 학습률 $\eta$가 $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$를 최소화하는지에 대해 이진 탐색을 수행합니다.

이 알고리즘은 빠르게 수렴합니다(분석과 증명은 예를 들어 :citet:`Boyd.Vandenberghe.2004` 참조). 그러나 딥러닝의 목적상 이는 그다지 실현 가능하지 않은데, 선 탐색의 각 단계에서 전체 데이터셋에 대해 목적 함수를 평가해야 하기 때문입니다. 이는 수행하기에 너무 비쌉니다.

## 요약

* 학습률이 중요합니다. 너무 크면 발산하고, 너무 작으면 진전이 없습니다.
* 경사 하강법은 지역 최솟값에 갇힐 수 있습니다.
* 고차원에서는 학습률을 조정하는 것이 복잡합니다.
* 사전 조건화는 스케일 조정에 도움이 될 수 있습니다.
* 뉴턴의 방법은 볼록 문제에서 적절히 작동하기 시작하면 훨씬 빠릅니다.
* 비볼록 문제에서는 조정 없이 뉴턴의 방법을 사용하는 것에 주의하세요.

## 연습문제

1. 경사 하강법에 대해 다양한 학습률과 목적 함수를 실험해 보세요.
1. 구간 $[a, b]$에서 볼록 함수를 최소화하기 위해 선 탐색을 구현해 보세요.
    1. 이진 탐색을 위해, 즉 $[a, (a+b)/2]$와 $[(a+b)/2, b]$ 중 어느 것을 선택할지 결정하기 위해 도함수가 필요한가요?
    1. 알고리즘의 수렴 속도는 얼마나 빠른가요?
    1. 알고리즘을 구현하고 $\log (\exp(x) + \exp(-2x -3))$을 최소화하는 데 적용해 보세요.
1. $\mathbb{R}^2$ 위에 정의된 목적 함수 중 경사 하강법이 극도로 느린 것을 설계하세요. 힌트: 서로 다른 좌표를 다르게 스케일링하세요.
1. 사전 조건화를 사용해 뉴턴 방법의 가벼운 버전을 구현하세요.
    1. 대각 헤시안을 사전 조건자로 사용하세요.
    1. 실제(부호가 있을 수 있는) 값 대신 그것의 절댓값을 사용하세요.
    1. 이를 위 문제에 적용하세요.
1. 위 알고리즘을 여러 목적 함수(볼록이든 아니든)에 적용해 보세요. 좌표를 $45$도 회전시키면 어떻게 되나요?

[Discussions](https://discuss.d2l.ai/t/351)
