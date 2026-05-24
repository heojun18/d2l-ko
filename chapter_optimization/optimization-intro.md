# 최적화와 딥러닝
:label:`sec_optimization-intro`

이 절에서는 최적화와 딥러닝 사이의 관계, 그리고 딥러닝에서 최적화를 사용할 때의 어려움에 대해 논의합니다.
딥러닝 문제의 경우, 일반적으로 먼저 *손실 함수*를 정의합니다. 손실 함수가 정해지면, 손실을 최소화하기 위해 최적화 알고리즘을 사용할 수 있습니다.
최적화에서 손실 함수는 흔히 최적화 문제의 *목적 함수*라고 불립니다. 전통과 관례에 따라 대부분의 최적화 알고리즘은 *최소화*에 관심을 둡니다. 목적 함수를 최대화해야 한다면 간단한 해결책이 있습니다. 목적 함수의 부호를 뒤집기만 하면 됩니다.

## 최적화의 목표

비록 최적화가 딥러닝의 손실 함수를 최소화하는 방법을 제공하지만, 본질적으로 최적화와 딥러닝의 목표는
근본적으로 다릅니다.
전자는 주로 목적 함수를 최소화하는 데 관심이 있는 반면,
후자는 유한한 양의 데이터가 주어졌을 때 적절한 모델을 찾는 데 관심이 있습니다.
:numref:`sec_generalization_basics`에서
저희는 이 두 가지 목표의 차이를 자세히 논의했습니다.
예를 들어,
학습 오차와 일반화 오차는 일반적으로 다릅니다. 최적화 알고리즘의 목적 함수는 보통 학습 데이터셋에 기반한
손실 함수이므로, 최적화의 목표는 학습 오차를 줄이는 것입니다.
그러나 딥러닝(혹은 더 넓게는 통계적 추론)의 목표는
일반화 오차를 줄이는 것입니다.
후자를 달성하려면 학습 오차를 줄이기 위해 최적화 알고리즘을 사용하는 것 외에도
과적합에 주의를 기울일 필요가 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

앞서 말한 서로 다른 목표를 설명하기 위해
경험적 위험(empirical risk)과 위험(risk)을
고려해 봅시다.
:numref:`subsec_empirical-risk-and-risk`에서 설명한 바와 같이,
경험적 위험은
학습 데이터셋에 대한 평균 손실인 반면,
위험은 전체 데이터 모집단에 대한
기댓값 손실입니다.
아래에서는 두 함수를 정의합니다.
위험 함수 `f`와
경험적 위험 함수 `g`입니다.
저희에게 유한한 양의 학습 데이터만 있다고 가정해 봅시다.
그 결과, 여기서 `g`는 `f`보다 덜 매끄럽습니다.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

아래 그래프는 학습 데이터셋에서의 경험적 위험의 최솟값이 위험(일반화 오차)의 최솟값과 다른 위치에 있을 수 있음을 보여줍니다.

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## 딥러닝에서의 최적화 과제

이 장에서는 모델의 일반화 오차가 아니라 목적 함수를 최소화하는 데 있어 최적화 알고리즘의 성능에 구체적으로 초점을 맞출 것입니다.
:numref:`sec_linear_regression`에서
저희는 최적화 문제에서 해석적 해와 수치적 해를 구분했습니다.
딥러닝에서 대부분의 목적 함수는
복잡하고 해석적 해가 존재하지 않습니다. 대신 수치적
최적화 알고리즘을 사용해야 합니다.
이 장의 최적화 알고리즘들은
모두 이 범주에 속합니다.

딥러닝 최적화에는 많은 어려움이 있습니다. 그중에서도 가장 골치 아픈 것들은 지역 최솟값(local minima), 안장점(saddle point), 그리고 사라지는 경사도(vanishing gradient)입니다.
이들을 살펴봅시다.


### 지역 최솟값(Local Minima)

임의의 목적 함수 $f(x)$에 대해,
$x$에서 $f(x)$의 값이 $x$ 주변의 다른 어떤 점들에서의 $f(x)$ 값보다 작다면, $f(x)$는 지역 최솟값일 수 있습니다.
$x$에서의 $f(x)$ 값이 전체 정의역에 걸쳐 목적 함수의 최솟값이라면,
$f(x)$는 전역 최솟값(global minimum)입니다.

예를 들어, 주어진 함수가

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

이때 이 함수의 지역 최솟값과 전역 최솟값을 근사할 수 있습니다.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

딥러닝 모델의 목적 함수는 보통 많은 지역 최적점을 가집니다.
최적화 문제의 수치적 해가 지역 최적점 근처에 있을 때, 목적 함수의 해의 경사도가 0에 가까워지거나 0이 되면서, 최종 반복으로 얻은 수치적 해는 목적 함수를 *전역적*으로가 아니라 *지역적*으로만 최소화할 수 있습니다.
어느 정도의 노이즈만이 파라미터를 지역 최솟값에서 벗어나게 할 수 있습니다. 실제로 이는
미니배치 확률적 경사 하강법의 유익한 특성 중 하나로, 미니배치에 걸친 경사도의 자연스러운 변동이 파라미터를 지역 최솟값에서 벗어나게 할 수 있습니다.


### 안장점(Saddle Points)

지역 최솟값 외에도, 안장점은 경사도가 사라지는 또 다른 이유입니다. *안장점*은 함수의 모든 경사도가 사라지지만 전역 최솟값도 지역 최솟값도 아닌 임의의 위치를 말합니다.
함수 $f(x) = x^3$을 생각해 봅시다. 그것의 1차 및 2차 도함수는 $x=0$에서 사라집니다. 최솟값이 아님에도 불구하고 이 지점에서 최적화가 멈출 수 있습니다.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

고차원에서의 안장점은 아래 예가 보여주듯 훨씬 더 음흉합니다. 함수 $f(x, y) = x^2 - y^2$을 생각해 봅시다. 이 함수는 $(0, 0)$에 안장점을 가집니다. 이 점은 $y$에 대해서는 최댓값이고 $x$에 대해서는 최솟값입니다. 더욱이, 이 점은 *안장*처럼 보이는데, 이 수학적 특성의 이름은 여기에서 유래되었습니다.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

함수의 입력이 $k$차원 벡터이고 출력이 스칼라라고 가정하면, 그것의 헤시안 행렬은 $k$개의 고윳값을 가질 것입니다.
함수의 해는 함수 경사도가 0이 되는 지점에서
지역 최솟값, 지역 최댓값, 또는 안장점일 수 있습니다.

* 경사도가 0인 위치에서 함수의 헤시안 행렬의 고윳값이 모두 양수이면, 함수의 지역 최솟값을 얻습니다.
* 경사도가 0인 위치에서 함수의 헤시안 행렬의 고윳값이 모두 음수이면, 함수의 지역 최댓값을 얻습니다.
* 경사도가 0인 위치에서 함수의 헤시안 행렬의 고윳값이 음수와 양수가 모두 있는 경우, 함수의 안장점을 얻습니다.

고차원 문제에서는 적어도 *일부* 고윳값이 음수일 가능성이 매우 높습니다. 이로 인해 안장점이 지역 최솟값보다 더 가능성이 높아집니다. 다음 절에서 볼록성을 소개할 때 이 상황의 몇몇 예외를 다룰 것입니다. 요약하면, 볼록 함수는 헤시안의 고윳값이 결코 음수가 되지 않는 함수입니다. 안타깝게도 대부분의 딥러닝 문제는 이 범주에 속하지 않습니다. 그럼에도 불구하고 이는 최적화 알고리즘을 연구하는 데 훌륭한 도구입니다.

### 사라지는 경사도(Vanishing Gradients)

마주칠 수 있는 가장 음흉한 문제는 아마도 사라지는 경사도일 것입니다.
:numref:`subsec_activation-functions`에서 흔히 사용되는 활성화 함수와 그 도함수를 떠올려 봅시다.
예를 들어, 함수 $f(x) = \tanh(x)$를 최소화하고자 하는데 마침 $x = 4$에서 시작했다고 가정해 봅시다. 보시다시피, $f$의 경사도는 거의 0에 가깝습니다.
더 구체적으로, $f'(x) = 1 - \tanh^2(x)$이므로 $f'(4) = 0.0013$이 됩니다.
결과적으로, 진전이 있기 전까지 최적화는 오랫동안 멈춰 있게 될 것입니다. 이는 ReLU 활성화 함수가 도입되기 전까지 딥러닝 모델의 학습이 꽤 까다로웠던 이유 중 하나로 밝혀집니다.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

보신 바와 같이, 딥러닝을 위한 최적화에는 많은 어려움이 있습니다. 다행히 잘 작동하고 초보자도 쉽게 사용할 수 있는 견고한 알고리즘들이 존재합니다. 더욱이, *가장* 좋은 해를 찾는 것이 꼭 필요한 것은 아닙니다. 지역 최적점이나 심지어 그것의 근사 해도 여전히 매우 유용합니다.

## 요약

* 학습 오차를 최소화하는 것이 일반화 오차를 최소화할 최적의 파라미터 집합을 찾는 것을 *보장하지는* 않습니다.
* 최적화 문제는 많은 지역 최솟값을 가질 수 있습니다.
* 일반적으로 문제는 볼록하지 않기 때문에 안장점이 훨씬 더 많을 수 있습니다.
* 사라지는 경사도는 최적화를 멈추게 할 수 있습니다. 종종 문제의 재매개변수화가 도움이 됩니다. 파라미터의 좋은 초기화 또한 유익할 수 있습니다.


## 연습문제

1. 은닉층에 $d$ 차원을 가지고 단일 출력을 가진 단순한 MLP를 생각해 봅시다. 임의의 지역 최솟값에 대해 동일하게 동작하는 적어도 $d!$ 개의 동등한 해가 존재함을 보이세요.
1. 항목 $M_{ij} = M_{ji}$가 각각 어떤 확률분포 $p_{ij}$에서 추출되는 대칭 랜덤 행렬 $\mathbf{M}$이 있다고 가정합시다. 또한 $p_{ij}(x) = p_{ij}(-x)$, 즉
   분포가 대칭이라고 가정합시다(자세한 내용은 예를 들어 :citet:`Wigner.1958` 참조).
    1. 고윳값에 대한 분포 또한 대칭임을 증명하세요. 즉, 임의의 고유벡터 $\mathbf{v}$에 대해 관련 고윳값 $\lambda$가 $P(\lambda > 0) = P(\lambda < 0)$를 만족할 확률.
    1. 위의 사실이 왜 $P(\lambda > 0) = 0.5$를 의미하지 *않는지* 설명하세요.
1. 딥러닝 최적화에 관련된 다른 어떤 어려움들을 생각해 볼 수 있을까요?
1. (실제) 공을 (실제) 안장 위에서 균형 잡고 싶다고 가정해 봅시다.
    1. 왜 이것이 어려울까요?
    1. 이 효과를 최적화 알고리즘에도 활용할 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
