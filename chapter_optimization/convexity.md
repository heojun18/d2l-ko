# 볼록성(Convexity)
:label:`sec_convexity`

볼록성은 최적화 알고리즘 설계에서 매우 중요한 역할을 합니다. 
이는 주로 그러한 맥락에서 알고리즘을 분석하고 테스트하기가 훨씬 더 쉽기 때문입니다. 
다시 말해,
알고리즘이 볼록 환경에서조차 성능이 좋지 않다면,
일반적으로 다른 곳에서 훌륭한 결과를 기대해서는 안 됩니다. 
더욱이, 딥러닝의 최적화 문제는 일반적으로 비볼록이지만, 지역 최솟값 근처에서는 종종 볼록 문제의 일부 특성을 보입니다. 이는 :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`와 같은 흥미로운 새로운 최적화 변종으로 이어질 수 있습니다.

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

## 정의

볼록 분석에 앞서,
저희는 *볼록 집합*과 *볼록 함수*를 정의해야 합니다.
이들은 머신러닝에 흔히 적용되는 수학적 도구로 이어집니다.


### 볼록 집합(Convex Sets)

집합은 볼록성의 기초입니다. 간단히 말해, 벡터 공간의 집합 $\mathcal{X}$는 임의의 $a, b \in \mathcal{X}$에 대해 $a$와 $b$를 잇는 선분도 $\mathcal{X}$ 안에 있다면 *볼록*입니다. 수학적 용어로 표현하면, 모든 $\lambda \in [0, 1]$에 대해 다음이 성립한다는 의미입니다.

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \textrm{ whenever } a, b \in \mathcal{X}.$$

이는 다소 추상적으로 들립니다. :numref:`fig_pacman`을 봅시다. 첫 번째 집합은 그 안에 포함되지 않는 선분이 존재하므로 볼록이 아닙니다.
다른 두 집합은 그런 문제가 없습니다.

![첫 번째 집합은 비볼록이고 나머지 두 집합은 볼록입니다.](../img/pacman.svg)
:label:`fig_pacman`

정의 자체는 그것으로 무엇인가를 할 수 있지 않다면 특별히 유용하지 않습니다.
이 경우 :numref:`fig_convex_intersect`에 나타난 것처럼 교집합을 살펴볼 수 있습니다.
$\mathcal{X}$와 $\mathcal{Y}$가 볼록 집합이라고 가정해 봅시다. 그러면 $\mathcal{X} \cap \mathcal{Y}$도 볼록입니다. 이를 확인하려면 임의의 $a, b \in \mathcal{X} \cap \mathcal{Y}$를 생각해 봅시다. $\mathcal{X}$와 $\mathcal{Y}$가 볼록이므로 $a$와 $b$를 잇는 선분은 $\mathcal{X}$와 $\mathcal{Y}$ 모두에 포함됩니다. 그렇다면 이는 또한 $\mathcal{X} \cap \mathcal{Y}$에도 포함되어야 하므로, 저희의 정리를 증명합니다.

![두 볼록 집합의 교집합은 볼록입니다.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

이 결과를 약간의 노력으로 강화할 수 있습니다. 볼록 집합 $\mathcal{X}_i$가 주어졌을 때, 그 교집합 $\cap_{i} \mathcal{X}_i$는 볼록입니다.
역이 성립하지 않음을 확인하려면, 두 개의 서로소인 집합 $\mathcal{X} \cap \mathcal{Y} = \emptyset$를 생각해 봅시다. 이제 $a \in \mathcal{X}$와 $b \in \mathcal{Y}$를 선택합니다. :numref:`fig_nonconvex`에서 $a$와 $b$를 잇는 선분은 저희가 $\mathcal{X} \cap \mathcal{Y} = \emptyset$라고 가정했으므로 $\mathcal{X}$에도 $\mathcal{Y}$에도 속하지 않는 부분을 포함해야 합니다. 따라서 그 선분은 $\mathcal{X} \cup \mathcal{Y}$에도 속하지 않으므로, 일반적으로 볼록 집합의 합집합은 볼록일 필요가 없음을 증명합니다.

![두 볼록 집합의 합집합은 볼록일 필요가 없습니다.](../img/nonconvex.svg)
:label:`fig_nonconvex`

일반적으로 딥러닝의 문제들은 볼록 집합 위에서 정의됩니다. 예를 들어, $\mathbb{R}^d$,
즉 실수의 $d$차원 벡터 집합은 볼록 집합입니다(결국, $\mathbb{R}^d$ 안의 임의의 두 점 사이의 선분은 $\mathbb{R}^d$ 안에 남아 있습니다). 어떤 경우에는 길이가 제한된 변수들로 작업합니다. 예를 들어 $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\| \leq r\}$로 정의된 반지름 $r$의 공과 같은 것입니다.

### 볼록 함수(Convex Functions)

이제 볼록 집합이 있으니 *볼록 함수* $f$를 도입할 수 있습니다.
볼록 집합 $\mathcal{X}$가 주어졌을 때, 함수 $f: \mathcal{X} \to \mathbb{R}$가 모든 $x, x' \in \mathcal{X}$와 모든 $\lambda \in [0, 1]$에 대해 다음을 만족하면 *볼록*입니다.

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

이를 설명하기 위해 몇 가지 함수를 그려보고 어느 것이 이 요구 조건을 만족하는지 확인해 봅시다.
아래에서는 볼록 함수와 비볼록 함수를 몇 개 정의합니다.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

예상한 대로, 코사인 함수는 *비볼록*이지만, 포물선과 지수 함수는 볼록입니다. 조건이 의미를 가지려면 $\mathcal{X}$가 볼록 집합이어야 한다는 요구는 필수임에 유의하세요. 그렇지 않으면 $f(\lambda x + (1-\lambda) x')$의 결과가 잘 정의되지 않을 수 있습니다.


### 옌센 부등식(Jensen's Inequality)

볼록 함수 $f$가 주어졌을 때,
가장 유용한 수학적 도구 중 하나는
*옌센 부등식*입니다.
이는 볼록성 정의의 일반화에 해당합니다.

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \textrm{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

여기서 $\alpha_i$는 $\sum_i \alpha_i = 1$를 만족하는 비음수 실수들이고 $X$는 확률 변수입니다.
다시 말해, 볼록 함수의 기댓값은 기댓값의 볼록 함수보다 작지 않으며, 후자는 보통 더 단순한 표현입니다. 
첫 번째 부등식을 증명하기 위해서는 합의 한 항씩에 볼록성의 정의를 반복적으로 적용합니다.


옌센 부등식의 일반적인 응용 중 하나는
더 단순한 표현으로 더 복잡한 표현을 한정하는 것입니다.
예를 들어,
부분 관측된 확률 변수들의 로그 가능도와 관련하여
적용될 수 있습니다. 즉, 저희는

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

을 사용하는데, 이는 $\int P(Y) P(X \mid Y) dY = P(X)$이기 때문입니다.
이것은 변분법에서 사용될 수 있습니다. 여기서 $Y$는 일반적으로 관측되지 않은 확률 변수이고, $P(Y)$는 그것이 어떻게 분포할 것인지에 대한 최선의 추측이며, $P(X)$는 $Y$가 적분되어 사라진 분포입니다. 예를 들어, 클러스터링에서 $Y$는 클러스터 레이블일 수 있고 $P(X \mid Y)$는 클러스터 레이블을 적용할 때의 생성 모델입니다.



## 속성

볼록 함수는 많은 유용한 속성을 가집니다. 아래에서 일반적으로 사용되는 몇 가지를 설명합니다.


### 지역 최솟값이 곧 전역 최솟값

무엇보다도 먼저, 볼록 함수의 지역 최솟값은 전역 최솟값이기도 합니다. 
저희는 다음과 같이 모순에 의해 그것을 증명할 수 있습니다.

볼록 집합 $\mathcal{X}$ 위에 정의된 볼록 함수 $f$를 생각해 봅시다.
$x^{\ast} \in \mathcal{X}$가 지역 최솟값이라고 가정해 봅시다.
즉, $0 < |x - x^{\ast}| \leq p$를 만족하는 $x \in \mathcal{X}$에 대해 $f(x^{\ast}) < f(x)$가 되는 작은 양수 $p$가 존재합니다.

지역 최솟값 $x^{\ast}$가
$f$의 전역 최솟값이 아니라고 가정해 봅시다.
$f(x') < f(x^{\ast})$인 $x' \in \mathcal{X}$가 존재합니다. 
또한
$\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$와 같은
$\lambda \in [0, 1)$도 존재하므로,
$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$가 됩니다. 

그러나,
볼록 함수의 정의에 따르면 저희는

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

를 얻는데, 이는 $x^{\ast}$가 지역 최솟값이라는 저희의 진술과 모순됩니다.
따라서 $f(x') < f(x^{\ast})$인 $x' \in \mathcal{X}$는 존재하지 않습니다. 지역 최솟값 $x^{\ast}$는 또한 전역 최솟값입니다.

예를 들어, 볼록 함수 $f(x) = (x-1)^2$는 $x=1$에서 지역 최솟값을 가지며, 이는 전역 최솟값이기도 합니다.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

볼록 함수의 지역 최솟값이 전역 최솟값이기도 하다는 사실은 매우 편리합니다. 
이는 함수를 최소화할 때 "갇히지" 않는다는 뜻입니다. 
다만, 이것이 전역 최솟값이 하나 이상 존재할 수 없다거나 심지어 하나라도 존재할 수도 있음을 의미하는 것은 아닙니다. 예를 들어, 함수 $f(x) = \mathrm{max}(|x|-1, 0)$는 구간 $[-1, 1]$에 걸쳐 최솟값을 갖습니다. 반대로, 함수 $f(x) = \exp(x)$는 $\mathbb{R}$에서 최솟값을 갖지 않습니다. $x \to -\infty$일 때 $0$에 점근하지만, $f(x) = 0$이 되는 $x$는 없습니다.

### 볼록 함수의 하위 집합은 볼록

저희는 볼록 함수의 *하위 집합*을 통해
편리하게 볼록 집합을 정의할 수 있습니다.
구체적으로,
볼록 집합 $\mathcal{X}$ 위에 정의된 볼록 함수 $f$가 주어졌을 때,
임의의 하위 집합

$$\mathcal{S}_b \stackrel{\textrm{def}}{=} \{x | x \in \mathcal{X} \textrm{ and } f(x) \leq b\}$$

는 볼록입니다. 

이를 빠르게 증명해 봅시다. 임의의 $x, x' \in \mathcal{S}_b$에 대해, $\lambda \in [0, 1]$인 한 $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$임을 보여야 한다는 것을 기억하세요. 
$f(x) \leq b$이고 $f(x') \leq b$이므로,
볼록성의 정의에 의해 저희는 

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

를 얻습니다.

### 볼록성과 2차 도함수

함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 2차 도함수가 존재한다면, $f$가 볼록인지 확인하는 것은 매우 쉽습니다. 
저희가 해야 할 일은 $f$의 헤시안이 양의 준정부호인지 확인하는 것뿐입니다. 즉, $\nabla^2f \succeq 0$, 다시 말해,
헤시안 행렬 $\nabla^2f$를 $\mathbf{H}$로 표기하면,
모든 $\mathbf{x} \in \mathbb{R}^n$에 대해
$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$입니다.
예를 들어, 함수 $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$는 $\nabla^2 f = \mathbf{1}$, 즉 그 헤시안이 단위 행렬이므로 볼록입니다.


공식적으로, 두 번 미분 가능한 1차원 함수 $f: \mathbb{R} \rightarrow \mathbb{R}$가 볼록일 필요충분조건은
2차 도함수 $f'' \geq 0$입니다. 임의의 두 번 미분 가능한 다차원 함수 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$에 대해,
볼록일 필요충분조건은 헤시안 $\nabla^2f \succeq 0$입니다.

먼저, 1차원의 경우를 증명해야 합니다.
$f$의 볼록성이 $f'' \geq 0$를 함의함을 보기 위해 저희는 다음 사실을 사용합니다.

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

2차 도함수가 유한차의 극한으로 주어지므로 다음이 성립합니다.

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

$f'' \geq 0$이 $f$가 볼록임을 함의함을 보기 위해, 저희는 $f'' \geq 0$이 $f'$가 단조 비감소 함수임을 함의한다는 사실을 사용합니다. $a < x < b$를 $\mathbb{R}$의 세 점이라고 하고,
$x = (1-\lambda)a + \lambda b$이고 $\lambda \in (0, 1)$이라고 하겠습니다.
평균값 정리에 따르면,
다음과 같은
$\alpha \in [a, x]$와 $\beta \in [x, b]$가 존재합니다.

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \textrm{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$


단조성에 의해 $f'(\beta) \geq f'(\alpha)$이므로,

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

$x = (1-\lambda)a + \lambda b$이므로,
저희는

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

를 얻고, 이로써 볼록성을 증명합니다.

둘째, 다차원의 경우를 증명하기 전에 보조정리가 필요합니다.
$f: \mathbb{R}^n \rightarrow \mathbb{R}$가 볼록일 필요충분조건은 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해

$$g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \textrm{ where } z \in [0,1]$$ 

가 볼록인 것입니다.

$f$의 볼록성이 $g$가 볼록임을 함의함을 증명하기 위해,
저희는 모든 $a, b, \lambda \in [0, 1]$에 대해(따라서
$0 \leq \lambda a + (1-\lambda) b \leq 1$)

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

를 보일 수 있습니다.

역을 증명하기 위해,
저희는 모든 $\lambda \in [0, 1]$에 대해

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\end{aligned}$$

를 보일 수 있습니다.


마지막으로,
위의 보조정리와 1차원 경우의 결과를 사용하여,
다차원 경우를 다음과 같이 증명할 수 있습니다.
다차원 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$가 볼록일 필요충분조건은 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해 $g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$ ($z \in [0,1]$)가
볼록인 것입니다.
1차원의 경우에 따르면,
이는 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$에 대해
$g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2f$)일 필요충분조건이며,
이는 양의 준정부호 행렬의 정의에 의해
$\mathbf{H} \succeq 0$과 동치입니다.


## 제약 조건

볼록 최적화의 좋은 성질 중 하나는 제약 조건을 효율적으로 처리할 수 있다는 것입니다. 즉, 다음 형태의 *제약 최적화* 문제를 풀 수 있게 해줍니다.

$$\begin{aligned} \mathop{\textrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \textrm{ subject to } & c_i(\mathbf{x}) \leq 0 \textrm{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

여기서 $f$는 목적이고 함수 $c_i$는 제약 함수입니다. 이것이 무엇을 하는지 보기 위해, $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$인 경우를 생각해 봅시다. 이 경우 파라미터 $\mathbf{x}$는 단위 공 안으로 제약됩니다. 두 번째 제약이 $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$라면, 이는 모든 $\mathbf{x}$가 반공간 위에 놓이는 것에 해당합니다. 두 제약을 동시에 만족하는 것은 공의 단면을 선택하는 것에 해당합니다.

### 라그랑지안(Lagrangian)

일반적으로, 제약 최적화 문제를 푸는 것은 어렵습니다. 한 가지 접근 방법은 다소 단순한 직관을 가진 물리학에서 비롯됩니다. 상자 안에 공이 있다고 상상해 봅시다. 공은 가장 낮은 곳으로 굴러갈 것이고 중력은 상자의 측면이 공에 가할 수 있는 힘과 균형을 이룰 것입니다. 요컨대, 목적 함수의 경사도(즉, 중력)는 제약 함수의 경사도(공은 벽이 "되밀어내는" 힘에 의해 상자 안에 머물러야 한다)에 의해 상쇄될 것입니다. 
일부 제약은 활성화되지 않을 수 있음에 유의하세요.
공이 닿지 않는 벽은
공에 어떤 힘도 가할 수 없을 것입니다.


*라그랑지안* $L$의 유도를 건너뛰면,
위의 추론은
다음 안장점 최적화 문제로 표현될 수 있습니다.

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \textrm{ where } \alpha_i \geq 0.$$

여기서 변수 $\alpha_i$ ($i=1,\ldots,n$)는 제약이 적절하게 시행되도록 보장하는 소위 *라그랑주 승수*입니다. 모든 $i$에 대해 $c_i(\mathbf{x}) \leq 0$를 보장할 만큼만 충분히 크게 선택됩니다. 예를 들어, 자연스럽게 $c_i(\mathbf{x}) < 0$인 임의의 $\mathbf{x}$에 대해, 저희는 결국 $\alpha_i = 0$를 선택하게 될 것입니다. 더욱이, 이는 모든 $\alpha_i$에 대해 $L$을 *최대화*하고 동시에 $\mathbf{x}$에 대해 *최소화*하고자 하는 안장점 최적화 문제입니다. 함수 $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$에 도달하는 방법을 설명하는 풍부한 문헌이 존재합니다. 저희의 목적에는 $L$의 안장점이 원래의 제약 최적화 문제가 최적으로 풀린 곳임을 아는 것으로 충분합니다.

### 페널티

제약 최적화 문제를 적어도 *근사적*으로 만족시키는 한 가지 방법은 라그랑지안 $L$을 적용하는 것입니다. 
$c_i(\mathbf{x}) \leq 0$를 만족시키는 대신, 저희는 단순히 목적 함수 $f(x)$에 $\alpha_i c_i(\mathbf{x})$를 더합니다. 이는 제약이 너무 심하게 위반되지 않음을 보장합니다.

사실, 저희는 이 기법을 줄곧 사용해 왔습니다. :numref:`sec_weight_decay`의 가중치 감쇠를 생각해 봅시다. 거기서 저희는 $\mathbf{w}$가 너무 크게 자라지 않도록 목적 함수에 $\frac{\lambda}{2} \|\mathbf{w}\|^2$를 더했습니다. 제약 최적화 관점에서 보면, 이는 어떤 반지름 $r$에 대해 $\|\mathbf{w}\|^2 - r^2 \leq 0$를 보장할 것임을 알 수 있습니다. $\lambda$의 값을 조정하면 $\mathbf{w}$의 크기를 변화시킬 수 있습니다.

일반적으로, 페널티를 더하는 것은 근사적 제약 만족을 보장하는 좋은 방법입니다. 실제로 이는 정확한 만족보다 훨씬 더 견고한 것으로 밝혀졌습니다. 더욱이, 비볼록 문제의 경우 볼록 경우에서 정확한 접근법을 그토록 매력적으로 만드는 많은 속성들(예: 최적성)이 더 이상 성립하지 않습니다.

### 사영(Projections)

제약을 만족시키는 또 다른 전략은 사영입니다. 다시 한번, 저희는 이전에 그것들을 만난 적이 있습니다. 예를 들어, :numref:`sec_rnn-scratch`의 경사도 클리핑을 다룰 때 그러했습니다. 거기서 저희는 경사도의 길이가 다음에 의해 $\theta$로 한정되도록 보장했습니다.

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

이는 반지름 $\theta$의 공 위로의 $\mathbf{g}$의 *사영*인 것으로 밝혀집니다. 더 일반적으로, 볼록 집합 $\mathcal{X}$ 위로의 사영은 다음과 같이 정의됩니다.

$$\textrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

이는 $\mathcal{X}$에서 $\mathbf{x}$에 가장 가까운 점입니다. 

![볼록 사영.](../img/projections.svg)
:label:`fig_projections`

사영의 수학적 정의는 다소 추상적으로 들릴 수 있습니다. :numref:`fig_projections`는 그것을 좀 더 명확하게 설명합니다. 거기에는 두 개의 볼록 집합, 원과 마름모가 있습니다. 
두 집합 안의 점들(노란색)은 사영 동안 변하지 않습니다. 
두 집합 바깥의 점들(검은색)은 원래 점(검은색)에 가장 가까운 집합 안의 점들(빨간색)로 사영됩니다.
$\ell_2$ 공의 경우 이것은 방향을 변하지 않게 하지만, 마름모의 경우에서 볼 수 있듯 일반적으로 그럴 필요는 없습니다.


볼록 사영의 용도 중 하나는 희소 가중치 벡터를 계산하는 것입니다. 이 경우 저희는 가중치 벡터를 $\ell_1$ 공 위로 사영하는데,
이는 :numref:`fig_projections`의 마름모 경우의 일반화된 버전입니다.


## 요약

딥러닝의 맥락에서 볼록 함수의 주된 목적은 최적화 알고리즘에 동기를 부여하고 그것들을 자세히 이해하는 데 도움을 주는 것입니다. 다음에서는 경사 하강법과 확률적 경사 하강법이 어떻게 유도되는지 살펴볼 것입니다.


* 볼록 집합의 교집합은 볼록입니다. 합집합은 그렇지 않습니다.
* 볼록 함수의 기댓값은 기댓값의 볼록 함수보다 작지 않습니다(옌센 부등식).
* 두 번 미분 가능한 함수가 볼록일 필요충분조건은 그것의 헤시안(2차 도함수의 행렬)이 양의 준정부호인 것입니다.
* 볼록 제약은 라그랑지안을 통해 추가될 수 있습니다. 실제로는 단순히 페널티와 함께 목적 함수에 더할 수 있습니다.
* 사영은 원래 점에 가장 가까운 볼록 집합 안의 점들로 매핑됩니다.

## 연습문제

1. 집합 안의 점들 사이의 모든 선분을 그리고 그 선분이 포함되는지 확인하여 집합의 볼록성을 검증한다고 가정해 봅시다.
    1. 경계상의 점들만 확인해도 충분함을 증명하세요.
    1. 집합의 꼭짓점만 확인해도 충분함을 증명하세요.
1. $p$-노름을 사용한 반지름 $r$의 공을 $\mathcal{B}_p[r] \stackrel{\textrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\|_p \leq r\}$로 표시합시다. 모든 $p \geq 1$에 대해 $\mathcal{B}_p[r]$이 볼록임을 증명하세요.
1. 볼록 함수 $f$와 $g$가 주어졌을 때, $\mathrm{max}(f, g)$ 또한 볼록임을 보이세요. $\mathrm{min}(f, g)$가 볼록이 아님을 증명하세요.
1. 소프트맥스 함수의 정규화가 볼록임을 증명하세요. 더 구체적으로 다음의 볼록성을 증명하세요.
    $f(x) = \log \sum_i \exp(x_i)$.
1. 선형 부분공간, 즉 $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$가 볼록 집합임을 증명하세요.
1. $\mathbf{b} = \mathbf{0}$인 선형 부분공간의 경우 사영 $\textrm{Proj}_\mathcal{X}$이 어떤 행렬 $\mathbf{M}$에 대해 $\mathbf{M} \mathbf{x}$로 쓰일 수 있음을 증명하세요.
1. 두 번 미분 가능한 볼록 함수 $f$에 대해 어떤 $\xi \in [0, \epsilon]$에 대해 $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$로 쓸 수 있음을 보이세요.
1. 볼록 집합 $\mathcal{X}$와 두 벡터 $\mathbf{x}$, $\mathbf{y}$가 주어졌을 때, 사영은 결코 거리를 증가시키지 않음을 증명하세요. 즉, $\|\mathbf{x} - \mathbf{y}\| \geq \|\textrm{Proj}_\mathcal{X}(\mathbf{x}) - \textrm{Proj}_\mathcal{X}(\mathbf{y})\|$.


[Discussions](https://discuss.d2l.ai/t/350)
