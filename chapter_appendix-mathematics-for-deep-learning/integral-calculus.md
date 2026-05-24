# 적분 적분학
:label:`sec_integral_calculus`

미분은 전통적인 미적분 교육 내용의 절반만 차지합니다. 다른 기둥인 적분은 처음에는 다소 동떨어진 질문, 즉 "이 곡선 아래의 면적은 무엇인가?"로 시작하는 것처럼 보입니다. 겉보기에는 관련이 없어 보이지만, 적분은 *미적분의 기본 정리*로 알려진 것을 통해 미분과 단단히 얽혀 있습니다.

이 책에서 논의하는 머신러닝의 수준에서는, 저희는 적분에 대한 깊은 이해가 필요하지 않을 것입니다. 그러나, 나중에 마주칠 추가 응용을 위한 기초를 놓기 위해 간단한 소개를 제공할 것입니다.

## 기하학적 해석
함수 $f(x)$가 있다고 가정해 보십시오. 단순함을 위해, $f(x)$가 음이 아니라고(0보다 작은 값을 결코 취하지 않는다고) 가정해 봅시다. 저희가 이해하려고 하는 것은: $f(x)$와 $x$축 사이에 포함된 면적은 무엇인가?

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

대부분의 경우, 이 면적은 무한하거나 정의되지 않을 것이므로($f(x) = x^{2}$ 아래의 면적을 고려해 보십시오), 사람들은 종종 한 쌍의 끝, 예를 들어 $a$와 $b$ 사이의 면적에 대해 이야기할 것입니다.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

저희는 아래의 적분 기호로 이 면적을 표시할 것입니다.

$$
\textrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

내부 변수는 $\sum$의 합의 인덱스와 매우 비슷하게 더미 변수이므로, 이는 저희가 원하는 어떤 내부 값으로도 동등하게 작성될 수 있습니다.

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

이러한 적분을 어떻게 근사할 수 있는지 이해하려고 시도하는 전통적인 방법이 있습니다. 저희는 $a$와 $b$ 사이의 영역을 취하여 $N$개의 수직 슬라이스로 자르는 것을 상상할 수 있습니다. 만약 $N$이 크다면, 각 슬라이스의 면적을 직사각형으로 근사할 수 있고, 그런 다음 면적을 합하여 곡선 아래의 총 면적을 얻을 수 있습니다. 코드에서 이를 하는 예제를 살펴봅시다. 나중 절에서 참값을 얻는 방법을 볼 것입니다.

```{.python .input}
#@tab mxnet
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

문제는 수치적으로 할 수 있지만, 다음과 같이 가장 단순한 함수에 대해서만 이 접근법을 해석적으로 할 수 있다는 것입니다.

$$
\int_a^b x \;dx.
$$

위 코드의 예와 같이 다소 더 복잡한 어떤 것

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$

은 그러한 직접적인 방법으로 풀 수 있는 것을 넘어섭니다.

저희는 대신 다른 접근법을 취할 것입니다. 면적의 개념으로 직관적으로 작업하고, 적분을 찾는 데 사용되는 주요 계산 도구인 *미적분의 기본 정리*를 배울 것입니다. 이는 적분에 대한 저희 연구의 기초가 될 것입니다.

## 미적분의 기본 정리

적분 이론에 더 깊이 다이빙하기 위해, 함수를 소개합시다.

$$
F(x) = \int_0^x f(y) dy.
$$

이 함수는 $x$를 어떻게 변경하는지에 따라 $0$과 $x$ 사이의 면적을 측정합니다. 이것이 저희에게 필요한 모든 것임에 유의하십시오. 왜냐하면

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

이는 :numref:`fig_area-subtract`에 표시된 것처럼 먼 끝점까지의 면적을 측정한 다음 가까운 끝점까지의 면적을 뺄 수 있다는 사실의 수학적 인코딩입니다.

![두 점 사이의 곡선 아래 면적을 계산하는 문제를 한 점의 왼쪽 면적을 계산하는 것으로 왜 축소할 수 있는지 시각화.](../img/sub-area.svg)
:label:`fig_area-subtract`

따라서, 저희는 $F(x)$가 무엇인지 알아냄으로써 어떤 구간에 대한 적분이 무엇인지 알아낼 수 있습니다.

그렇게 하기 위해, 실험을 고려해 봅시다. 미적분에서 자주 하듯이, 값을 약간 이동했을 때 무슨 일이 일어나는지 상상해 봅시다. 위의 코멘트에서, 저희는 다음을 압니다.

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

이는 함수가 함수의 작은 조각 아래의 면적만큼 변한다는 것을 알려줍니다.

이것이 저희가 근사를 하는 지점입니다. 이런 작은 면적의 조각을 보면, 이 면적은 높이가 $f(x)$의 값이고 밑변이 $\epsilon$인 직사각형 면적에 가깝습니다. 사실, $\epsilon \rightarrow 0$일 때 이 근사가 점점 더 좋아진다는 것을 보일 수 있습니다. 따라서 저희는 다음과 같이 결론 내릴 수 있습니다.

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

그러나, 이제 저희는 다음을 알아챌 수 있습니다. 이것이 정확히 저희가 $F$의 도함수를 계산한다면 기대했을 패턴입니다! 따라서 저희는 다음의 다소 놀라운 사실을 봅니다.

$$
\frac{dF}{dx}(x) = f(x).
$$

이것이 *미적분의 기본 정리*입니다. 저희는 이를 다음과 같이 확장된 형태로 작성할 수 있습니다.
$$\frac{d}{dx}\int_0^x  f(y) \; dy = f(x).$$
:eqlabel:`eq_ftc`

이는 면적을 찾는 개념(*선험적으로* 다소 어려운)을 취하여, 도함수에 대한 진술(훨씬 더 완전히 이해된 것)로 축소합니다. 저희가 해야 할 마지막 코멘트는 이것이 정확히 $F(x)$가 무엇인지 알려주지 않는다는 것입니다. 사실 어떤 $C$에 대해서도 $F(x) + C$는 같은 도함수를 가집니다. 이는 적분 이론에서의 삶의 사실입니다. 다행스럽게도, 정적분으로 작업할 때, 상수가 사라지므로, 결과와 무관합니다.

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

이는 추상적인 무의미해 보일 수 있지만, 적분을 계산하는 데 완전히 새로운 관점을 제공했다는 것을 잠시 감사하도록 합시다. 저희의 목표는 더 이상 면적을 복구하려고 어떤 종류의 자르고-합하는 과정을 하는 것이 아니라, 도함수가 저희가 가진 함수인 함수를 찾기만 하면 됩니다! 이는 놀라운데, 왜냐하면 이제 저희는 :numref:`sec_derivative_table`의 테이블을 뒤집기만 함으로써 많은 다소 어려운 적분들을 나열할 수 있기 때문입니다. 예를 들어, $x^{n}$의 도함수가 $nx^{n-1}$임을 알고 있습니다. 따라서, 기본 정리 :eqref:`eq_ftc`를 사용하여 다음과 같이 말할 수 있습니다.

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

유사하게, $e^{x}$의 도함수가 자기 자신임을 알고 있으므로, 그것은 다음을 의미합니다.

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

이런 식으로, 저희는 미분 적분학의 아이디어를 자유롭게 활용하여 적분의 전체 이론을 발전시킬 수 있습니다. 모든 적분 규칙은 이 한 가지 사실에서 유도됩니다.

## 변수 변환
:label:`subsec_integral_example`

미분과 마찬가지로, 적분의 계산을 더 다루기 쉽게 만드는 많은 규칙이 있습니다. 사실, 모든 미분 적분학 규칙(예: 곱 규칙, 합 규칙, 그리고 연쇄 규칙)은 적분 적분학에 대해 해당하는 규칙(각각 부분 적분, 적분의 선형성, 그리고 변수 변환 공식)을 가집니다. 이 절에서는, 저희는 이 리스트에서 가장 중요하다고 할 수 있는 변수 변환 공식에 다이빙할 것입니다.

먼저, 그 자체로 적분인 함수가 있다고 가정해 봅시다.

$$
F(x) = \int_0^x f(y) \; dy.
$$

이 함수를 다른 함수와 합성하여 $F(u(x))$를 얻을 때 어떻게 보이는지 알고 싶다고 가정해 봅시다. 연쇄 규칙에 의해, 저희는 다음을 압니다.

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

위와 같이 기본 정리 :eqref:`eq_ftc`를 사용하여 이를 적분에 대한 진술로 바꿀 수 있습니다. 이는 다음을 제공합니다.

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

$F$가 그 자체로 적분임을 떠올리면 왼쪽이 다음과 같이 다시 작성될 수 있음을 알 수 있습니다.

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

유사하게, $F$가 적분임을 떠올리면 기본 정리 :eqref:`eq_ftc`를 사용하여 $\frac{dF}{dx} = f$임을 인식할 수 있고, 따라서 다음과 같이 결론 내릴 수 있습니다.

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

이것이 *변수 변환* 공식입니다.

더 직관적인 유도를 위해, $x$와 $x+\epsilon$ 사이에서 $f(u(x))$의 적분을 취할 때 무슨 일이 일어나는지 고려해 보십시오. 작은 $\epsilon$에 대해, 이 적분은 대략 $\epsilon f(u(x))$, 즉 연관된 직사각형의 면적입니다. 이제, 이를 $u(x)$에서 $u(x+\epsilon)$까지의 $f(y)$의 적분과 비교해 봅시다. 저희는 $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$임을 알고 있으므로, 이 직사각형의 면적은 대략 $\epsilon \frac{du}{dx}(x)f(u(x))$입니다. 따라서, 이 두 직사각형의 면적이 일치하도록 만들기 위해서는, :numref:`fig_rect-transform`에서 보여지는 것처럼 첫 번째 것에 $\frac{du}{dx}(x)$를 곱해야 합니다.

![변수 변환 하에서 단일 얇은 직사각형의 변환을 시각화.](../img/rect-trans.svg)
:label:`fig_rect-transform`

이는 저희에게 다음을 알려줍니다.

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$

이것이 단일 작은 직사각형에 대해 표현된 변수 변환 공식입니다.

만약 $u(x)$와 $f(x)$가 적절히 선택된다면, 이는 매우 복잡한 적분의 계산을 허용할 수 있습니다. 예를 들어, $f(y) = 1$과 $u(x) = e^{-x^{2}}$를 선택한다면(이는 $\frac{du}{dx}(x) = -2xe^{-x^{2}}$를 의미합니다), 이는 예를 들어 다음을 보일 수 있습니다.

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

따라서 재배열하여

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## 부호 관례에 대한 코멘트

날카로운 눈을 가진 독자들은 위의 계산에 대해 이상한 것을 관찰할 것입니다. 즉, 다음과 같은 계산은

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

음수를 생성할 수 있습니다. 면적에 대해 생각할 때, 음수 값을 보는 것은 이상할 수 있으므로, 관례가 무엇인지 파헤쳐 볼 가치가 있습니다.

수학자들은 부호 있는 면적의 개념을 취합니다. 이는 두 가지 방식으로 나타납니다. 첫째, 만약 때때로 0보다 작은 함수 $f(x)$를 고려하면, 면적도 음수일 것입니다. 그래서 예를 들어

$$
\int_0^{1} (-1)\;dx = -1.
$$

유사하게, 왼쪽에서 오른쪽이 아니라 오른쪽에서 왼쪽으로 진행되는 적분도 음의 면적으로 취해집니다.

$$
\int_0^{-1} 1\; dx = -1.
$$

표준 면적(양수 함수의 왼쪽에서 오른쪽)은 항상 양수입니다. 그것을 뒤집어서 얻은 것은 무엇이든(예를 들어, 음수의 적분을 얻기 위해 $x$축 위로 뒤집거나, 잘못된 순서로 적분을 얻기 위해 $y$축 위로 뒤집는 것) 음의 면적을 생성할 것입니다. 그리고 실제로, 두 번 뒤집으면 양의 면적을 가지도록 상쇄되는 한 쌍의 음의 부호가 생깁니다.

$$
\int_0^{-1} (-1)\;dx =  1.
$$

만약 이 논의가 친숙하게 들린다면, 그것은 사실입니다! :numref:`sec_geometry-linear-algebraic-ops`에서 저희는 행렬식이 매우 비슷한 방식으로 부호 있는 면적을 어떻게 나타내는지 논의했습니다.

## 다중 적분
어떤 경우에는, 저희가 더 높은 차원에서 작업해야 할 것입니다. 예를 들어, $f(x, y)$와 같은 두 변수의 함수가 있고 $x$가 $[a, b]$ 범위에서 변하고 $y$가 $[c, d]$ 범위에서 변할 때 $f$ 아래의 부피를 알고 싶다고 가정해 봅시다.

```{.python .input}
#@tab mxnet
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy())
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

저희는 이를 다음과 같이 작성합니다.

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

이 적분을 계산하고 싶다고 가정해 봅시다. 저의 주장은 먼저 $x$에 대한 적분을 계산한 다음 $y$에 대한 적분으로 이동하여 반복적으로 이를 할 수 있다는 것입니다. 즉,

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

이것이 왜 그런지 봅시다.

위의 그림을 고려하십시오. 여기서 저희는 함수를 정수 좌표 $i, j$로 인덱싱할 $\epsilon \times \epsilon$ 정사각형으로 분할했습니다. 이 경우, 저희의 적분은 대략 다음과 같습니다.

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

문제를 이산화하고 나면, 저희는 값을 변경하는 것에 대해 걱정하지 않고 원하는 순서로 이 정사각형의 값을 더할 수 있습니다. 이는 :numref:`fig_sum-order`에 표시되어 있습니다. 특히, 저희는 다음과 같이 말할 수 있습니다.

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![많은 정사각형에 대한 합을 먼저 열에 대한 합(1)으로 분해한 다음 열 합을 함께 더하는 방법을 보여주는 그림(2).](../img/sum-order.svg)
:label:`fig_sum-order`

내부의 합은 정확히 다음 적분의 이산화입니다.

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

마지막으로, 이 두 식을 결합하면 다음을 얻는다는 점에 유의하십시오.

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

따라서 모두 종합하면, 저희는 다음을 가집니다.

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

일단 이산화되면, 저희가 한 일은 단지 숫자 목록을 더하는 순서를 재배열한 것뿐이라는 점에 유의하십시오. 이는 그것이 아무것도 아닌 것처럼 보이게 할 수 있지만, 이 결과(*푸비니의 정리*라고 불리는)는 항상 참이 아닙니다! 머신러닝을 할 때 마주치는 종류의 수학(연속 함수)의 경우, 우려할 필요가 없지만, 실패하는 예제를 만들 수 있습니다(예를 들어, 직사각형 $[0,2]\times[0,1]$ 위의 함수 $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$).

먼저 $x$에 대한 적분을 한 다음 $y$에 대한 적분을 하는 선택은 임의적이었다는 점에 유의하십시오. 저희는 동등하게 먼저 $y$를 한 다음 $x$를 선택하여 다음을 볼 수 있었습니다.

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

종종, 저희는 벡터 표기법으로 축약하고, $U = [a, b]\times [c, d]$에 대해 이것이 다음과 같다고 말할 것입니다.

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## 다중 적분에서의 변수 변환
:eqref:`eq_change_var`의 단일 변수와 마찬가지로, 고차원 적분 내에서 변수를 변경하는 능력은 핵심 도구입니다. 유도 없이 결과를 요약합시다.

저희는 적분 영역을 재매개변수화하는 함수가 필요합니다. 저희는 이를 $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$, 즉 $n$개의 실수 변수를 받아 다른 $n$개를 반환하는 어떤 함수든 취할 수 있습니다. 식을 깔끔하게 유지하기 위해, 저희는 $\phi$가 *단사*라고 가정할 것입니다. 이는 그것이 결코 자신을 접지 않는다는 것을 말합니다($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$).

이 경우, 저희는 다음과 같이 말할 수 있습니다.

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

여기서 $D\phi$는 $\phi$의 *야코비안*이며, 이는 $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$의 편도함수의 행렬입니다.

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

자세히 보면, 이는 단일 변수 연쇄 규칙 :eqref:`eq_change_var`와 유사하다는 것을 알 수 있는데, $\frac{du}{dx}(x)$ 항을 $\left|\det(D\phi(\mathbf{x}))\right|$로 대체했다는 점만 다릅니다. 이 항을 어떻게 해석할 수 있는지 봅시다. $\frac{du}{dx}(x)$ 항이 $u$를 적용함으로써 저희의 $x$축을 얼마나 늘렸는지를 말하기 위해 존재했음을 떠올리십시오. 더 높은 차원에서의 같은 과정은 $\boldsymbol{\phi}$를 적용함으로써 작은 정사각형(또는 작은 *초정육면체*)의 면적(또는 부피, 또는 초부피)을 얼마나 늘리는지 결정하는 것입니다. 만약 $\boldsymbol{\phi}$가 행렬에 의한 곱셈이었다면, 저희는 행렬식이 이미 답을 어떻게 제공하는지 압니다.

약간의 작업으로, 저희가 도함수와 그래디언트로 선이나 평면으로 근사할 수 있었던 것과 같은 방식으로 *야코비안*이 한 점에서 다변수 함수 $\boldsymbol{\phi}$에 대한 행렬에 의한 가장 좋은 근사를 제공함을 보일 수 있습니다. 따라서 야코비안의 행렬식은 정확히 저희가 1차원에서 식별한 스케일링 인자를 반영합니다.

여기에 세부사항을 채우는 데 약간의 작업이 필요하므로, 지금 명확하지 않더라도 걱정하지 마십시오. 나중에 사용할 적어도 하나의 예제를 봅시다. 적분을 고려해 보십시오.

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

이 적분을 직접 가지고 노는 것은 저희를 아무 데도 데려가지 않을 것이지만, 변수를 변경하면 상당한 진전을 이룰 수 있습니다. 만약 $\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$로 놓으면(이는 $x = r \cos(\theta)$, $y = r \sin(\theta)$라고 말하는 것입니다), 변수 변환 공식을 적용하여 이것이 다음과 같음을 볼 수 있습니다.

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

여기서

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

따라서, 적분은 다음과 같습니다.

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

여기서 마지막 등식은 :numref:`subsec_integral_example`에서 사용한 것과 같은 계산에 의해 따릅니다.

저희는 :numref:`sec_random_variables`에서 연속 확률 변수를 공부할 때 이 적분을 다시 만날 것입니다.

## 요약

* 적분 이론은 면적이나 부피에 대한 질문에 답할 수 있게 해줍니다.
* 미적분의 기본 정리는 적분 중인 함수의 값에 의해 어떤 지점까지의 면적의 도함수가 주어진다는 관찰을 통해 도함수에 대한 지식을 활용하여 면적을 계산할 수 있게 해줍니다.
* 더 높은 차원의 적분은 단일 변수 적분을 반복함으로써 계산될 수 있습니다.

## 연습문제
1. $\int_1^2 \frac{1}{x} \;dx$는 무엇입니까?
2. 변수 변환 공식을 사용하여 $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$를 적분하십시오.
3. $\int_{[0,1]^2} xy \;dx\;dy$는 무엇입니까?
4. 변수 변환 공식을 사용하여 $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$와 $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$를 계산하여 그것들이 다름을 확인하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1092)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1093)
:end_tab:
