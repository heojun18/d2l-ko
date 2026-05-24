# 단변수 미적분
:label:`sec_single_variable_calculus`

:numref:`sec_calculus`에서, 저희는 미분 적분학의 기본 요소를 보았습니다. 이 절에서는 미적분의 기초에 대해 더 깊이 다이빙하고, 머신러닝의 맥락에서 그것을 어떻게 이해하고 적용할 수 있는지 살펴봅니다.

## 미분 적분학
미분 적분학은 근본적으로 함수가 작은 변화 하에서 어떻게 동작하는지에 대한 연구입니다. 이것이 왜 딥러닝의 핵심인지 알아보기 위해, 예제를 고려해 보겠습니다.

편의를 위해 가중치가 단일 벡터 $\mathbf{w} = (w_1, \ldots, w_n)$로 연결된 심층 신경망이 있다고 가정해 보십시오. 훈련 데이터셋이 주어지면, 이 데이터셋에 대한 신경망의 손실을 고려하는데, 이를 $\mathcal{L}(\mathbf{w})$로 쓸 것입니다.

이 함수는 매우 복잡하며, 이 데이터셋에 대한 주어진 아키텍처의 모든 가능한 모델의 성능을 인코딩하므로, 어떤 가중치 집합 $\mathbf{w}$가 손실을 최소화할지 알기는 거의 불가능합니다. 따라서 실제로는, 저희는 종종 가중치를 *무작위로* 초기화하고, 그런 다음 가능한 한 빠르게 손실을 감소시키는 방향으로 반복적으로 작은 걸음을 떼는 것으로 시작합니다.

그러면 질문은 표면적으로는 더 쉽지 않은 것이 됩니다. 가중치가 가능한 한 빠르게 감소하는 방향을 어떻게 찾을까요? 이를 파고들기 위해, 먼저 단일 가중치만 있는 경우를 살펴보겠습니다. 단일 실숫값 $x$에 대해 $L(\mathbf{w}) = L(x)$입니다.

$x$를 취하고 작은 양만큼 $x + \epsilon$로 변경할 때 무슨 일이 일어나는지 이해하려고 노력해 봅시다. 구체적으로 말하자면, $\epsilon = 0.0000001$과 같은 숫자를 생각해 보십시오. 무슨 일이 일어나는지 시각화하기 위해, 예제 함수 $f(x) = \sin(x^x)$를 $[0, 3]$ 범위에서 그래프로 그려봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

이 큰 스케일에서, 함수의 동작은 단순하지 않습니다. 그러나, 범위를 $[1.75,2.25]$와 같이 더 작은 것으로 줄이면, 그래프가 훨씬 더 단순해지는 것을 봅니다.

```{.python .input}
#@tab mxnet
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

이를 극단으로 가져가면, 작은 세그먼트로 확대하면 동작이 훨씬 더 단순해집니다. 단지 직선입니다.

```{.python .input}
#@tab mxnet
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

이것이 단변수 미적분의 핵심 관찰입니다. 친숙한 함수의 동작은 충분히 작은 범위에서 직선으로 모델링될 수 있습니다. 이는 대부분의 함수에 대해, 함수의 $x$ 값을 약간 이동하면 출력 $f(x)$도 약간 이동할 것이라고 기대하는 것이 합리적임을 의미합니다. 저희가 답해야 할 유일한 질문은 "입력의 변화에 비해 출력의 변화는 얼마나 큰가? 절반 크기인가? 두 배 크기인가?"입니다.

따라서, 함수 입력의 작은 변화에 대한 함수 출력의 변화 비율을 고려할 수 있습니다. 이를 다음과 같이 형식적으로 쓸 수 있습니다.

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

이것은 이미 코드에서 가지고 놀기에 충분합니다. 예를 들어, $L(x) = x^{2} + 1701(x-4)^3$임을 안다고 가정하면, $x = 4$ 지점에서 이 값이 얼마나 큰지 다음과 같이 볼 수 있습니다.

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

이제, 주의깊게 관찰하면, 이 숫자의 출력이 의심스럽게 $8$에 가깝다는 것을 알아챌 것입니다. 사실, $\epsilon$를 감소시키면, 값이 점점 더 $8$에 가까워지는 것을 볼 것입니다. 따라서 저희는 올바르게도, $x=4$ 지점에서 저희가 찾는 값(입력의 변화가 출력을 변화시키는 정도)이 $8$이어야 한다고 결론 내릴 수 있습니다. 수학자가 이 사실을 인코딩하는 방법은 다음과 같습니다.

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

약간의 역사적 여담으로, 신경망 연구의 첫 몇십 년 동안, 과학자들은 작은 섭동 하에서 손실 함수가 어떻게 변하는지 평가하기 위해 이 알고리즘(*유한 차분법*)을 사용했습니다. 가중치를 변경하고 손실이 어떻게 변했는지 보기만 하면 됩니다. 이는 계산적으로 비효율적이며, 한 변수의 단일 변화가 손실에 어떻게 영향을 미치는지 보기 위해 손실 함수의 두 번의 평가가 필요합니다. 만약 저희가 이를 단지 몇 천 개의 매개변수로라도 시도했다면, 전체 데이터셋에 대해 네트워크의 수천 번의 평가가 필요했을 것입니다! 1986년이 되어서야 :citet:`Rumelhart.Hinton.Williams.ea.1988`에서 소개된 *역전파 알고리즘*이 가중치의 *어떤* 변경이 함께 손실을 어떻게 변경할지를 데이터셋에 대한 네트워크의 단일 예측과 같은 계산 시간에 계산하는 방법을 제공했습니다.

저희의 예제로 돌아가서, 이 값 $8$은 $x$의 다른 값에 대해 다르므로, 이를 $x$의 함수로 정의하는 것이 의미가 있습니다. 더 형식적으로, 이 값 의존적 변화율은 *도함수*라고 하며, 다음과 같이 작성됩니다.

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

다른 텍스트는 도함수에 대해 다른 표기법을 사용할 것입니다. 예를 들어, 아래의 모든 표기법은 같은 것을 나타냅니다.

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

대부분의 저자는 단일 표기법을 선택하여 고수할 것이지만, 그것조차 보장되지 않습니다. 이 모든 것에 친숙해지는 것이 가장 좋습니다. 저희는 이 텍스트 전반에 걸쳐 $\frac{df}{dx}$ 표기법을 사용할 것이며, 복잡한 식의 도함수를 취하고 싶을 때를 제외하고는 그러한 경우에 다음과 같은 식을 작성하기 위해 $\frac{d}{dx}f$를 사용할 것입니다.
$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

종종, $x$의 작은 변경을 했을 때 함수가 어떻게 변하는지 보기 위해 도함수의 정의 :eqref:`eq_der_def`를 다시 풀어보는 것이 직관적으로 유용합니다.

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

마지막 방정식은 명시적으로 부각시킬 가치가 있습니다. 이는 어떤 함수든 취하고 입력을 작은 양만큼 변경하면, 출력이 도함수에 의해 스케일된 그 작은 양만큼 변할 것이라고 알려줍니다.

이런 식으로, 저희는 도함수를 입력의 변화로부터 출력에서 얻는 변화의 크기를 알려주는 스케일링 인자로 이해할 수 있습니다.

## 미적분 규칙
:label:`sec_derivative_table`

이제 명시적인 함수의 도함수를 어떻게 계산하는지 이해하는 작업으로 넘어갑니다. 미적분의 완전한 형식적 처리는 모든 것을 제1원리에서 유도할 것입니다. 저희는 여기서 이 유혹에 빠지지 않고, 대신 마주치는 일반적인 규칙에 대한 이해를 제공할 것입니다.

### 일반적인 도함수
:numref:`sec_calculus`에서 본 것처럼, 도함수를 계산할 때 종종 일련의 규칙을 사용하여 계산을 몇 가지 핵심 함수로 축소할 수 있습니다. 참조의 편의를 위해 여기서 그것들을 반복합니다.

* **상수의 도함수.** $\frac{d}{dx}c = 0$.
* **선형 함수의 도함수.** $\frac{d}{dx}(ax) = a$.
* **거듭제곱 규칙.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **지수의 도함수.** $\frac{d}{dx}e^x = e^x$.
* **로그의 도함수.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### 도함수 규칙
모든 도함수가 별도로 계산되어 테이블에 저장되어야 한다면, 미분 적분학은 거의 불가능할 것입니다. 위의 도함수를 일반화하고 $f(x) = \log\left(1+(x-1)^{10}\right)$의 도함수를 찾는 것과 같은 더 복잡한 도함수를 계산할 수 있는 것은 수학의 선물입니다. :numref:`sec_calculus`에서 언급된 것처럼, 그렇게 하는 열쇠는 함수를 취하여 다양한 방식, 가장 중요하게는 합, 곱, 그리고 합성으로 결합할 때 무슨 일이 일어나는지를 코드화하는 것입니다.

* **합 규칙.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **곱 규칙.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **연쇄 규칙.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

이러한 규칙을 이해하기 위해 :eqref:`eq_small_change`를 어떻게 사용할 수 있는지 봅시다. 합 규칙의 경우, 다음 추론 체인을 고려해 보십시오.

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

이 결과를 $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$라는 사실과 비교하면, 원하는 대로 $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$임을 알 수 있습니다. 여기서의 직관은 입력 $x$를 변경할 때, $g$와 $h$가 출력의 변화에 $\frac{dg}{dx}(x)$와 $\frac{dh}{dx}(x)$만큼 공동으로 기여한다는 것입니다.


곱은 더 미묘하며, 이러한 식을 다루는 방법에 대한 새로운 관찰이 필요할 것입니다. 저희는 이전과 같이 :eqref:`eq_small_change`를 사용하여 시작할 것입니다.

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$


이는 위에서 한 계산과 닮았고, 실제로 저희의 답($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$)이 $\epsilon$ 옆에 앉아 있는 것을 보지만, 그 크기 $\epsilon^{2}$ 항의 문제가 있습니다. 저희는 이를 *고차 항*이라고 부를 것인데, 왜냐하면 $\epsilon^2$의 거듭제곱이 $\epsilon^1$의 거듭제곱보다 높기 때문입니다. 나중 절에서 저희가 때때로 이러한 것들을 추적하고 싶을 것임을 볼 것이지만, 지금은 $\epsilon = 0.0000001$이면, $\epsilon^{2}= 0.0000000000001$이며, 이는 훨씬 더 작다는 것을 관찰하십시오. $\epsilon \rightarrow 0$을 보내면, 저희는 안전하게 고차 항을 무시할 수 있습니다. 이 부록의 일반적인 관례로, 저희는 두 항이 고차 항까지 같다는 것을 나타내기 위해 "$\approx$"를 사용할 것입니다. 그러나, 더 형식적이 되고 싶다면 차분 몫을 검토할 수 있습니다.

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

그리고 $\epsilon \rightarrow 0$을 보낼 때, 오른쪽 항도 0으로 간다는 것을 봅니다.

마지막으로, 연쇄 규칙으로, 저희는 다시 이전처럼 :eqref:`eq_small_change`를 사용하여 진행할 수 있고 다음을 볼 수 있습니다.

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

여기서 두 번째 줄에서 저희는 함수 $g$가 그 입력($h(x)$)이 작은 양 $\epsilon \frac{dh}{dx}(x)$만큼 이동된 것으로 봅니다.

이러한 규칙은 저희에게 본질적으로 원하는 어떤 식이든 계산할 수 있는 유연한 도구 집합을 제공합니다. 예를 들어,

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

여기서 각 줄은 다음 규칙을 사용했습니다.

1. 연쇄 규칙과 로그의 도함수.
2. 합 규칙.
3. 상수의 도함수, 연쇄 규칙, 그리고 거듭제곱 규칙.
4. 합 규칙, 선형 함수의 도함수, 상수의 도함수.

이 예제를 한 후 두 가지가 분명해야 합니다.

1. 합, 곱, 상수, 거듭제곱, 지수, 그리고 로그를 사용하여 쓸 수 있는 어떤 함수든 이러한 규칙을 따름으로써 기계적으로 그 도함수를 계산할 수 있습니다.
2. 인간이 이러한 규칙을 따르는 것은 지루하고 오류가 발생하기 쉬울 수 있습니다!

다행스럽게도, 이 두 사실은 함께 앞으로 나아갈 길을 시사합니다. 이는 기계화를 위한 완벽한 후보입니다! 사실 이 절의 나중에 다시 살펴볼 역전파가 정확히 그것입니다.

### 선형 근사
도함수를 다룰 때, 위에서 사용된 근사를 기하학적으로 해석하는 것이 종종 유용합니다. 특히, 다음 방정식이

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

점 $(x, f(x))$를 지나고 기울기가 $\frac{df}{dx}(x)$인 직선으로 $f$의 값을 근사한다는 점에 유의하십시오. 이런 식으로 저희는 도함수가 아래에 표시된 것처럼 함수 $f$에 대한 선형 근사를 제공한다고 말합니다.

```{.python .input}
#@tab mxnet
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### 고차 도함수

이제 표면적으로는 이상해 보일 수 있는 무언가를 해 보겠습니다. 함수 $f$를 취하고 도함수 $\frac{df}{dx}$를 계산합니다. 이는 어떤 점에서든 $f$의 변화율을 제공합니다.

그러나, 도함수 $\frac{df}{dx}$는 그 자체로 함수로 볼 수 있으므로, $\frac{df}{dx}$의 도함수를 계산하여 $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$를 얻는 것을 막을 수 있는 것은 없습니다. 저희는 이를 $f$의 2계 도함수라고 부를 것입니다. 이 함수는 $f$의 변화율의 변화율, 또는 다시 말해, 변화율이 어떻게 변하는지입니다. 저희는 도함수를 임의의 횟수만큼 적용하여 *$n$계* 도함수라고 불리는 것을 얻을 수 있습니다. 표기를 깔끔하게 유지하기 위해, 저희는 $n$계 도함수를 다음과 같이 표기할 것입니다.

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

이것이 *왜* 유용한 개념인지 이해해 보겠습니다. 아래에서, 저희는 $f^{(2)}(x)$, $f^{(1)}(x)$, 그리고 $f(x)$를 시각화합니다.

먼저, 2계 도함수 $f^{(2)}(x)$가 양의 상수인 경우를 고려해 보십시오. 이는 1계 도함수의 기울기가 양수임을 의미합니다. 결과적으로, 1계 도함수 $f^{(1)}(x)$는 음수로 시작하여 어떤 점에서 0이 되고, 그런 다음 결국 양수가 될 수 있습니다. 이는 저희에게 원래 함수 $f$의 기울기를 알려주며, 따라서 함수 $f$ 자체가 감소하고, 평평해지고, 그런 다음 증가합니다. 다시 말해, 함수 $f$는 위로 굽고, :numref:`fig_positive-second`에 표시된 것처럼 단일 최솟값을 가집니다.

![2계 도함수가 양의 상수라고 가정하면, 1계 도함수는 증가하고 있으며, 이는 함수 자체가 최솟값을 가짐을 의미합니다.](../img/posSecDer.svg)
:label:`fig_positive-second`


둘째, 2계 도함수가 음의 상수라면, 그것은 1계 도함수가 감소하고 있다는 것을 의미합니다. 이는 1계 도함수가 양수로 시작하여 어떤 점에서 0이 되고, 그런 다음 음수가 될 수 있음을 의미합니다. 따라서 함수 $f$ 자체는 증가하고, 평평해지고, 그런 다음 감소합니다. 다시 말해, 함수 $f$는 아래로 굽고, :numref:`fig_negative-second`에 표시된 것처럼 단일 최댓값을 가집니다.

![2계 도함수가 음의 상수라고 가정하면, 1계 도함수는 감소하고 있으며, 이는 함수 자체가 최댓값을 가짐을 의미합니다.](../img/negSecDer.svg)
:label:`fig_negative-second`


셋째, 2계 도함수가 항상 0이라면, 1계 도함수는 결코 변하지 않을 것입니다(상수입니다!). 이는 $f$가 고정된 비율로 증가(또는 감소)하고, $f$ 자체가 :numref:`fig_zero-second`에 표시된 것처럼 직선임을 의미합니다.

![2계 도함수가 0이라고 가정하면, 1계 도함수는 상수이며, 이는 함수 자체가 직선임을 의미합니다.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

요약하면, 2계 도함수는 함수 $f$가 굽는 방식을 설명하는 것으로 해석될 수 있습니다. 양의 2계 도함수는 위쪽 곡선으로 이어지고, 음의 2계 도함수는 $f$가 아래쪽으로 굽는다는 것을 의미하며, 0인 2계 도함수는 $f$가 전혀 굽지 않는다는 것을 의미합니다.

이를 한 단계 더 진행해 보겠습니다. 함수 $g(x) = ax^{2}+ bx + c$를 고려해 보십시오. 그러면 다음을 계산할 수 있습니다.

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

만약 마음 속에 어떤 원래 함수 $f(x)$가 있다면, 처음 두 도함수를 계산하고 이 계산과 일치하는 $a, b$, 그리고 $c$의 값을 찾을 수 있습니다. 이전 절에서 1계 도함수가 직선으로 가장 좋은 근사를 제공한다는 것을 본 것과 유사하게, 이 구성은 이차식으로 가장 좋은 근사를 제공합니다. $f(x) = \sin(x)$에 대해 이를 시각화해 봅시다.

```{.python .input}
#@tab mxnet
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

저희는 다음 절에서 이 아이디어를 *테일러 급수*의 아이디어로 확장할 것입니다.

### 테일러 급수


*테일러 급수*는 점 $x_0$에서 처음 $n$개 도함수의 값, 즉 $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$이 주어졌을 때 함수 $f(x)$를 근사하는 방법을 제공합니다. 아이디어는 $x_0$에서 모든 주어진 도함수와 일치하는 차수 $n$ 다항식을 찾는 것입니다.

저희는 이전 절에서 $n=2$의 경우를 보았고 약간의 대수가 이것이 다음과 같음을 보여줍니다.

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

위에서 볼 수 있듯이, 분모의 $2$는 $x^2$의 두 도함수를 취할 때 얻는 $2$를 소거하기 위해 거기 있으며, 다른 항들은 모두 0입니다. 같은 논리가 1계 도함수와 값 자체에 적용됩니다.

논리를 $n=3$으로 더 밀어붙이면, 다음과 같이 결론지을 것입니다.

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

여기서 $6 = 3 \times 2 = 3!$은 $x^3$의 세 도함수를 취할 때 앞에 얻는 상수에서 옵니다.


또한, 다음과 같이 하여 차수 $n$ 다항식을 얻을 수 있습니다.

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

여기서 표기법은

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


사실, $P_n(x)$는 저희 함수 $f(x)$에 대한 가장 좋은 $n$차 다항식 근사로 볼 수 있습니다.

위 근사의 오차에 대해 완전히 다이빙하지는 않을 것이지만, 무한 극한을 언급할 가치가 있습니다. 이 경우, $\cos(x)$ 또는 $e^{x}$와 같이 잘 동작하는 함수(실해석 함수로 알려진)의 경우, 저희는 무한한 수의 항을 써서 정확히 같은 함수를 근사할 수 있습니다.

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

$f(x) = e^{x}$를 예로 들어보십시오. $e^{x}$는 자체 도함수이므로, $f^{(n)}(x) = e^{x}$임을 알고 있습니다. 따라서, $e^{x}$는 $x_0 = 0$에서 테일러 급수를 취함으로써 재구성될 수 있습니다. 즉,

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

코드에서 이것이 어떻게 작동하는지 보고, 테일러 근사의 차수를 증가시키면 원하는 함수 $e^x$에 더 가까워지는 것을 관찰해 보겠습니다.

```{.python .input}
#@tab mxnet
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

테일러 급수는 두 가지 주요 응용을 가집니다.

1. *이론적 응용*: 종종 너무 복잡한 함수를 이해하려고 할 때, 테일러 급수를 사용하면 그것을 직접 다룰 수 있는 다항식으로 바꿀 수 있습니다.

2. *수치적 응용*: $e^{x}$ 또는 $\cos(x)$와 같은 일부 함수는 기계가 계산하기 어렵습니다. 그들은 고정된 정밀도로 값의 테이블을 저장할 수 있고(이것은 종종 수행됩니다), 그러나 여전히 "$\cos(1)$의 1000번째 자릿수는 무엇인가?"와 같은 열린 질문을 남겨둡니다. 테일러 급수는 종종 이러한 질문에 답하는 데 도움이 됩니다.


## 요약

* 도함수는 입력을 작은 양만큼 변경할 때 함수가 어떻게 변하는지 표현하는 데 사용될 수 있습니다.
* 기본 도함수는 도함수 규칙을 사용하여 결합되어 임의로 복잡한 도함수를 만들 수 있습니다.
* 도함수는 반복되어 2계 또는 더 높은 차수 도함수를 얻을 수 있습니다. 차수의 각 증가는 함수의 동작에 대한 더 세분화된 정보를 제공합니다.
* 단일 데이터 예제의 도함수에 있는 정보를 사용하여, 테일러 급수에서 얻은 다항식으로 잘 동작하는 함수를 근사할 수 있습니다.


## 연습문제

1. $x^3-4x+1$의 도함수는 무엇입니까?
2. $\log(\frac{1}{x})$의 도함수는 무엇입니까?
3. 참 또는 거짓: $f'(x) = 0$이면 $f$는 $x$에서 최댓값 또는 최솟값을 가집니까?
4. $x\ge0$에 대해 $f(x) = x\log(x)$의 최솟값은 어디입니까(여기서 $f$가 $f(0)$에서 극한값 $0$을 가진다고 가정합니다)?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab:
