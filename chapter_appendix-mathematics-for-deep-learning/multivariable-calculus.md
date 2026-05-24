# 다변수 미적분
:label:`sec_multivariable_calculus`

이제 단일 변수의 함수의 도함수에 대해 꽤 강력한 이해를 가졌으므로, 잠재적으로 수십억 개의 가중치의 손실 함수를 고려하고 있던 원래의 질문으로 돌아갑시다.

## 고차원 미분
:numref:`sec_single_variable_calculus`가 저희에게 알려주는 것은, 이 수십억 개의 가중치 중 하나만 변경하고 다른 모든 것을 고정시킨다면, 무슨 일이 일어날지 안다는 것입니다! 이것은 단일 변수의 함수에 불과하므로, 다음과 같이 쓸 수 있습니다.

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

다른 변수를 고정하면서 한 변수에 대한 도함수를 *편도함수*라고 부를 것이며, :eqref:`eq_part_der`의 도함수에 대해 $\frac{\partial}{\partial w_1}$ 표기법을 사용할 것입니다.

이제, 이것을 가져다가 $w_2$를 약간 $w_2 + \epsilon_2$로 변경해 봅시다.

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

저희는 다시 $\epsilon_1\epsilon_2$가 :eqref:`eq_part_der`에서 본 것과 함께, 이전 절에서 $\epsilon^{2}$를 버릴 수 있었던 것과 같은 방식으로 버릴 수 있는 고차 항이라는 아이디어를 사용했습니다. 이런 식으로 계속하면, 다음과 같이 쓸 수 있습니다.

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

이것이 엉망진창처럼 보일 수 있지만, 오른쪽의 합이 정확히 내적처럼 보인다는 것을 알아챔으로써 이것을 더 친숙하게 만들 수 있으므로, 다음과 같이 놓으면

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \textrm{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

그러면

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

저희는 벡터 $\nabla_{\mathbf{w}} L$를 $L$의 *그래디언트*라고 부를 것입니다.

방정식 :eqref:`eq_nabla_use`는 잠시 숙고할 가치가 있습니다. 이것은 저희가 1차원에서 마주친 것과 정확히 같은 형식을 가지며, 단지 모든 것을 벡터와 내적으로 변환했을 뿐입니다. 이는 저희가 입력에 어떤 섭동이 주어졌을 때 함수 $L$이 대략 어떻게 변할지 알려주는 것을 허용합니다. 다음 절에서 볼 것처럼, 이는 저희에게 그래디언트에 포함된 정보를 사용하여 어떻게 학습할 수 있는지 기하학적으로 이해하는 중요한 도구를 제공할 것입니다.

그러나 먼저, 예제로 이 근사가 작동하는 것을 봅시다. 다음 함수로 작업하고 있다고 가정해 보십시오.

$$
f(x, y) = \log(e^x + e^y) \textrm{ with gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

$(0, \log(2))$와 같은 점을 보면, 다음을 봅니다.

$$
f(x, y) = \log(3) \textrm{ with gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

따라서, $(\epsilon_1, \log(2) + \epsilon_2)$에서 $f$를 근사하고 싶다면, :eqref:`eq_nabla_use`의 특정 인스턴스를 가져야 한다는 것을 봅니다.

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

근사가 얼마나 좋은지 보기 위해 코드에서 이를 테스트할 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## 그래디언트의 기하학과 경사 하강법
다시 :eqref:`eq_nabla_use`의 식을 고려해 보십시오.

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

이를 사용하여 저희의 손실 $L$을 최소화하는 데 도움이 되고 싶다고 가정해 봅시다. :numref:`sec_autograd`에서 처음 설명된 경사 하강법의 알고리즘을 기하학적으로 이해해 봅시다. 저희가 할 일은 다음과 같습니다.

1. 초기 매개변수 $\mathbf{w}$에 대한 무작위 선택으로 시작합니다.
2. $\mathbf{w}$에서 $L$이 가장 빠르게 감소하는 방향 $\mathbf{v}$를 찾습니다.
3. 그 방향으로 작은 걸음을 떼십시오. $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. 반복합니다.

저희가 어떻게 해야 할지 정확히 모르는 유일한 것은 두 번째 단계에서 벡터 $\mathbf{v}$를 계산하는 것입니다. 저희는 이러한 방향을 *가장 가파른 하강 방향*이라고 부를 것입니다. :numref:`sec_geometry-linear-algebraic-ops`의 내적에 대한 기하학적 이해를 사용하여, 저희는 :eqref:`eq_nabla_use`를 다음과 같이 다시 쓸 수 있음을 봅니다.

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

편의를 위해 저희의 방향을 길이 1을 가지도록 취했고, $\mathbf{v}$와 $\nabla_{\mathbf{w}} L(\mathbf{w})$ 사이의 각도에 $\theta$를 사용했음에 유의하십시오. 만약 $L$을 가능한 한 빠르게 감소시키는 방향을 찾고 싶다면, 이 식을 가능한 한 음수로 만들고 싶습니다. 저희가 선택하는 방향이 이 방정식에 들어가는 유일한 방법은 $\cos(\theta)$를 통해서이며, 따라서 저희는 이 코사인을 가능한 한 음수로 만들고 싶습니다. 이제, 코사인의 모양을 떠올리면, $\cos(\theta) = -1$을 만들거나 동등하게 그래디언트와 저희가 선택한 방향 사이의 각도를 $\pi$ 라디안, 또는 동등하게 $180$도로 만듦으로써 이를 가능한 한 음수로 만들 수 있습니다. 이를 달성하는 유일한 방법은 정반대 방향으로 향하는 것입니다. $\mathbf{v}$를 $\nabla_{\mathbf{w}} L(\mathbf{w})$와 정확히 반대 방향을 가리키도록 선택하십시오!

이는 저희를 머신러닝에서 가장 중요한 수학적 개념 중 하나로 데려갑니다. 가장 가파른 하강 방향은 $-\nabla_{\mathbf{w}}L(\mathbf{w})$의 방향을 가리킵니다. 따라서 저희의 비공식적인 알고리즘은 다음과 같이 다시 쓸 수 있습니다.

1. 초기 매개변수 $\mathbf{w}$에 대한 무작위 선택으로 시작합니다.
2. $\nabla_{\mathbf{w}} L(\mathbf{w})$를 계산합니다.
3. 그 방향의 반대로 작은 걸음을 떼십시오. $\mathbf{w} \leftarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. 반복합니다.


이 기본 알고리즘은 많은 연구자들에 의해 많은 방식으로 수정되고 적응되었지만, 핵심 개념은 그것들 모두에서 동일하게 유지됩니다. 그래디언트를 사용하여 가능한 한 빠르게 손실을 감소시키는 방향을 찾고, 그 방향으로 걸음을 떼기 위해 매개변수를 업데이트합니다.

## 수학적 최적화에 대한 노트
이 책 전반에 걸쳐, 저희는 딥러닝 환경에서 마주치는 모든 함수가 명시적으로 최소화하기에는 너무 복잡하다는 실용적인 이유로 수치적 최적화 기법에 정면으로 초점을 맞춥니다.

그러나, 위에서 얻은 기하학적 이해가 함수를 직접 최적화하는 것에 대해 저희에게 무엇을 알려주는지 고려하는 것은 유용한 연습입니다.

어떤 함수 $L(\mathbf{x})$를 최소화하는 $\mathbf{x}_0$의 값을 찾고 싶다고 가정해 보십시오. 게다가 누군가가 저희에게 값을 주고 그것이 $L$을 최소화하는 값이라고 알려준다고 가정해 봅시다. 그들의 답이 그럴듯한지조차 확인할 수 있는 것이 있을까요?

다시 :eqref:`eq_nabla_use`를 고려해 보십시오.
$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$

만약 그래디언트가 0이 아니라면, 더 작은 $L$의 값을 찾기 위해 $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ 방향으로 걸음을 뗄 수 있다는 것을 압니다. 따라서, 저희가 진정으로 최솟값에 있다면, 이는 그럴 수 없습니다! 저희는 만약 $\mathbf{x}_0$가 최솟값이라면, $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$이라고 결론 내릴 수 있습니다. 저희는 $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$인 점들을 *임계점*이라고 부릅니다.

이는 좋습니다. 왜냐하면 일부 드문 설정에서, 저희는 *그래디언트가 0인 모든 점을 명시적으로 찾을 수 있고, 가장 작은 값을 가진 것을 찾을 수 있기* 때문입니다.

구체적인 예로, 함수를 고려해 보십시오.
$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$

이 함수는 다음의 도함수를 가집니다.
$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$

최솟값의 유일한 가능한 위치는 $x = -1, 0, 2$에서이며, 여기서 함수는 각각 $-5,0, -32$의 값을 취하므로, 저희는 $x = 2$일 때 저희 함수를 최소화한다고 결론 내릴 수 있습니다. 빠른 플롯이 이를 확인합니다.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

이는 이론적으로 또는 수치적으로 작업할 때 알아야 할 중요한 사실을 강조합니다. 함수를 최소화(또는 최대화)할 수 있는 유일한 가능한 점들은 그래디언트가 0인 점들일 것이지만, 그래디언트가 0인 모든 점이 진정한 *전역* 최솟값(또는 최댓값)인 것은 아닙니다.

## 다변수 연쇄 규칙
많은 항을 합성하여 만들 수 있는 네 변수($w, x, y$, 그리고 $z$)의 함수가 있다고 가정해 봅시다.

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

이러한 방정식의 체인은 신경망으로 작업할 때 일반적이므로, 이러한 함수의 그래디언트를 어떻게 계산할지 이해하려고 하는 것이 핵심입니다. 어떤 변수가 다른 변수와 직접적으로 관련되어 있는지 살펴보면 :numref:`fig_chain-1`에서 이 연결의 시각적 힌트를 보기 시작할 수 있습니다.

![노드가 값을 나타내고 간선이 함수적 의존성을 보여주는 위의 함수 관계.](../img/chain-net1.svg)
:label:`fig_chain-1`

:eqref:`eq_multi_func_def`의 모든 것을 그냥 합성하고 다음과 같이 작성하는 것을 막을 수 있는 것은 없습니다.

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

그런 다음 단일 변수 도함수를 사용하여 도함수를 취할 수 있지만, 그렇게 했다면 빠르게 항들에 압도되어 자신을 발견할 것이며, 그 중 많은 것은 반복입니다! 사실, 예를 들어, 다음을 볼 수 있습니다.

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

만약 저희가 또한 $\frac{\partial f}{\partial x}$를 계산하고 싶었다면, 많은 반복된 항과 두 도함수 사이의 많은 *공유된* 반복된 항을 가진 유사한 방정식으로 다시 끝났을 것입니다. 이는 엄청난 양의 낭비된 작업을 나타내며, 만약 저희가 이런 식으로 도함수를 계산해야 한다면, 전체 딥러닝 혁명은 시작되기 전에 정체되었을 것입니다!


문제를 분해해 봅시다. 본질적으로 $w, x, y$, 그리고 $z$가 모두 존재하지 않는다고 가정하고, $a$를 변경할 때 $f$가 어떻게 변하는지 이해하려고 시도하는 것부터 시작할 것입니다. 저희가 처음으로 그래디언트로 작업했을 때 했던 것처럼 추론할 것입니다. $a$를 가져다가 작은 양 $\epsilon$를 더해봅시다.

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

첫 번째 줄은 편도함수의 정의로부터 따르고, 두 번째는 그래디언트의 정의로부터 따릅니다. $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$ 식에서처럼 모든 도함수를 정확히 어디서 평가하는지 추적하는 것은 표기상으로 부담스러우므로, 저희는 종종 이것을 훨씬 더 기억하기 쉬운 것으로 축약합니다.

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

과정의 의미에 대해 생각하는 것이 유용합니다. 저희는 $f(u(a, b), v(a, b))$ 형태의 함수가 $a$의 변경에 따라 어떻게 값이 변하는지 이해하려고 노력하고 있습니다. 이것이 발생할 수 있는 두 가지 경로가 있습니다. $a \rightarrow u \rightarrow f$인 경로와 $a \rightarrow v \rightarrow f$인 경로가 있습니다. 저희는 연쇄 규칙을 통해 이 두 기여를 모두 계산할 수 있습니다. 각각 $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$와 $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$를 계산하고 더합니다.

:numref:`fig_chain-2`에 표시된 것처럼 오른쪽의 함수가 왼쪽에 연결된 것들에 의존하는 다른 함수 네트워크가 있다고 상상해 보십시오.

![연쇄 규칙의 또 다른 더 미묘한 예.](../img/chain-net2.svg)
:label:`fig_chain-2`

$\frac{\partial f}{\partial y}$와 같은 것을 계산하려면, $y$에서 $f$까지의 모든(이 경우 $3$개) 경로에 대해 합산해야 다음을 얻습니다.

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

이런 방식으로 연쇄 규칙을 이해하는 것은 그래디언트가 네트워크를 통해 어떻게 흐르는지, 그리고 LSTM(:numref:`sec_lstm`) 또는 잔차 계층(:numref:`sec_resnet`)과 같은 다양한 아키텍처 선택이 그래디언트 흐름을 제어함으로써 학습 과정을 어떻게 형성하는 데 도움이 될 수 있는지 이해하려고 할 때 큰 배당금을 지불할 것입니다.

## 역전파 알고리즘

이전 절의 :eqref:`eq_multi_func_def` 예제로 돌아가 보겠습니다.

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

만약 $\frac{\partial f}{\partial w}$를 계산하고 싶다면, 다변수 연쇄 규칙을 적용하여 다음을 볼 수 있습니다.

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

이 분해를 사용하여 $\frac{\partial f}{\partial w}$를 계산해 봅시다. 여기서 저희에게 필요한 것은 다양한 단일 단계 편도함수뿐임에 유의하십시오.

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

만약 이것을 코드로 작성한다면, 이것은 꽤 다루기 쉬운 식이 됩니다.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

그러나, 이것이 여전히 $\frac{\partial f}{\partial x}$와 같은 것을 계산하기 쉽게 만들지는 않는다는 점에 유의하십시오. 그 이유는 저희가 연쇄 규칙을 적용하기로 선택한 *방식* 때문입니다. 위에서 한 것을 보면, 저희는 가능할 때마다 분모에 $\partial w$를 항상 유지했습니다. 이런 식으로, 저희는 $w$가 다른 모든 변수를 어떻게 변경하는지 보면서 연쇄 규칙을 적용하기로 선택했습니다. 만약 저희가 원하는 것이 그것이라면, 이것은 좋은 아이디어일 것입니다. 그러나, 딥러닝에서의 저희의 동기를 다시 생각해 보십시오. 저희는 모든 매개변수가 *손실*을 어떻게 변경하는지 보고 싶습니다. 본질적으로, 저희는 가능할 때마다 분자에 $\partial f$를 유지하면서 연쇄 규칙을 적용하고 싶습니다!

더 명시적으로 말하면, 저희는 다음과 같이 쓸 수 있음에 유의하십시오.

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

연쇄 규칙의 이 적용은 저희가 $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \textrm{and} \; \frac{\partial f}{\partial w}$를 명시적으로 계산하도록 한다는 점에 유의하십시오. 다음 방정식도 포함하는 것을 막을 수 있는 것은 없습니다.

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

그리고 전체 네트워크의 *어떤* 노드를 변경할 때 $f$가 어떻게 변하는지 추적합니다. 이를 구현해 봅시다.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

저희가 입력에서 출력으로 앞으로(위의 첫 번째 코드 스니펫에서 했던 것처럼)가 아니라 $f$에서 입력 쪽으로 거꾸로 도함수를 계산한다는 사실이 이 알고리즘에 *역전파*라는 이름을 부여하는 것입니다. 두 단계가 있다는 점에 유의하십시오.
1. 함수의 값과 단일 단계 편도함수를 앞에서 뒤로 계산합니다. 위에서는 하지 않았지만, 이는 단일 *순방향 전달*로 결합될 수 있습니다.
2. $f$의 그래디언트를 뒤에서 앞으로 계산합니다. 저희는 이를 *역방향 전달*이라고 부릅니다.

이는 모든 딥러닝 알고리즘이 한 번의 전달에서 네트워크의 모든 가중치에 대한 손실의 그래디언트 계산을 허용하기 위해 구현하는 것 그 자체입니다. 저희가 그러한 분해를 가지고 있다는 것은 놀라운 사실입니다.

이를 어떻게 캡슐화하는지 보기 위해, 이 예제를 빠르게 살펴봅시다.

```{.python .input}
#@tab mxnet
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

저희가 위에서 한 모든 것은 `f.backwards()`를 호출함으로써 자동으로 수행될 수 있습니다.


## 헤시안
단변수 미적분과 마찬가지로, 그래디언트만 사용하는 것보다 함수에 대한 더 나은 근사를 어떻게 얻을 수 있는지 파악하기 위해 고차 도함수를 고려하는 것이 유용합니다.

여러 변수의 함수의 고차 도함수로 작업할 때 즉시 마주치는 한 가지 문제가 있는데, 그것은 그것들이 많은 수가 있다는 것입니다. 만약 $n$ 변수의 함수 $f(x_1, \ldots, x_n)$가 있다면, $n^{2}$개의 2계 도함수를 취할 수 있습니다. 즉, $i$와 $j$의 어떤 선택에 대해서도 다음과 같습니다.

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

이는 전통적으로 *헤시안*이라고 불리는 행렬로 조립됩니다.

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

이 행렬의 모든 항목이 독립적인 것은 아닙니다. 사실, *혼합 편도함수*(둘 이상의 변수에 대한 편도함수) 둘 다 존재하고 연속인 한, 어떤 $i$와 $j$에 대해서도 다음과 같이 말할 수 있음을 보일 수 있습니다.

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

이는 먼저 $x_i$ 방향으로 함수를 섭동하고, 그런 다음 $x_j$로 섭동하는 것을 고려한 후, 먼저 $x_j$를, 그런 다음 $x_i$를 섭동했을 때 무슨 일이 일어나는지의 결과와 비교함으로써 따릅니다. 이 두 순서 모두 $f$의 출력의 같은 최종 변화로 이어진다는 지식과 함께 말입니다.

단일 변수와 마찬가지로, 저희는 이러한 도함수를 사용하여 함수가 한 점 근처에서 어떻게 동작하는지에 대한 훨씬 더 나은 아이디어를 얻을 수 있습니다. 특히, 단일 변수에서 본 것처럼 점 $\mathbf{x}_0$ 근처에서 가장 잘 맞는 이차식을 찾는 데 이를 사용할 수 있습니다.

예제를 봅시다. $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$라고 가정해 봅시다. 이는 두 변수의 이차식의 일반적인 형태입니다. 함수의 값, 그래디언트, 그리고 헤시안 :eqref:`eq_hess_def`를 모두 0 지점에서 보면 다음과 같습니다.

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

저희는 다음과 같이 말함으로써 원래의 다항식을 되찾을 수 있습니다.

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

일반적으로, 만약 어떤 점 $\mathbf{x}_0$에서 이 전개를 계산했다면, 다음을 봅니다.

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

이는 어떤 차원의 입력에 대해서도 작동하며, 한 점에서 어떤 함수에 대해서도 가장 잘 근사하는 이차식을 제공합니다. 예를 들어, 다음 함수를 플롯해 봅시다.

$$
f(x, y) = xe^{-x^2-y^2}.
$$

그래디언트와 헤시안이 다음과 같음을 계산할 수 있습니다.
$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \textrm{and} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$

따라서, 약간의 대수로, $[-1,0]^\top$에서 근사하는 이차식이 다음과 같음을 봅니다.

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
#@tab mxnet
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), w.asnumpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

이는 :numref:`sec_gd`에서 논의된 뉴턴 알고리즘의 기초를 형성하는데, 여기서 저희는 가장 잘 맞는 이차식을 반복적으로 찾고 그 이차식을 정확히 최소화함으로써 수치적 최적화를 수행합니다.

## 약간의 행렬 미적분
행렬을 포함하는 함수의 도함수는 특히 좋은 것으로 밝혀집니다. 이 절은 표기상으로 무거워질 수 있으므로, 첫 번째 읽기에서는 건너뛸 수 있지만, 특히 행렬 연산이 딥러닝 응용에 얼마나 중심적인지를 고려할 때 일반적인 행렬 연산을 포함하는 함수의 도함수가 처음에 예상할 수 있는 것보다 종종 훨씬 더 깔끔하다는 것을 아는 것이 유용합니다.

예제로 시작합시다. 어떤 고정된 열 벡터 $\boldsymbol{\beta}$가 있다고 가정하고, 곱 함수 $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$를 취하여 $\mathbf{x}$를 변경할 때 내적이 어떻게 변하는지 이해하고 싶다고 가정해 봅시다.

ML에서 행렬 도함수로 작업할 때 유용한 약간의 표기법은 *분모 레이아웃 행렬 도함수*라고 하는데, 여기서 저희는 편도함수를 미분의 분모에 있는 어떤 벡터, 행렬, 또는 텐서의 모양으로 조립합니다. 이 경우, 다음과 같이 작성할 것입니다.

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

여기서 저희는 열 벡터 $\mathbf{x}$의 모양에 맞춥니다.

만약 함수를 성분으로 작성한다면 이것은 다음과 같습니다.

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

이제 예를 들어 $\beta_1$에 대해 편도함수를 취하면, 모든 것이 0이지만 첫 번째 항만 그렇지 않고, 이는 단지 $x_1$에 $\beta_1$를 곱한 것이므로, 다음을 얻습니다.

$$
\frac{df}{dx_1} = \beta_1,
$$

또는 더 일반적으로

$$
\frac{df}{dx_i} = \beta_i.
$$

이제 이를 행렬로 다시 조립하여 다음을 봅니다.

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

이는 이 절 전반에 걸쳐 자주 마주치게 될 행렬 미적분에 대한 몇 가지 요소를 보여줍니다.

* 첫째, 계산은 다소 관련될 것입니다.
* 둘째, 최종 결과는 중간 과정보다 훨씬 더 깔끔하며, 항상 단일 변수 경우와 유사하게 보일 것입니다. 이 경우, $\frac{d}{dx}(bx) = b$와 $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$가 모두 유사하다는 점에 유의하십시오.
* 셋째, 전치는 종종 어디에서도 나타나지 않는 것처럼 보일 수 있습니다. 이에 대한 핵심 이유는 저희가 분모의 모양에 맞추는 관례이며, 따라서 행렬을 곱할 때 원래 항의 모양으로 다시 맞추기 위해 전치를 취해야 할 것입니다.

직관을 계속 쌓기 위해, 약간 더 어려운 계산을 시도해 봅시다. 열 벡터 $\mathbf{x}$와 정사각 행렬 $A$가 있고 다음을 계산하고 싶다고 가정해 봅시다.

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

조작하기 쉬운 표기법으로 가기 위해, 아인슈타인 표기법을 사용하여 이 문제를 고려해 봅시다. 이 경우 함수를 다음과 같이 쓸 수 있습니다.

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

저희의 도함수를 계산하려면, 모든 $k$에 대해, 다음의 값이 무엇인지 이해해야 합니다.

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

곱 규칙에 의해, 이는 다음과 같습니다.

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

$\frac{dx_i}{dx_k}$와 같은 항의 경우, 이는 $i=k$일 때 1이고 그렇지 않으면 0이라는 것을 보기 어렵지 않습니다. 이는 $i$와 $k$가 다른 모든 항이 이 합에서 사라진다는 것을 의미하므로, 그 첫 번째 합에 남아 있는 유일한 항은 $i=k$인 것입니다. 같은 추론이 $j=k$를 필요로 하는 두 번째 항에도 성립합니다. 이는 다음을 제공합니다.

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

이제, 아인슈타인 표기법에서 인덱스의 이름은 임의적입니다($i$와 $j$가 다르다는 사실은 이 시점에서 이 계산에 중요하지 않으므로, 둘 다 $i$를 사용하도록 재인덱싱하여 다음을 볼 수 있습니다.

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

이제, 여기서 더 나아가기 위해 약간의 연습이 필요해지기 시작합니다. 이 결과를 행렬 연산의 관점에서 식별해 보겠습니다. $a_{ki} + a_{ik}$는 $\mathbf{A} + \mathbf{A}^\top$의 $k, i$번째 성분입니다. 이는 다음을 제공합니다.

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

유사하게, 이 항은 이제 행렬 $\mathbf{A} + \mathbf{A}^\top$와 벡터 $\mathbf{x}$의 곱이므로, 다음을 봅니다.

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

따라서, :eqref:`eq_mat_goal_1`에서 원하는 도함수의 $k$번째 항목이 오른쪽 벡터의 $k$번째 항목과 같다는 것을 봅니다. 따라서 두 개는 같습니다. 이는 다음을 산출합니다.

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

이는 마지막 것보다 상당히 더 많은 작업이 필요했지만, 최종 결과는 작습니다. 그것보다 더, 전통적인 단일 변수 도함수에 대한 다음 계산을 고려해 보십시오.

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$

동등하게 $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$. 다시, 저희는 단일 변수 결과처럼 보이지만 전치가 던져진 결과를 얻습니다.

이 시점에서, 패턴은 다소 의심스럽게 보여야 하므로, 그 이유를 알아내려고 해봅시다. 이런 식으로 행렬 도함수를 취할 때, 먼저 저희가 얻을 식이 또 다른 행렬 식이 될 것이라고 가정해 봅시다. 즉, 행렬과 그들의 전치의 곱과 합의 관점에서 그것을 쓸 수 있는 식입니다. 만약 그러한 식이 존재한다면, 그것은 모든 행렬에 대해 참이어야 할 것입니다. 특히, 그것은 $1 \times 1$ 행렬에 대해 참이어야 할 것이며, 이 경우 행렬 곱은 단지 숫자의 곱이고, 행렬 합은 단지 합이며, 전치는 전혀 아무 것도 하지 않습니다! 다시 말해, 저희가 얻는 어떤 식이든 *반드시* 단일 변수 식과 일치해야 합니다. 이는 약간의 연습으로, 종종 연관된 단일 변수 식이 어떻게 보여야 하는지 아는 것만으로 행렬 도함수를 추측할 수 있다는 것을 의미합니다!

이를 시도해 봅시다. $\mathbf{X}$가 $n \times m$ 행렬이고, $\mathbf{U}$가 $n \times r$이고 $\mathbf{V}$가 $r \times m$이라고 가정해 봅시다. 다음을 계산해 봅시다.

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

이 계산은 행렬 인수분해라고 불리는 분야에서 중요합니다. 그러나 저희에게는 단지 계산할 도함수일 뿐입니다. 이것이 $1\times1$ 행렬에 대해 어떻게 될지 상상해 봅시다. 그 경우, 저희는 식

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

을 얻습니다. 여기서 도함수는 다소 표준적입니다. 이를 행렬 식으로 다시 변환하려고 하면 다음을 얻습니다.

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

그러나, 이를 보면 잘 작동하지 않습니다. $\mathbf{X}$는 $n \times m$이고 $\mathbf{U}\mathbf{V}$도 그렇기 때문에, 행렬 $2(\mathbf{X} - \mathbf{U}\mathbf{V})$는 $n \times m$임을 떠올리십시오. 반면에 $\mathbf{U}$는 $n \times r$이고, 차원이 일치하지 않기 때문에 $n \times m$과 $n \times r$ 행렬을 곱할 수 없습니다!

저희는 $\mathbf{V}$와 같은 모양인 $\frac{d}{d\mathbf{V}}$를 얻고 싶은데, 이는 $r \times m$입니다. 그래서 어떻게든 $n \times m$ 행렬과 $n \times r$ 행렬을 취하고, 그들을 함께 (아마도 약간의 전치와 함께) 곱하여 $r \times m$을 얻어야 합니다. 저희는 $U^\top$를 $(\mathbf{X} - \mathbf{U}\mathbf{V})$로 곱함으로써 이를 할 수 있습니다. 따라서, :eqref:`eq_mat_goal_2`의 해를 다음과 같이 추측할 수 있습니다.

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

이것이 작동한다는 것을 보이기 위해, 자세한 계산을 제공하지 않으면 부주의할 것입니다. 만약 이미 이 경험 법칙이 작동한다고 믿는다면, 이 유도를 건너뛰어도 됩니다. 다음을 계산하려면

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

모든 $a$와 $b$에 대해 다음을 찾아야 합니다.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

$\frac{d}{dv_{ab}}$에 관한 한 $\mathbf{X}$와 $\mathbf{U}$의 모든 항목이 상수임을 떠올리면, 도함수를 합 안으로 밀어넣고, 제곱에 연쇄 규칙을 적용하여 다음을 얻을 수 있습니다.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

이전 유도에서처럼, $\frac{dv_{kj}}{dv_{ab}}$는 $k=a$이고 $j=b$인 경우에만 0이 아니라는 점에 주목할 수 있습니다. 만약 그 조건 중 어느 것도 성립하지 않는다면, 합의 항은 0이며, 저희는 자유롭게 그것을 버릴 수 있습니다. 저희는 다음을 봅니다.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

여기서 중요한 미묘함은 $k=a$ 요건이 내부 항 안에서 합산되는 $k$가 더미 변수이기 때문에 내부 합 안에서 발생하지 않는다는 것입니다. 표기상으로 더 깔끔한 예의 경우, 왜 다음과 같은지 고려해 보십시오.

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

이 시점에서, 저희는 합의 성분을 식별하기 시작할 수 있습니다. 첫째,

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

따라서 합의 내부에 있는 전체 식은 다음과 같습니다.

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

이는 이제 저희의 도함수를 다음과 같이 쓸 수 있음을 의미합니다.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

저희는 이것이 행렬의 $a, b$ 원소처럼 보이기를 원하므로, 이전 예제에서와 같은 기법을 사용하여 행렬 식에 도달할 수 있습니다. 이는 $u_{ia}$의 인덱스 순서를 교환해야 한다는 것을 의미합니다. 만약 $u_{ia} = [\mathbf{U}^\top]_{ai}$임을 알아챈다면, 다음과 같이 쓸 수 있습니다.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

이는 행렬 곱이며, 따라서 다음과 같이 결론 내릴 수 있습니다.

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

따라서 :eqref:`eq_mat_goal_2`의 해를 다음과 같이 쓸 수 있습니다.

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

이는 저희가 위에서 추측한 해와 일치합니다!

이 시점에서 "왜 그냥 제가 배운 모든 미적분 규칙의 행렬 버전을 적어둘 수 없나요? 이것이 여전히 기계적임은 분명합니다. 왜 그냥 끝내지 않나요!"라고 묻는 것이 합리적입니다. 그리고 실제로 그러한 규칙이 있고 :cite:`Petersen.Pedersen.ea.2008`은 훌륭한 요약을 제공합니다. 그러나, 단일 값과 비교하여 행렬 연산이 결합될 수 있는 방법이 무수히 많기 때문에, 단일 변수보다 훨씬 더 많은 행렬 도함수 규칙이 있습니다. 종종 인덱스로 작업하거나 적절할 때 자동 미분에 맡기는 것이 가장 좋습니다.

## 요약

* 더 높은 차원에서, 저희는 1차원의 도함수와 같은 목적을 수행하는 그래디언트를 정의할 수 있습니다. 이는 다변수 함수가 입력에 임의로 작은 변경을 했을 때 어떻게 변하는지 볼 수 있게 해줍니다.
* 역전파 알고리즘은 많은 편도함수의 효율적인 계산을 허용하기 위해 다변수 연쇄 규칙을 조직하는 방법으로 볼 수 있습니다.
* 행렬 미적분은 행렬 식의 도함수를 간결한 방식으로 쓸 수 있게 해줍니다.

## 연습문제
1. 열 벡터 $\boldsymbol{\beta}$가 주어졌을 때, $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$와 $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$ 둘 다의 도함수를 계산하십시오. 왜 같은 답을 얻습니까?
2. $\mathbf{v}$를 $n$ 차원 벡터라고 합시다. $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$는 무엇입니까?
3. $L(x, y) = \log(e^x + e^y)$라고 합시다. 그래디언트를 계산하십시오. 그래디언트의 성분의 합은 무엇입니까?
4. $f(x, y) = x^2y + xy^2$라고 합시다. 유일한 임계점이 $(0,0)$임을 보이십시오. $f(x, x)$를 고려함으로써, $(0,0)$이 최댓값인지, 최솟값인지, 아니면 둘 다 아닌지 결정하십시오.
5. 함수 $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$를 최소화하고 있다고 가정해 봅시다. $g$와 $h$의 관점에서 $\nabla f = 0$ 조건을 어떻게 기하학적으로 해석할 수 있습니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab:
