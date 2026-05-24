```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 자동 미분
:label:`sec_autograd`

:numref:`sec_calculus`에서 도함수를 계산하는 것이
심층 신경망을 훈련시키는 데 사용할 모든 최적화 알고리즘의
중요한 단계임을 떠올리십시오.
계산은 간단하지만,
수작업으로 풀어내는 것은 지루하고 오류가 발생하기 쉬우며,
이러한 문제들은 모델이 더 복잡해질수록
커지기만 합니다.

다행히도 모든 현대 딥러닝 프레임워크는
*자동 미분*(흔히 *autograd*로 줄여 부름)을 제공함으로써
이 작업을 저희 손에서 가져가 줍니다.
저희가 데이터를 각 연속적인 함수에 통과시키면,
프레임워크는 각 값이 다른 값에 어떻게 의존하는지를 추적하는
*계산 그래프*를 구축합니다.
도함수를 계산하기 위해,
자동 미분은 이 그래프를 거꾸로 통과하면서
연쇄 법칙을 적용합니다.
이러한 방식으로 연쇄 법칙을 적용하는 계산 알고리즘은
*역전파*라고 부릅니다.

지난 10년 동안 autograd 라이브러리가
뜨거운 관심사가 되었지만,
이들은 오랜 역사를 가지고 있습니다.
실제로 autograd에 대한 가장 초기의 참고 문헌은
반세기 전으로 거슬러 올라갑니다 :cite:`Wengert.1964`.
현대 역전파의 핵심 아이디어는
1980년 박사 학위 논문 :cite:`Speelpenning.1980`으로 거슬러 올라가며
1980년대 후반에 더 발전되었습니다 :cite:`Griewank.1989`.
역전파가 그래디언트를 계산하기 위한 기본 방법이 되었지만,
유일한 선택지는 아닙니다.
예를 들어, Julia 프로그래밍 언어는
순방향 전파를 사용합니다 :cite:`Revels.Lubin.Papamarkou.2016`.
방법들을 살펴보기에 앞서,
먼저 autograd 패키지를 마스터해 봅시다.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## 간단한 함수

저희가 (**열 벡터 $\mathbf{x}$에 대해
함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$를
미분하는 것에**) 관심이 있다고 가정합시다.
시작하기 위해, `x`에 초기값을 할당합니다.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**$\mathbf{x}$에 대한
$y$의 그래디언트를 계산하기 전에,
이를 저장할 곳이 필요합니다.**]
일반적으로, 저희는 도함수를 취할 때마다
새로운 메모리를 할당하는 것을 피합니다.
딥러닝은 동일한 파라미터에 대해
도함수를 매우 여러 번 연속적으로 계산해야 하며,
메모리가 부족해질 위험이 있기 때문입니다.
벡터 $\mathbf{x}$에 대한 스칼라값 함수의 그래디언트는
$\mathbf{x}$와 같은 모양의 벡터값을 가진다는 점에 유의하십시오.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**이제 `x`의 함수를 계산하고 결과를 `y`에 할당합니다.**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
이제 `y`의 `backward` 메서드를 호출하여
[**`x`에 대한 `y`의 그래디언트를 취할 수 있습니다**].
다음으로, `x`의 `grad` 속성을 통해
그래디언트에 접근할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이제 `y`의 `backward` 메서드를 호출하여
[**`x`에 대한 `y`의 그래디언트를 취할 수 있습니다**].
다음으로, `x`의 `grad` 속성을 통해
그래디언트에 접근할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
이제 `gradient` 메서드를 호출하여
[**`x`에 대한 `y`의 그래디언트를 계산할 수 있습니다**].
:end_tab:

:begin_tab:`jax`
이제 `grad` 변환을 통과시켜
[**`x`에 대한 `y`의 그래디언트를 취할 수 있습니다**].
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

(**저희는 이미 함수 $y = 2\mathbf{x}^{\top}\mathbf{x}$의
$\mathbf{x}$에 대한 그래디언트가 $4\mathbf{x}$여야 한다는 것을 알고 있습니다.**)
이제 자동 그래디언트 계산과
예상 결과가 동일한지 확인할 수 있습니다.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**이제 `x`의 다른 함수를 계산하고
그 그래디언트를 취해 봅시다.**]
MXNet은 새로운 그래디언트를 기록할 때마다
그래디언트 버퍼를 리셋한다는 점에 유의하십시오.
:end_tab:

:begin_tab:`pytorch`
[**이제 `x`의 다른 함수를 계산하고
그 그래디언트를 취해 봅시다.**]
PyTorch는 새로운 그래디언트를 기록할 때
그래디언트 버퍼를 자동으로 리셋하지 않는다는 점에 유의하십시오.
대신, 새로운 그래디언트가
이미 저장된 그래디언트에 더해집니다.
이러한 동작은 여러 목적 함수의 합을
최적화하고자 할 때 유용합니다.
그래디언트 버퍼를 리셋하려면,
다음과 같이 `x.grad.zero_()`를 호출할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
[**이제 `x`의 다른 함수를 계산하고
그 그래디언트를 취해 봅시다.**]
TensorFlow는 새로운 그래디언트를 기록할 때마다
그래디언트 버퍼를 리셋한다는 점에 유의하십시오.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## 비스칼라 변수에 대한 역방향

`y`가 벡터일 때,
벡터 `x`에 대한 `y`의 도함수의
가장 자연스러운 표현은
`y`의 각 성분에 대한 `x`의 각 성분에 대한
편미분을 포함하는
*자코비안*이라고 불리는 행렬입니다.
마찬가지로, 고차의 `y`와 `x`의 경우,
미분의 결과는 훨씬 더 고차의 텐서가 될 수 있습니다.

자코비안은 일부 고급 머신러닝 기법에서
나타나기는 하지만,
더 흔하게는 전체 벡터 `x`에 대한
`y`의 각 성분의 그래디언트를 합산해
`x`와 같은 모양의 벡터를 얻고자 합니다.
예를 들어, 저희는 종종 훈련 예제의 *배치* 중
각 예제에 대해 개별적으로 계산된
손실 함수의 값을 나타내는 벡터를 가집니다.
여기서는 단지 (**각 예제에 대해 개별적으로 계산된
그래디언트를 합산**)하고자 합니다.

:begin_tab:`mxnet`
MXNet은 그래디언트를 계산하기 전에 합산함으로써
모든 텐서를 스칼라로 축소하여
이 문제를 처리합니다.
다시 말해, 자코비안
$\partial_{\mathbf{x}} \mathbf{y}$를 반환하는 대신,
합의 그래디언트
$\partial_{\mathbf{x}} \sum_i y_i$를 반환합니다.
:end_tab:

:begin_tab:`pytorch`
딥러닝 프레임워크들이 비스칼라 텐서의 그래디언트를
해석하는 방식이 다르기 때문에,
PyTorch는 혼동을 피하기 위해 몇 가지 단계를 취합니다.
비스칼라에 대해 `backward`를 호출하면
객체를 스칼라로 어떻게 축소할지 PyTorch에게 알려주지 않는 한
오류가 발생합니다.
보다 형식적으로, 저희는 `backward`가
$\partial_{\mathbf{x}} \mathbf{y}$ 대신
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$를 계산하도록
어떤 벡터 $\mathbf{v}$를 제공해야 합니다.
다음 부분은 혼란스러울 수 있지만,
나중에 명확해질 이유로,
이 인수($\mathbf{v}$를 나타내는)는 `gradient`라고 이름 붙여졌습니다.
보다 자세한 설명은, Yang Zhang의
[Medium 포스트](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29)를 참조하십시오.
:end_tab:

:begin_tab:`tensorflow`
기본적으로, TensorFlow는 합의 그래디언트를 반환합니다.
다시 말해, 자코비안
$\partial_{\mathbf{x}} \mathbf{y}$를 반환하는 대신,
합의 그래디언트
$\partial_{\mathbf{x}} \sum_i y_i$를 반환합니다.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## 계산 분리하기

때때로, 저희는 [**일부 계산을
기록된 계산 그래프 바깥으로 이동**]시키고자 합니다.
예를 들어, 입력을 사용하여
그래디언트를 계산하고 싶지 않은
보조 중간 항들을 생성한다고 합시다.
이 경우, 해당 계산 그래프를
최종 결과로부터 *분리*해야 합니다.
다음의 장난감 예제가 이를 더 명확하게 해 줍니다.
`z = x * y`이고 `y = x * x`이지만
`y`를 통해 전달되는 영향이 아닌
`z`에 대한 `x`의 *직접적인* 영향에 집중하고 싶다고 가정합시다.
이 경우, `y`와 같은 값을 가지지만
그 *출처*(어떻게 생성되었는지)가 지워진
새로운 변수 `u`를 생성할 수 있습니다.
따라서 `u`는 그래프에서 조상이 없고
그래디언트는 `u`를 통해 `x`로 흐르지 않습니다.
예를 들어, `z = x * u`의 그래디언트를 취하면
(`z = x * x * x`이므로 예상했을 수 있는 `3 * x * x`가 아니라)
결과 `u`를 얻게 됩니다.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

이 절차가 `z`로 이어지는 그래프에서
`y`의 조상들을 분리하지만,
`y`로 이어지는 계산 그래프는
지속되므로 `x`에 대한 `y`의 그래디언트를
계산할 수 있다는 점에 유의하십시오.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## 그래디언트와 Python 제어 흐름

지금까지 저희는 입력에서 출력까지의 경로가
`z = x * x * x`와 같은 함수를 통해 잘 정의된 경우를 살펴봤습니다.
프로그래밍은 결과를 계산하는 방식에 훨씬 더 많은 자유를 제공합니다.
예를 들어, 보조 변수에 의존하게 하거나
중간 결과에 따라 선택을 조건부로 할 수 있습니다.
자동 미분을 사용하는 한 가지 이점은
(**함수의 계산 그래프를 구축하는 데
Python 제어 흐름의 미로(예: 조건문, 루프, 임의의 함수 호출)를 통과해야 했더라도**)
[**여전히**]
(**결과 변수의 그래디언트를 계산할 수 있다는 것입니다.**)
이를 설명하기 위해, `while` 루프의 반복 횟수와
`if` 문의 평가가 모두 입력 `a`의 값에 의존하는
다음 코드 조각을 살펴봅시다.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

아래에서, 저희는 입력으로 무작위 값을 전달하면서 이 함수를 호출합니다.
입력이 무작위 변수이기 때문에,
계산 그래프가 어떤 형태를 띨지 알 수 없습니다.
그러나, 특정 입력에 대해 `f(a)`를 실행할 때마다,
저희는 특정 계산 그래프를 실현하며
이후에 `backward`를 실행할 수 있습니다.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

저희 함수 `f`가 시연 목적으로 다소 작위적이긴 하지만,
입력에 대한 의존성은 매우 단순합니다.
이는 구간별로 정의된 스케일을 갖는
`a`의 *선형* 함수입니다.
따라서, `f(a) / a`는 상수 항목들의 벡터이며,
나아가, `f(a) / a`는 `a`에 대한 `f(a)`의 그래디언트와
일치해야 합니다.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

동적 제어 흐름은 딥러닝에서 매우 흔합니다.
예를 들어, 텍스트를 처리할 때, 계산 그래프는
입력의 길이에 의존합니다.
이러한 경우, 그래디언트를 *사전에* 계산하는 것이 불가능하기 때문에
자동 미분은 통계적 모델링에 필수적이 됩니다.

## 논의

이제 여러분은 자동 미분의 위력을 맛보셨습니다.
도함수를 자동으로 그리고 효율적으로
계산하기 위한 라이브러리의 개발은
딥러닝 실무자들에게 엄청난 생산성 향상을 가져다주었고,
그들이 덜 단조로운 작업에 집중할 수 있도록 해방시켰습니다.
나아가, autograd는 펜과 종이로 그래디언트를 계산하는 것이
엄청나게 시간이 많이 걸릴 수 있는 거대한 모델을 설계할 수 있게 해 줍니다.
흥미롭게도, 저희가 (통계적 의미에서)
모델을 *최적화*하기 위해 autograd를 사용하는 한편,
autograd 라이브러리 자체의 (계산적 의미에서) *최적화*는
프레임워크 설계자들에게 매우 중요한
풍부한 주제입니다.
여기서, 컴파일러와 그래프 조작의 도구들이
가장 신속하고 메모리 효율적인 방식으로
결과를 계산하기 위해 활용됩니다.

지금은, 이 기본 사항들을 기억하려고 노력해 보세요. (i) 도함수를 원하는 변수들에 그래디언트를 부착하고, (ii) 목표 값의 계산을 기록하고, (iii) 역전파 함수를 실행하고, (iv) 결과 그래디언트에 접근합니다.


## 연습문제

1. 이계도함수는 왜 일계도함수보다 계산 비용이 훨씬 더 비쌉니까?
1. 역전파를 위한 함수를 실행한 후, 즉시 다시 실행해 보고 무슨 일이 일어나는지 보세요. 조사해 보세요.
1. `a`에 대한 `d`의 도함수를 계산하는 제어 흐름 예제에서, 변수 `a`를 무작위 벡터나 행렬로 바꾸면 어떤 일이 일어날까요? 이 시점에서, 계산 `f(a)`의 결과는 더 이상 스칼라가 아닙니다. 결과에 어떤 일이 일어날까요? 이를 어떻게 분석합니까?
1. $f(x) = \sin(x)$라고 합시다. $f$와 그 도함수 $f'$의 그래프를 그리세요. $f'(x) = \cos(x)$라는 사실을 이용하지 말고, 자동 미분을 사용하여 결과를 얻으세요.
1. $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$라고 합시다. $x$에서 $f(x)$까지의 결과를 추적하는 의존성 그래프를 적어 보세요.
1. 연쇄 법칙을 사용하여 앞서 언급한 함수의 도함수 $\frac{df}{dx}$를 계산하고, 각 항을 앞서 구성한 의존성 그래프에 배치하세요.
1. 그래프와 중간 도함수 결과가 주어지면, 그래디언트를 계산할 때 여러 가지 선택지가 있습니다. $x$에서 $f$까지 한 번, $f$에서 $x$로 다시 추적하여 한 번, 결과를 평가해 보세요. $x$에서 $f$까지의 경로는 일반적으로 *순방향 미분*으로 알려져 있는 반면, $f$에서 $x$까지의 경로는 역방향 미분으로 알려져 있습니다.
1. 언제 순방향 미분을 사용하고, 언제 역방향 미분을 사용하고자 할 수 있을까요? 힌트: 필요한 중간 데이터의 양, 단계들을 병렬화하는 능력, 관련된 행렬과 벡터의 크기를 고려해 보세요.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17970)
:end_tab: