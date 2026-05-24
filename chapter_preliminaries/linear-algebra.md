```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형대수
:label:`sec_linear-algebra`

이제 저희는 데이터셋을 텐서로 로드하고
기본적인 수학 연산으로 이러한 텐서를
조작할 수 있습니다.
정교한 모델을 구축하기 시작하기 위해서,
선형대수의 몇 가지 도구도 필요할 것입니다.
이 절은 스칼라 산술부터 시작해서
행렬 곱셈까지 단계적으로 올라가며,
가장 핵심적인 개념들에 대한
온화한 소개를 제공합니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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

## 스칼라


일상적인 수학의 대부분은
한 번에 하나의 숫자를 조작하는 것으로
이루어져 있습니다.
형식적으로, 저희는 이러한 값을 *스칼라*라고 부릅니다.
예를 들어, Palo Alto의 기온은
화씨 $72$도로 화창합니다.
기온을 섭씨로 변환하고 싶다면,
$f$를 $72$로 설정하여
$c = \frac{5}{9}(f - 32)$라는 식을 계산하면 됩니다.
이 방정식에서, $5$, $9$, $32$의 값들은
상수 스칼라입니다.
변수 $c$와 $f$는
일반적으로 미지의 스칼라를 나타냅니다.

저희는 스칼라를
일반적인 소문자(예: $x$, $y$, $z$)로 표시하고
모든 (연속) *실수값* 스칼라의 공간을
$\mathbb{R}$로 표시합니다.
편의를 위해, 저희는 *공간*의 엄밀한 정의는 생략합니다.
단지 식 $x \in \mathbb{R}$이
$x$가 실수값 스칼라임을 형식적으로 말하는 방법이라는 것만 기억하십시오.
기호 $\in$("in"으로 발음)은
집합에서의 멤버십을 나타냅니다.
예를 들어, $x, y \in \{0, 1\}$은
$x$와 $y$가 $0$ 또는 $1$의 값만 가질 수 있는
변수임을 나타냅니다.

(**스칼라는 단 하나의 원소만 포함하는
텐서로 구현됩니다.**)
아래에서, 두 개의 스칼라를 할당하고
익숙한 덧셈, 곱셈, 나눗셈,
거듭제곱 연산을 수행합니다.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## 벡터

현재 목적상, [**벡터는 스칼라의 고정 길이 배열로 생각할 수 있습니다.**]
코드 상의 대응체와 마찬가지로,
저희는 이러한 스칼라들을 벡터의 *원소*라고 부릅니다
(동의어로 *항목*과 *성분*이 있습니다).
벡터가 실제 데이터셋의 예제를 나타낼 때,
그 값들은 어떤 실제 의미를 가집니다.
예를 들어, 대출의 채무 불이행 위험을 예측하는 모델을 훈련시킨다면,
각 신청자를 그들의 소득, 고용 기간,
이전 채무 불이행 횟수와 같은 양에 대응되는
성분을 가진 벡터와 연관시킬 수 있습니다.
심장 마비의 위험을 연구한다면,
각 벡터는 환자를 나타낼 수 있고
그 성분은 가장 최근의 활력 징후, 콜레스테롤 수치,
하루 운동 시간 등에 대응될 수 있습니다.
저희는 벡터를 굵은 소문자로
(예: $\mathbf{x}$, $\mathbf{y}$, $\mathbf{z}$)
표시합니다.

벡터는 $1$차 텐서로 구현됩니다.
일반적으로, 이러한 텐서는 메모리 제한에 따라
임의의 길이를 가질 수 있습니다. 주의: Python에서는 대부분의 프로그래밍 언어와 마찬가지로 벡터 인덱스가 $0$에서 시작하며, 이는 *0 기반 인덱싱*이라고도 알려져 있습니다. 반면 선형대수에서 첨자는 $1$에서 시작합니다(1 기반 인덱싱).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

저희는 첨자를 사용하여 벡터의 원소를 참조할 수 있습니다.
예를 들어, $x_2$는 $\mathbf{x}$의 두 번째 원소를 나타냅니다.
$x_2$가 스칼라이기 때문에, 굵게 표시하지 않습니다.
기본적으로, 저희는 벡터를
원소들을 수직으로 쌓아 시각화합니다.

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

여기서 $x_1, \ldots, x_n$은 벡터의 원소입니다.
나중에, 저희는 이러한 *열 벡터*와
원소들이 수평으로 쌓인 *행 벡터*를 구별할 것입니다.
[**저희는 인덱싱을 통해 텐서의 원소에 접근**]한다는 것을 떠올리십시오.

```{.python .input}
%%tab all
x[2]
```

벡터가 $n$개의 원소를 포함함을 나타내기 위해,
저희는 $\mathbf{x} \in \mathbb{R}^n$이라고 씁니다.
형식적으로, 저희는 $n$을 벡터의 *차원성*이라고 부릅니다.
[**코드에서, 이는 텐서의 길이에 해당**]하며,
Python의 내장 `len` 함수를 통해 접근할 수 있습니다.

```{.python .input}
%%tab all
len(x)
```

저희는 `shape` 속성을 통해서도 길이에 접근할 수 있습니다.
모양은 각 축을 따른 텐서의 길이를 나타내는 튜플입니다.
(**단 하나의 축을 가진 텐서는 단 하나의 원소를 가진 모양을 가집니다.**)

```{.python .input}
%%tab all
x.shape
```

종종, "차원"이라는 단어가 축의 개수와
특정 축을 따른 길이 모두를 의미하는 것으로
과부하됩니다.
이러한 혼란을 피하기 위해,
저희는 *차수*를 사용하여 축의 개수를 가리키고
*차원성*은 오직 성분의 개수를 가리키는 데에만
사용합니다.


## 행렬

스칼라가 $0$차 텐서이고
벡터가 $1$차 텐서인 것처럼,
행렬은 $2$차 텐서입니다.
저희는 행렬을 굵은 대문자로
(예: $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{Z}$)
표시하고, 코드에서는 두 개의 축을 가진 텐서로 표현합니다.
식 $\mathbf{A} \in \mathbb{R}^{m \times n}$은
행렬 $\mathbf{A}$가 $m$행과 $n$열로 배열된
$m \times n$개의 실수값 스칼라를
포함함을 나타냅니다.
$m = n$일 때, 행렬이 *정사각*이라고 합니다.
시각적으로, 저희는 어떤 행렬이든 테이블로 표현할 수 있습니다.
개별 원소를 참조하기 위해,
저희는 행과 열 인덱스 모두를 첨자로 표시합니다. 예를 들어,
$a_{ij}$는 $\mathbf{A}$의 $i$번째 행과 $j$번째 열에 속하는 값입니다.

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


코드에서, 저희는 행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$을
모양이 ($m$, $n$)인 $2$차 텐서로 표현합니다.
[**원하는 모양을 `reshape`에 전달하여
적절한 크기의 모든 $m \times n$ 텐서를
$m \times n$ 행렬로 변환할 수 있습니다**].

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

때때로 저희는 축을 뒤집고 싶습니다.
행렬의 행과 열을 교환하면,
결과는 그것의 *전치*라고 부릅니다.
형식적으로, 저희는 행렬 $\mathbf{A}$의 전치를
$\mathbf{A}^\top$로 표시하고 $\mathbf{B} = \mathbf{A}^\top$이면,
모든 $i$와 $j$에 대해 $b_{ij} = a_{ji}$입니다.
따라서, $m \times n$ 행렬의 전치는
$n \times m$ 행렬입니다.

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

코드에서, 다음과 같이 어떤 (**행렬의 전치**)에든 접근할 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**대칭 행렬은 자신의 전치와 같은
정사각 행렬의 부분 집합입니다.
$\mathbf{A} = \mathbf{A}^\top$.**]
다음 행렬은 대칭입니다.

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

행렬은 데이터셋을 표현하는 데 유용합니다.
일반적으로, 행은 개별 레코드에 해당하고
열은 서로 다른 속성에 해당합니다.



## 텐서

여러분이 머신러닝 여정에서 스칼라, 벡터, 행렬만으로도
멀리 나아갈 수 있지만,
결국에는 더 고차의 [**텐서**]를
다뤄야 할 수도 있습니다.
텐서는 (**$n$차 배열로의 확장을 설명하는
일반적인 방법을 제공합니다.**)
저희가 *텐서 클래스*의 소프트웨어 객체들을 "텐서"라고 부르는 것은
정확히 그것들도 임의의 수의 축을 가질 수 있기 때문입니다.
수학적 객체와 그것의 코드 상 구현 모두에
*텐서*라는 단어를 사용하는 것이 혼란스러울 수 있지만,
저희의 의미는 보통 문맥에서 명확할 것입니다.
저희는 일반 텐서를
특별한 글꼴의 대문자로
(예: $\mathsf{X}$, $\mathsf{Y}$, $\mathsf{Z}$)
표시하고 그것들의 인덱싱 메커니즘
(예: $x_{ijk}$와 $[\mathsf{X}]_{1, 2i-1, 3}$)은
행렬의 그것으로부터 자연스럽게 따라옵니다.

텐서는 저희가 이미지를 다루기 시작할 때
더욱 중요해질 것입니다.
각 이미지는 높이, 너비, *채널*에 대응되는 축을 가진
$3$차 텐서로 도착합니다.
각 공간적 위치에서, 각 색상(빨강, 초록, 파랑)의
강도가 채널을 따라 쌓입니다.
나아가, 이미지의 집합은
코드에서 $4$차 텐서로 표현되며,
서로 다른 이미지가 첫 번째 축을 따라 인덱싱됩니다.
더 고차의 텐서는 벡터와 행렬이 그러했던 것처럼,
모양 성분의 수를 늘림으로써 구성됩니다.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## 텐서 산술의 기본 속성

스칼라, 벡터, 행렬,
그리고 더 고차의 텐서는
모두 몇 가지 유용한 속성을 가집니다.
예를 들어, 원소별 연산은
피연산자와 같은 모양을 가진
출력을 생성합니다.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

[**두 행렬의 원소별 곱은
그들의 *아다마르 곱*이라고 부릅니다***] ($\odot$로 표시).
두 행렬
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$의 아다마르 곱의
항목들을 구체적으로 적을 수 있습니다.



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**스칼라와 텐서를 더하거나 곱하면**] 원본 텐서와
같은 모양의 결과가 생성됩니다.
여기서, 텐서의 각 원소가 스칼라에 더해지(거나 곱해)집니다.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## 축소
:label:`subsec_lin-alg-reduction`

저희는 종종 [**텐서 원소들의 합을 계산**]하고 싶어 합니다.
길이가 $n$인 벡터 $\mathbf{x}$의 원소들의 합을 표현하기 위해,
저희는 $\sum_{i=1}^n x_i$라고 씁니다. 이를 위한 간단한 함수가 있습니다.

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

[**임의의 모양의 텐서의 원소들에 대한 합**]을 표현하기 위해,
저희는 단순히 모든 축에 대해 합산합니다.
예를 들어, $m \times n$ 행렬 $\mathbf{A}$의 원소들의 합은
$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$로 쓸 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

기본적으로, 합 함수를 호출하면
텐서를 그 모든 축을 따라 *축소*하여,
결국 스칼라를 생성합니다.
저희의 라이브러리들은 또한 [**텐서가 축소되어야 하는
축을 지정**]할 수 있게 해 줍니다.
행(축 0)을 따라 모든 원소에 대해 합산하기 위해,
저희는 `sum`에서 `axis=0`을 지정합니다.
입력 행렬이 축 0을 따라 축소되어
출력 벡터를 생성하기 때문에,
이 축은 출력의 모양에서 누락됩니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

`axis=1`을 지정하면 모든 열의 원소들을 합산하여 열 차원(축 1)을 축소합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

행과 열 모두에 대해 합산을 통해 행렬을 축소하는 것은
행렬의 모든 원소를 합산하는 것과 동등합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**관련된 양은 *평균*이며, *average*라고도 불립니다.**]
저희는 합을 전체 원소 수로 나눠서 평균을 계산합니다.
평균을 계산하는 것이 매우 일반적이기 때문에,
이는 `sum`과 유사하게 작동하는
전용 라이브러리 함수를 갖습니다.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

마찬가지로, 평균을 계산하는 함수도
특정 축을 따라 텐서를 축소할 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## 비축소 합
:label:`subsec_lin-alg-non-reduction`

때때로 합이나 평균을 계산하는 함수를 호출할 때
[**축의 수를 변경하지 않고 유지**]하는 것이 유용할 수 있습니다.
이는 브로드캐스트 메커니즘을 사용하고자 할 때 중요합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

예를 들어, `sum_A`가 각 행을 합산한 후에도 두 개의 축을 유지하기 때문에,
저희는 (**브로드캐스팅으로 `A`를 `sum_A`로 나누어**)
각 행의 합이 $1$이 되는 행렬을 생성할 수 있습니다.

```{.python .input}
%%tab all
A / sum_A
```

[**어떤 축을 따라 `A`의 원소들의 누적 합**]을 계산하고 싶다면,
예를 들어 `axis=0`(행 단위로), `cumsum` 함수를 호출할 수 있습니다.
설계상, 이 함수는 어떤 축을 따라서도 입력 텐서를 축소하지 않습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## 내적

지금까지, 저희는 원소별 연산, 합, 평균만 수행했습니다.
그리고 만약 이것이 저희가 할 수 있는 전부였다면, 선형대수는
별도의 절을 가질 만한 가치가 없을 것입니다.
다행히도, 여기서부터 더 흥미로워집니다.
가장 기본적인 연산 중 하나는 내적입니다.
두 벡터 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$가 주어지면,
그것들의 *내적* $\mathbf{x}^\top \mathbf{y}$(*inner product*, $\langle \mathbf{x}, \mathbf{y}  \rangle$로도 알려진)은
같은 위치에 있는 원소들의 곱에 대한 합입니다.
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~두 벡터의 *내적*은 같은 위치에 있는 원소들의 곱에 대한 합입니다~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

동등하게, (**원소별 곱셈을 수행한 후 합을 수행하여
두 벡터의 내적을 계산할 수 있습니다.**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

내적은 광범위한 맥락에서 유용합니다.
예를 들어, 벡터 $\mathbf{x}  \in \mathbb{R}^n$로 표시되는 어떤 값의 집합과,
$\mathbf{w} \in \mathbb{R}^n$로 표시되는 가중치의 집합이 주어지면,
가중치 $\mathbf{w}$에 따른 $\mathbf{x}$의 값들의
가중합은 내적 $\mathbf{x}^\top \mathbf{w}$로
표현될 수 있습니다.
가중치가 음이 아니고
합이 $1$이 될 때, 즉, $\left(\sum_{i=1}^{n} {w_i} = 1\right)$이면,
내적은 *가중 평균*을 표현합니다.
두 벡터를 단위 길이를 갖도록 정규화한 후에,
내적은 그들 사이 각도의 코사인을 표현합니다.
이 절의 후반부에서, 저희는 이 *길이* 개념을 형식적으로 소개할 것입니다.


## 행렬-벡터 곱

이제 내적을 계산하는 방법을 알고 있으므로,
$m \times n$ 행렬 $\mathbf{A}$와
$n$차원 벡터 $\mathbf{x}$ 사이의
*곱*을 이해하기 시작할 수 있습니다.
시작하기 위해, 저희는 행렬을
그것의 행 벡터의 관점에서 시각화합니다.

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

여기서 각 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$은
행렬 $\mathbf{A}$의 $i$번째 행을 나타내는
행 벡터입니다.

[**행렬-벡터 곱 $\mathbf{A}\mathbf{x}$는
단순히 길이가 $m$인 열 벡터이며,
그것의 $i$번째 원소는 내적
$\mathbf{a}^\top_i \mathbf{x}$입니다.**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

저희는 행렬
$\mathbf{A}\in \mathbb{R}^{m \times n}$과의 곱셈을
$\mathbb{R}^{n}$에서 $\mathbb{R}^{m}$으로 벡터를 사영하는
변환으로 생각할 수 있습니다.
이러한 변환은 놀랍도록 유용합니다.
예를 들어, 회전을
특정 정사각 행렬과의 곱셈으로 표현할 수 있습니다.
행렬-벡터 곱은 또한 이전 층의 출력이 주어졌을 때
신경망의 각 층의 출력을 계산하는 데 관여하는
핵심 계산을 설명합니다.

:begin_tab:`mxnet`
코드에서 행렬-벡터 곱을 표현하기 위해,
저희는 동일한 `dot` 함수를 사용합니다.
연산은 인수의 유형에 기반하여
추론됩니다.
`A`의 열 차원(축 1을 따른 그것의 길이)이
`x`의 차원(그것의 길이)과 동일해야 한다는 점에
유의하십시오.
:end_tab:

:begin_tab:`pytorch`
코드에서 행렬-벡터 곱을 표현하기 위해,
저희는 `mv` 함수를 사용합니다.
`A`의 열 차원(축 1을 따른 그것의 길이)이
`x`의 차원(그것의 길이)과 동일해야 한다는 점에
유의하십시오.
Python에는 (인수에 따라)
행렬-벡터와 행렬-행렬 곱을 모두 실행할 수 있는
편리한 연산자 `@`가 있습니다.
따라서 저희는 `A@x`로 쓸 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
코드에서 행렬-벡터 곱을 표현하기 위해,
저희는 `matvec` 함수를 사용합니다.
`A`의 열 차원(축 1을 따른 그것의 길이)이
`x`의 차원(그것의 길이)과 동일해야 한다는 점에
유의하십시오.
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## 행렬-행렬 곱셈

내적과 행렬-벡터 곱에 익숙해지면,
*행렬-행렬 곱셈*은 간단할 것입니다.

저희가 두 개의 행렬
$\mathbf{A} \in \mathbb{R}^{n \times k}$와
$\mathbf{B} \in \mathbb{R}^{k \times m}$을 가지고 있다고 합시다.

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


$\mathbf{a}^\top_{i} \in \mathbb{R}^k$가
행렬 $\mathbf{A}$의 $i$번째 행을 나타내는 행 벡터이고
$\mathbf{b}_{j} \in \mathbb{R}^k$가
행렬 $\mathbf{B}$의 $j$번째 열에서 온
열 벡터라고 합시다.

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


행렬 곱 $\mathbf{C} \in \mathbb{R}^{n \times m}$을 형성하기 위해,
저희는 단순히 각 원소 $c_{ij}$를
$\mathbf{A}$의 $i$번째 행과
$\mathbf{B}$의 $j$번째 열 사이의 내적으로
계산합니다. 즉, $\mathbf{a}^\top_i \mathbf{b}_j$입니다.

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**저희는 행렬-행렬 곱셈 $\mathbf{AB}$를
$m$개의 행렬-벡터 곱이나
$m \times n$개의 내적을 수행하고
결과들을 함께 꿰매어 $n \times m$ 행렬을 형성하는 것으로
생각할 수 있습니다.**]
다음 코드 조각에서,
저희는 `A`와 `B`에 대해 행렬 곱셈을 수행합니다.
여기서, `A`는 두 행과 세 열을 가진 행렬이고,
`B`는 세 행과 네 열을 가진 행렬입니다.
곱셈 후에, 두 행과 네 열을 가진 행렬을 얻습니다.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

*행렬-행렬 곱셈*이라는 용어는
종종 *행렬 곱셈*으로 단순화되며,
아다마르 곱과 혼동되어서는 안 됩니다.


## 노름
:label:`subsec_lin-algebra-norms`

선형대수에서 가장 유용한 연산자 중 일부는 *노름*입니다.
비공식적으로, 벡터의 노름은 그것이 얼마나 *큰지*를 알려줍니다.
예를 들어, $\ell_2$ 노름은
벡터의 (유클리드) 길이를 측정합니다.
여기서, 저희는 벡터의 성분의 크기에 관한 *크기* 개념을 사용하고 있습니다
(그것의 차원성이 아니라).

노름은 벡터를 스칼라로 매핑하고
다음 세 가지 속성을 만족하는 함수 $\| \cdot \|$입니다.

1. 임의의 벡터 $\mathbf{x}$가 주어졌을 때, 벡터(의 모든 원소)를
   스칼라 $\alpha \in \mathbb{R}$로 스케일링하면, 그것의 노름은 그에 따라 스케일링됩니다.
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. 임의의 벡터 $\mathbf{x}$와 $\mathbf{y}$에 대해,
   노름은 삼각 부등식을 만족합니다.
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. 벡터의 노름은 음이 아니며 벡터가 영일 때에만 소멸합니다.
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

많은 함수들이 유효한 노름이며 서로 다른 노름들은
서로 다른 크기 개념을 인코딩합니다.
저희가 모두 초등학교 기하학에서
직각삼각형의 빗변을 계산할 때 배운 유클리드 노름은
벡터 원소들의 제곱의 합의 제곱근입니다.
형식적으로, 이는 [**$\ell_2$ *노름***]이라고 부르며 다음과 같이 표현됩니다.

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

메서드 `norm`은 $\ell_2$ 노름을 계산합니다.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**$\ell_1$ 노름**]도 일반적이며
관련된 측도는 맨해튼 거리라고 부릅니다.
정의에 의해, $\ell_1$ 노름은
벡터 원소들의 절댓값을 합산합니다.

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

$\ell_2$ 노름과 비교하여, 이상치에 덜 민감합니다.
$\ell_1$ 노름을 계산하기 위해,
저희는 절댓값을 합 연산과 합성합니다.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

$\ell_2$와 $\ell_1$ 노름은 모두
더 일반적인 $\ell_p$ *노름*의 특수한 경우입니다.

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

행렬의 경우, 문제는 더 복잡합니다.
결국, 행렬은 개별 항목들의 집합으로 볼 수도 있고
*그리고* 벡터에 작용하여 그것들을 다른 벡터로 변환하는 객체로 볼 수도 있습니다.
예를 들어, 행렬-벡터 곱 $\mathbf{X} \mathbf{v}$가
$\mathbf{v}$에 비해 얼마나 더 길 수 있는지를
물을 수 있습니다.
이러한 사고의 흐름은 *스펙트럼* 노름이라고 부르는 것으로 이어집니다.
지금은, [**계산하기 훨씬 쉬운 *프로베니우스 노름*을 소개**]하며
이는 행렬 원소들의 제곱의 합의 제곱근으로 정의됩니다.

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

프로베니우스 노름은 마치
행렬 모양의 벡터의 $\ell_2$ 노름인 것처럼 동작합니다.
다음 함수를 호출하면
행렬의 프로베니우스 노름을 계산할 것입니다.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

저희가 너무 앞서 나가고 싶지는 않지만,
이러한 개념들이 왜 유용한지에 대한 약간의 직관을 이미 심을 수 있습니다.
딥러닝에서는, 종종 최적화 문제를 풀려고 합니다.
관찰된 데이터에 할당된 확률을 *최대화*하고,
추천 모델과 관련된 수익을 *최대화*하고,
예측과 정답 관찰 사이의 거리를 *최소화*하고,
같은 사람의 사진의 표현들 사이의 거리는 *최소화*하면서
다른 사람들의 사진의 표현들 사이의 거리는 *최대화*하는 것입니다.
딥러닝 알고리즘의 목표를 구성하는 이러한 거리는
종종 노름으로 표현됩니다.


## 논의

이 절에서, 저희는 여러분이 현대 딥러닝의 상당 부분을
이해하는 데 필요한 모든 선형대수를 살펴보았습니다.
하지만 선형대수에는 훨씬 더 많은 것이 있고,
그 많은 부분이 머신러닝에 유용합니다.
예를 들어, 행렬은 인자로 분해될 수 있으며,
이러한 분해는 실제 데이터셋의
저차원 구조를 드러낼 수 있습니다.
데이터셋의 구조를 발견하고
예측 문제를 풀기 위해
행렬 분해와 고차 텐서로의 그것의 일반화를
사용하는 데 초점을 맞춘
머신러닝의 전체 하위 분야가 있습니다.
하지만 이 책은 딥러닝에 초점을 맞춥니다.
그리고 저희는 여러분이 일단 실제 데이터셋에 머신러닝을 적용하면서
직접 다뤄본 후에 더 많은 수학을 배우는 데
더 끌리게 될 것이라고 믿습니다.
따라서 나중에 더 많은 수학을 소개할 권리는 유보하면서도,
저희는 여기서 이 절을 마무리합니다.

더 많은 선형대수를 배우고 싶다면,
훌륭한 책과 온라인 자료가 많이 있습니다.
보다 고급의 속성 강좌를 원한다면,
:citet:`Strang.1993`, :citet:`Kolter.2008`, :citet:`Petersen.Pedersen.ea.2008`을 확인해 보십시오.

요약하자면,

* 스칼라, 벡터, 행렬, 텐서는
  선형대수에서 사용되는 기본 수학적 객체이며
  각각 0, 1, 2, 임의의 수의 축을 가집니다.
* 텐서는 인덱싱을 통해 지정된 축을 따라 슬라이싱되거나
  `sum`과 `mean` 같은 연산을 통해 각각 축소될 수 있습니다.
* 원소별 곱은 아다마르 곱이라고 부릅니다.
  대조적으로, 내적, 행렬-벡터 곱, 행렬-행렬 곱은
  원소별 연산이 아니며 일반적으로 피연산자와는 다른 모양을 가진
  객체를 반환합니다.
* 아다마르 곱과 비교하여, 행렬-행렬 곱은
  계산하는 데 상당히 더 오래 걸립니다(이차 시간이 아닌 삼차 시간).
* 노름은 벡터(또는 행렬)의 크기에 대한 다양한 개념을 포착하며,
  거리를 측정하기 위해 두 벡터의 차이에
  일반적으로 적용됩니다.
* 일반적인 벡터 노름에는 $\ell_1$과 $\ell_2$ 노름이 포함되며,
   일반적인 행렬 노름에는 *스펙트럼*과 *프로베니우스* 노름이 포함됩니다.


## 연습문제

1. 행렬의 전치의 전치가 그 행렬 자체임을 증명하세요: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. 두 행렬 $\mathbf{A}$와 $\mathbf{B}$가 주어졌을 때, 합과 전치가 가환임을 보이세요: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. 임의의 정사각 행렬 $\mathbf{A}$가 주어졌을 때, $\mathbf{A} + \mathbf{A}^\top$은 항상 대칭입니까? 이전 두 연습문제의 결과만을 사용하여 그 결과를 증명할 수 있나요?
1. 저희는 이 절에서 모양이 (2, 3, 4)인 텐서 `X`를 정의했습니다. `len(X)`의 출력은 무엇입니까? 코드를 구현하지 않고 답을 작성한 다음, 코드를 사용하여 답을 확인하세요.
1. 임의의 모양의 텐서 `X`에 대해, `len(X)`는 항상 `X`의 특정 축의 길이에 해당합니까? 그 축은 무엇입니까?
1. `A / A.sum(axis=1)`을 실행하고 무슨 일이 일어나는지 보세요. 결과를 분석할 수 있나요?
1. 맨해튼 시내의 두 지점 사이를 이동할 때, 좌표의 측면에서, 즉 거리(avenues)와 거리(streets)의 측면에서 이동해야 하는 거리는 얼마입니까? 대각선으로 이동할 수 있습니까?
1. 모양이 (2, 3, 4)인 텐서를 고려해 보세요. 축 0, 1, 2를 따른 합산 출력의 모양은 무엇입니까?
1. 세 개 이상의 축을 가진 텐서를 `linalg.norm` 함수에 공급하고 그 출력을 관찰해 보세요. 이 함수는 임의의 모양의 텐서에 대해 무엇을 계산합니까?
1. 세 개의 큰 행렬, 예를 들어 가우시안 무작위 변수로 초기화된 $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$, $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$를 고려해 보세요. 곱 $\mathbf{A} \mathbf{B} \mathbf{C}$를 계산하고자 합니다. $(\mathbf{A} \mathbf{B}) \mathbf{C}$를 계산하는지 $\mathbf{A} (\mathbf{B} \mathbf{C})$를 계산하는지에 따라 메모리 사용량과 속도에 차이가 있습니까? 왜 그렇습니까?
1. 세 개의 큰 행렬, 예를 들어 $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$, $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$을 고려해 보세요. $\mathbf{A} \mathbf{B}$를 계산하는지 $\mathbf{A} \mathbf{C}^\top$을 계산하는지에 따라 속도에 차이가 있습니까? 왜 그렇습니까? 메모리를 복제하지 않고 $\mathbf{C} = \mathbf{B}^\top$로 초기화하면 무엇이 바뀝니까? 왜 그렇습니까?
1. 세 개의 행렬, 예를 들어 $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$을 고려해 보세요. $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$를 쌓아 세 개의 축을 가진 텐서를 구성하세요. 차원성은 무엇입니까? 세 번째 축의 두 번째 좌표를 잘라내어 $\mathbf{B}$를 복원하세요. 답이 올바른지 확인하세요.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17968)
:end_tab:
