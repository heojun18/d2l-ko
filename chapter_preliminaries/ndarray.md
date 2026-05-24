```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 데이터 조작
:label:`sec_ndarray`

무언가를 해내려면,
데이터를 저장하고 조작하는 방법이 필요합니다.
일반적으로, 데이터로 해야 할 중요한 두 가지가 있습니다.
(i) 데이터를 획득하는 것,
그리고 (ii) 컴퓨터 안으로 들어온 데이터를 처리하는 것입니다.
저장할 방법도 없이 데이터를 획득하는 것은 의미가 없으므로,
시작하기 위해 *텐서*라고도 부르는
$n$차원 배열을 직접 다뤄봅시다.
NumPy 과학 컴퓨팅 패키지를
이미 알고 있다면, 이는 식은 죽 먹기일 것입니다.
모든 현대 딥러닝 프레임워크에서,
*텐서 클래스*(MXNet의 `ndarray`,
PyTorch와 TensorFlow의 `Tensor`)는
NumPy의 `ndarray`와 닮아 있으며,
몇 가지 강력한 기능이 추가되어 있습니다.
첫째, 텐서 클래스는
자동 미분을 지원합니다.
둘째, NumPy가 CPU에서만 실행되는 것과 달리
GPU를 활용해 수치 계산을 가속합니다.
이러한 특성들은 신경망을
코딩하기 쉽고 빠르게 실행되게 만들어 줍니다.



## 시작하기

:begin_tab:`mxnet`
시작하기 위해, MXNet에서 `np`(`numpy`)와
`npx`(`numpy_extension`) 모듈을 임포트합니다.
여기서 `np` 모듈은 NumPy가 지원하는
함수들을 포함하고 있으며,
`npx` 모듈은 NumPy와 유사한 환경 내에서
딥러닝을 강화하기 위해 개발된
확장 기능들의 집합을 포함합니다.
텐서를 사용할 때, 저희는 거의 항상
`set_np` 함수를 호출합니다.
이는 MXNet의 다른 구성 요소들에 의한
텐서 처리의 호환성을 위한 것입니다.
:end_tab:

:begin_tab:`pytorch`
(**시작하기 위해, PyTorch 라이브러리를 임포트합니다.
패키지 이름은 `torch`라는 점에 유의하십시오.**)
:end_tab:

:begin_tab:`tensorflow`
시작하기 위해, `tensorflow`를 임포트합니다.
간결함을 위해, 실무자들은 종종
별칭 `tf`를 할당합니다.
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**텐서는 (잠재적으로 다차원의) 수치 값 배열을 나타냅니다.**]
1차원의 경우, 즉 데이터에 단 하나의 축만 필요할 때,
텐서는 *벡터*라고 부릅니다.
두 개의 축을 가지면, 텐서는 *행렬*이라고 부릅니다.
$k > 2$개의 축을 가지면, 저희는 특수한 이름을 사용하지 않고
그 객체를 단순히 $k$차-*텐서*라고 부릅니다.

:begin_tab:`mxnet`
MXNet은 미리 값을 채운 새로운 텐서를
생성하기 위한 다양한 함수를 제공합니다.
예를 들어, `arange(n)`을 호출하면,
0(포함)에서 시작해 `n`(미포함)에서 끝나는
균등하게 간격이 있는 값들의 벡터를
생성할 수 있습니다.
기본적으로, 간격 크기는 $1$입니다.
별도로 지정하지 않는 한,
새로운 텐서는 메인 메모리에 저장되며
CPU 기반 계산을 위해 지정됩니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch는 미리 값을 채운 새로운 텐서를
생성하기 위한 다양한 함수를 제공합니다.
예를 들어, `arange(n)`을 호출하면,
0(포함)에서 시작해 `n`(미포함)에서 끝나는
균등하게 간격이 있는 값들의 벡터를
생성할 수 있습니다.
기본적으로, 간격 크기는 $1$입니다.
별도로 지정하지 않는 한,
새로운 텐서는 메인 메모리에 저장되며
CPU 기반 계산을 위해 지정됩니다.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow는 미리 값을 채운 새로운 텐서를
생성하기 위한 다양한 함수를 제공합니다.
예를 들어, `range(n)`을 호출하면,
0(포함)에서 시작해 `n`(미포함)에서 끝나는
균등하게 간격이 있는 값들의 벡터를
생성할 수 있습니다.
기본적으로, 간격 크기는 $1$입니다.
별도로 지정하지 않는 한,
새로운 텐서는 메인 메모리에 저장되며
CPU 기반 계산을 위해 지정됩니다.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
이 값들 각각은
텐서의 *원소*라고 부릅니다.
텐서 `x`는 12개의 원소를 포함합니다.
텐서의 전체 원소 개수는
`size` 속성을 통해 확인할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이 값들 각각은
텐서의 *원소*라고 부릅니다.
텐서 `x`는 12개의 원소를 포함합니다.
텐서의 전체 원소 개수는
`numel` 메서드를 통해 확인할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
이 값들 각각은
텐서의 *원소*라고 부릅니다.
텐서 `x`는 12개의 원소를 포함합니다.
텐서의 전체 원소 개수는
`size` 함수를 통해 확인할 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**저희는 텐서의 *모양***)
(각 축을 따른 길이)을
`shape` 속성을 통해 접근할 수 있습니다.
여기서는 벡터를 다루고 있으므로,
`shape`는 단 하나의 원소만 포함하며
크기와 동일합니다.

```{.python .input}
%%tab all
x.shape
```

`reshape`를 호출하여
[**텐서의 크기나 값을 바꾸지 않고
텐서의 모양을 변경할 수 있습니다**].
예를 들어, 모양이 (12,)인 벡터 `x`를
모양이 (3, 4)인 행렬 `X`로 변환할 수 있습니다.
이 새로운 텐서는 모든 원소를 유지하지만
이를 행렬로 재구성합니다.
벡터의 원소들이 한 행씩 차례로 배치되므로
`x[3] == X[0, 3]`이라는 점에 유의하십시오.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

`reshape`에 모든 모양 구성 요소를 지정하는 것은
중복적이라는 점에 유의하십시오.
텐서의 크기를 이미 알고 있기 때문에,
나머지가 주어졌을 때 모양의 한 구성 요소를 계산할 수 있습니다.
예를 들어, 크기가 $n$인 텐서와
목표 모양 ($h$, $w$)이 주어지면,
$w = n/h$임을 알 수 있습니다.
모양의 한 구성 요소를 자동으로 추론하기 위해,
자동으로 추론되어야 하는 모양 구성 요소에
`-1`을 넣을 수 있습니다.
저희의 경우, `x.reshape(3, 4)`를 호출하는 대신,
`x.reshape(-1, 4)` 또는 `x.reshape(3, -1)`을 호출해도 동등합니다.

실무자들은 종종 모든 원소가 0이나 1로 초기화된
텐서를 다뤄야 합니다.
[**모든 원소가 0으로 설정된 텐서를 구성할 수 있으며**] (~~또는 1~~)
모양이 (2, 3, 4)인 텐서를 `zeros` 함수를 통해 만들 수 있습니다.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

마찬가지로, `ones`를 호출하여
모든 원소가 1인 텐서를 생성할 수 있습니다.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

저희는 종종 주어진 확률 분포로부터
[**각 원소를 (독립적으로) 무작위로 샘플링**]
하고자 합니다.
예를 들어, 신경망의 파라미터는
종종 무작위로 초기화됩니다.
다음 코드 조각은 평균 0과 표준편차 1을 가진
표준 가우시안(정규) 분포로부터 추출된 원소들로
텐서를 생성합니다.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

마지막으로, 수치 리터럴을 포함하는
(잠재적으로 중첩된) Python 리스트(들)를 제공함으로써
[**각 원소에 대한 정확한 값을 제공하여**]
텐서를 구성할 수 있습니다.
여기서는 리스트의 리스트로 행렬을 구성하며,
가장 바깥쪽 리스트는 축 0에 해당하고,
안쪽 리스트는 축 1에 해당합니다.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 인덱싱과 슬라이싱

Python 리스트와 마찬가지로,
인덱싱(0에서 시작)을 통해
텐서 원소에 접근할 수 있습니다.
리스트의 끝을 기준으로 한 상대적 위치에 따라
원소에 접근하려면,
음수 인덱싱을 사용할 수 있습니다.
마지막으로, 슬라이싱(예: `X[start:stop]`)을 통해
인덱스의 전체 범위에 접근할 수 있으며,
반환된 값은 첫 번째 인덱스(`start`)를 포함하지만
*마지막은 포함하지 않습니다* (`stop`).
마지막으로, $k$차 텐서에 대해
단 하나의 인덱스(또는 슬라이스)만 지정되면,
이는 축 0을 따라 적용됩니다.
따라서, 다음 코드에서,
[**`[-1]`은 마지막 행을 선택하고 `[1:3]`은
두 번째와 세 번째 행을 선택합니다**].

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
읽는 것을 넘어서, (**인덱스를 지정하여 행렬의 원소를 *쓸* 수도 있습니다.**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow의 `Tensors`는 불변이며, 할당할 수 없습니다.
TensorFlow의 `Variables`는 할당을 지원하는
가변 상태 컨테이너입니다. TensorFlow의 그래디언트는
`Variable` 할당을 통해 역방향으로 흐르지 않는다는 점을 명심하십시오.

전체 `Variable`에 값을 할당하는 것을 넘어서, 인덱스를 지정하여 `Variable`의
원소를 쓸 수 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

[**여러 원소에 동일한 값을 할당하고자 한다면,
할당 연산의 좌변에
인덱싱을 적용합니다.**]
예를 들어, `[:2, :]`는 첫 번째와 두 번째 행에 접근하며,
`:`는 축 1(열)을 따라 모든 원소를 취합니다.
저희는 행렬에 대한 인덱싱을 논의했지만,
이는 벡터와 2차원 이상의 텐서에도
적용됩니다.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## 연산

이제 텐서를 구성하는 방법과
그 원소를 읽고 쓰는 방법을 알게 되었으므로,
다양한 수학적 연산으로
텐서를 조작하기 시작할 수 있습니다.
이 중 가장 유용한 것은
*원소별* 연산입니다.
이는 텐서의 각 원소에
표준 스칼라 연산을 적용합니다.
두 텐서를 입력으로 받는 함수의 경우,
원소별 연산은 대응되는 각 원소 쌍에
어떤 표준 이항 연산자를 적용합니다.
스칼라에서 스칼라로 매핑하는
모든 함수로부터 원소별 함수를
생성할 수 있습니다.

수학적 표기법에서, 저희는 이러한
*단항* 스칼라 연산자(하나의 입력을 받는)를
시그니처
$f: \mathbb{R} \rightarrow \mathbb{R}$로 표시합니다.
이는 단지 함수가 임의의 실수를
다른 실수로 매핑한다는 것을 의미합니다.
$e^x$와 같은 단항 연산자를 포함한 대부분의 표준 연산자는 원소별로 적용될 수 있습니다.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

마찬가지로, 실수의 쌍을
(단일) 실수로 매핑하는
*이항* 스칼라 연산자를
시그니처
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$로 표시합니다.
*같은 모양의* 임의의 두 벡터 $\mathbf{u}$와
$\mathbf{v}$, 그리고 이항 연산자 $f$가 주어지면, 
모든 $i$에 대해 $c_i \gets f(u_i, v_i)$로 설정하여
벡터
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$를 생성할 수 있습니다.
여기서 $c_i, u_i$, $v_i$는 벡터 $\mathbf{c}, \mathbf{u}$, $\mathbf{v}$의
$i$번째 원소입니다.
여기서, 저희는 스칼라 함수를
원소별 벡터 연산으로 *끌어올려서*
벡터값
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$를
생성했습니다.
덧셈(`+`), 뺄셈(`-`),
곱셈(`*`), 나눗셈(`/`), 
거듭제곱(`**`)을 위한
일반적인 표준 산술 연산자는 모두
임의의 모양의 동일한 모양 텐서에 대해
원소별 연산으로 *끌어올려졌습니다*.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

원소별 계산 외에도,
내적과 행렬 곱셈과 같은
선형대수 연산도 수행할 수 있습니다.
이에 대해서는
:numref:`sec_linear-algebra`에서 자세히 설명할 것입니다.

저희는 또한 [**여러 텐서를 *연결*할 수도 있습니다**].
이를 끝과 끝을 맞붙여 더 큰 텐서를 만듭니다.
저희는 단지 텐서들의 리스트를 제공하고
어떤 축을 따라 연결할지 시스템에 알려주기만 하면 됩니다.
아래 예제는 두 행렬을 열(축 1) 대신
행(축 0)을 따라 연결할 때
어떤 일이 일어나는지 보여줍니다.
첫 번째 출력의 축 0 길이($6$)는
두 입력 텐서의 축 0 길이의 합($3 + 3$)이며,
두 번째 출력의 축 1 길이($8$)는
두 입력 텐서의 축 1 길이의 합($4 + 4$)임을 볼 수 있습니다.

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

때때로, 저희는
[***논리 문*을 통해 이진 텐서를 구성**]하고 싶습니다.
예로 `X == Y`를 들어봅시다.
각 위치 `i, j`에 대해, `X[i, j]`와 `Y[i, j]`가 같으면,
결과의 해당 항목은 값 `1`을 가지고,
그렇지 않으면 값 `0`을 가집니다.

```{.python .input}
%%tab all
X == Y
```

[**텐서의 모든 원소를 합산하면**] 단 하나의 원소를 가진 텐서가 됩니다.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## 브로드캐스팅
:label:`subsec_broadcasting`

지금까지, 여러분은 동일한 모양의 두 텐서에 대해
원소별 이항 연산을 수행하는 방법을 알게 되었습니다.
특정 조건 하에서는,
모양이 다를 때조차도,
*브로드캐스팅 메커니즘*을 호출하여
[**원소별 이항 연산을 수행**]할 수 있습니다.
브로드캐스팅은 다음의 2단계 절차에 따라 동작합니다.
(i) 길이가 1인 축을 따라 원소를 복사하여
하나 또는 두 배열을 확장함으로써,
이 변환 후에 두 텐서가 동일한 모양을 갖도록 하고,
(ii) 결과 배열에 대해
원소별 연산을 수행합니다.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

`a`와 `b`가 각각 $3\times1$과
$1\times2$ 행렬이기 때문에,
그 모양이 일치하지 않습니다.
브로드캐스팅은 행렬 `a`를 열을 따라
복제하고 행렬 `b`를 행을 따라 복제한 후
원소별로 더함으로써
더 큰 $3\times2$ 행렬을 생성합니다.

```{.python .input}
%%tab all
a + b
```

## 메모리 절약

[**연산을 실행하면 결과를 담기 위해 새로운 메모리가
할당될 수 있습니다.**]
예를 들어, `Y = X + Y`라고 쓰면,
저희는 `Y`가 가리키던 텐서를 역참조하고
대신 `Y`를 새로 할당된 메모리에 가리키게 합니다.
이 문제는 메모리에서 참조된 객체의 정확한 주소를 알려주는
Python의 `id()` 함수로 입증할 수 있습니다.
`Y = Y + X`를 실행한 후,
`id(Y)`가 다른 위치를 가리킨다는 점에 유의하십시오.
이는 Python이 먼저 `Y + X`를 평가하여,
결과를 위한 새로운 메모리를 할당한 후
`Y`를 메모리의 이 새로운 위치로 가리키게 하기 때문입니다.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

이는 두 가지 이유로 바람직하지 않을 수 있습니다.
첫째, 저희는 항상 불필요하게 메모리를 할당하면서
돌아다니고 싶지 않습니다.
머신러닝에서는, 종종 수백 메가바이트의 파라미터를 가지고
초당 여러 번 그것들 모두를 업데이트합니다.
가능할 때마다, 저희는 이러한 업데이트를 *제자리에서* 수행하고자 합니다.
둘째, 저희는 여러 변수로부터
동일한 파라미터를 가리킬 수도 있습니다.
제자리에서 업데이트하지 않으면,
메모리 누수를 일으키거나 의도치 않게 오래된 파라미터를 참조하지 않도록
이러한 모든 참조를 신중하게 업데이트해야 합니다.

:begin_tab:`mxnet, pytorch`
다행스럽게도, (**제자리 연산을 수행하는 것**)은 쉽습니다.
저희는 슬라이스 표기법을 사용하여
이전에 할당된 배열 `Y`에 연산 결과를
할당할 수 있습니다: `Y[:] = <expression>`.
이 개념을 설명하기 위해,
저희는 `zeros_like`를 사용하여 `Y`와 같은 모양을 갖도록
초기화한 후 텐서 `Z`의 값을 덮어씁니다.
:end_tab:

:begin_tab:`tensorflow`
`Variables`는 TensorFlow에서 가변 상태 컨테이너입니다. 그것들은
모델 파라미터를 저장하는 방법을 제공합니다.
저희는 `assign`을 사용하여
`Variable`에 연산 결과를 할당할 수 있습니다.
이 개념을 설명하기 위해, 
`zeros_like`를 사용하여 `Y`와 같은 모양을 갖도록
초기화한 후 `Variable` `Z`의 값을 덮어씁니다.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**`X`의 값이 이후의 계산에서 재사용되지 않는다면,
연산의 메모리 오버헤드를 줄이기 위해
`X[:] = X + Y` 또는 `X += Y`를 사용할 수도 있습니다.**]
:end_tab:

:begin_tab:`tensorflow`
일단 `Variable`에 영구적으로 상태를 저장하더라도,
모델 파라미터가 아닌 텐서에 대한 과도한 할당을
피함으로써 메모리 사용량을 추가로 줄이고 싶을 수 있습니다.
TensorFlow `Tensors`는 불변이고
그래디언트가 `Variable` 할당을 통해 흐르지 않기 때문에,
TensorFlow는 개별 연산을 제자리에서 실행하기 위한
명시적인 방법을 제공하지 않습니다.

그러나, TensorFlow는 실행 전에 컴파일되고 최적화되는
TensorFlow 그래프 내부에 계산을 감싸기 위해
`tf.function` 데코레이터를 제공합니다.
이를 통해 TensorFlow는 사용되지 않는 값을 제거하고,
더 이상 필요하지 않은 이전 할당을 재사용할 수 있습니다.
이는 TensorFlow 계산의 메모리 오버헤드를 최소화합니다.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 다른 Python 객체로의 변환

:begin_tab:`mxnet, tensorflow`
[**NumPy 텐서(`ndarray`)로 변환**]하거나, 그 반대도 쉽습니다.
변환된 결과는 메모리를 공유하지 않습니다.
이 작은 불편함은 사실 꽤 중요합니다.
CPU나 GPU에서 연산을 수행할 때,
Python의 NumPy 패키지가 동일한 메모리 청크로
다른 일을 하고 싶어할 수도 있는지를
보기 위해 계산을 중단하고 싶지는 않을 것이기 때문입니다.
:end_tab:

:begin_tab:`pytorch`
[**NumPy 텐서(`ndarray`)로 변환**]하거나, 그 반대도 쉽습니다.
torch 텐서와 NumPy 배열은
바탕이 되는 메모리를 공유하며,
하나를 제자리 연산을 통해 변경하면
다른 것도 변경됩니다.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

(**크기 1의 텐서를 Python 스칼라로 변환**)하기 위해,
`item` 함수나 Python의 내장 함수를 호출할 수 있습니다.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## 요약

텐서 클래스는 딥러닝 라이브러리에서 데이터를 저장하고 조작하기 위한 주요 인터페이스입니다.
텐서는 구성 루틴, 인덱싱과 슬라이싱, 기본적인 수학 연산, 브로드캐스팅, 메모리 효율적인 할당, 다른 Python 객체와의 상호 변환을 포함한 다양한 기능을 제공합니다.


## 연습문제

1. 이 절의 코드를 실행해 보세요. 조건문 `X == Y`를 `X < Y` 또는 `X > Y`로 바꾼 후, 어떤 종류의 텐서를 얻을 수 있는지 확인해 보세요.
1. 브로드캐스팅 메커니즘에서 원소별로 연산하는 두 텐서를 다른 모양, 예를 들어 3차원 텐서로 바꿔보세요. 결과가 예상과 동일한가요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17966)
:end_tab:
