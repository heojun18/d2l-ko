```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 이미지를 위한 합성곱
:label:`sec_conv_layer`

이제 저희는 합성곱 계층이 이론적으로 어떻게 작동하는지 이해했으므로,
실제로 그것이 어떻게 작동하는지 살펴볼 준비가 되었습니다.
이미지 데이터의 구조를 탐색하기 위한 효율적인 아키텍처로서의
합성곱 신경망에 대한 저희의 동기를 바탕으로,
저희는 이미지를 반복되는 예제로 계속 사용합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 상호상관 연산

엄밀히 말하면 합성곱 계층은 잘못된 명칭이라는 점을 떠올려 보세요.
이 계층이 표현하는 연산은 상호상관(cross-correlation)으로 더 정확하게 기술되기 때문입니다.
:numref:`sec_why-conv`에서의 합성곱 계층에 대한 저희의 기술에 따르면,
그러한 계층에서는 입력 텐서와 커널 텐서가 결합되어
(**상호상관 연산을 통해**) 출력 텐서를 생성합니다.

지금은 채널은 무시하고 2차원 데이터와 은닉 표현으로
이것이 어떻게 작동하는지 살펴봅시다.
:numref:`fig_correlation`에서
입력은 높이 3, 너비 3인 2차원 텐서입니다.
저희는 텐서의 모양을 $3 \times 3$ 또는 ($3$, $3$)로 표시합니다.
커널의 높이와 너비는 모두 2입니다.
*커널 윈도우*(또는 *합성곱 윈도우*)의 모양은
커널의 높이와 너비로 주어집니다
(여기서는 $2 \times 2$).

![2차원 상호상관 연산. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

2차원 상호상관 연산에서는,
합성곱 윈도우를 입력 텐서의 왼쪽 위 모서리에 위치시키는 것으로 시작하여,
왼쪽에서 오른쪽으로, 위에서 아래로 입력 텐서를 가로질러 미끄러뜨립니다.
합성곱 윈도우가 특정 위치로 미끄러질 때,
해당 윈도우에 포함된 입력 부분 텐서와 커널 텐서가
원소별로 곱해지고,
결과 텐서가 합산되어
단일 스칼라 값을 산출합니다.
이 결과는 대응하는 위치에서의 출력 텐서 값을 제공합니다.
여기서 출력 텐서는 높이 2, 너비 2이며,
네 개의 요소는 2차원 상호상관 연산으로부터 도출됩니다.

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

각 축을 따라 출력 크기가
입력 크기보다 약간 작다는 점에 주목하세요.
커널의 너비와 높이가 $1$보다 크기 때문에,
저희는 커널이 이미지 내에 완전히 들어맞는 위치에 대해서만
상호상관을 적절하게 계산할 수 있고,
출력 크기는 입력 크기 $n_\textrm{h} \times n_\textrm{w}$에서
합성곱 커널 크기 $k_\textrm{h} \times k_\textrm{w}$를 뺀
다음 식으로 주어집니다.

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$

이는 저희가 합성곱 커널을 이미지를 가로질러 "이동"시키기 위해
충분한 공간이 필요하기 때문입니다.
나중에 저희는 커널을 이동시킬 충분한 공간이 있도록
이미지 경계 주변을 0으로 패딩하여
크기를 변경하지 않고 유지하는 방법을 살펴보겠습니다.
다음으로, 저희는 입력 텐서 `X`와 커널 텐서 `K`를 받아
출력 텐서 `Y`를 반환하는 `corr2d` 함수에서 이 과정을 구현합니다.

```{.python .input}
%%tab mxnet
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
%%tab jax
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y = Y.at[i, j].set((X[i:i + h, j:j + w] * K).sum())
    return Y
```

```{.python .input}
%%tab tensorflow
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

2차원 상호상관 연산의 [**위 구현의 출력을 검증하기 위해**]
:numref:`fig_correlation`에서 입력 텐서 `X`와 커널 텐서 `K`를
구성할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 합성곱 계층

합성곱 계층은 입력과 커널을 상호상관시키고
스칼라 편향을 더해 출력을 생성합니다.
합성곱 계층의 두 매개변수는
커널과 스칼라 편향입니다.
합성곱 계층에 기반한 모델을 훈련할 때,
저희는 일반적으로 완전 연결 계층에서와 마찬가지로
커널을 무작위로 초기화합니다.

이제 저희는 위에서 정의한 `corr2d` 함수에 기반한
[**2차원 합성곱 계층을 구현**]할 준비가 되었습니다.
`__init__` 생성자 메서드에서,
저희는 `weight`와 `bias`를 두 개의 모델 매개변수로 선언합니다.
순전파 메서드는
`corr2d` 함수를 호출하고 편향을 더합니다.

```{.python .input}
%%tab mxnet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
%%tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

```{.python .input}
%%tab jax
class Conv2D(nn.Module):
    kernel_size: int

    def setup(self):
        self.weight = nn.param('w', nn.initializers.uniform, self.kernel_size)
        self.bias = nn.param('b', nn.initializers.zeros, 1)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

$h \times w$ 합성곱
또는 $h \times w$ 합성곱 커널에서,
합성곱 커널의 높이와 너비는 각각 $h$와 $w$입니다.
또한 $h \times w$ 합성곱 커널을 가진 합성곱 계층을 단순히
$h \times w$ 합성곱 계층이라고 부르기도 합니다.


## 이미지에서의 객체 에지 검출

[**합성곱 계층의 간단한 응용,
즉 이미지에서 객체의 에지를 검출하는 것을**]
픽셀 변화 위치를 찾음으로써 잠시 살펴봅시다.
먼저, 저희는 $6\times 8$ 픽셀의 "이미지"를 구성합니다.
가운데 네 열은 검정($0$)이고 나머지는 흰색($1$)입니다.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
%%tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

```{.python .input}
%%tab jax
X = jnp.ones((6, 8))
X = X.at[:, 2:6].set(0)
X
```

다음으로, 높이 1, 너비 2의 커널 `K`를 구성합니다.
저희가 입력과 상호상관 연산을 수행할 때,
수평으로 인접한 요소들이 같다면
출력은 0이 됩니다. 그렇지 않으면 출력은 0이 아닙니다.
이 커널은 유한 차분 연산자의 특별한 경우라는 점에 주목하세요. 위치 $(i,j)$에서 그것은 $x_{i,j} - x_{(i+1),j}$를 계산합니다. 즉, 수평으로 인접한 픽셀들의 값 사이의 차를 계산합니다. 이는 수평 방향의 1차 도함수의 이산 근사입니다. 결국, 함수 $f(i,j)$에 대해 그것의 도함수 $-\partial_i f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j) - f(i+\epsilon,j)}{\epsilon}$입니다. 실제로 이것이 어떻게 작동하는지 살펴봅시다.

```{.python .input}
%%tab all
K = d2l.tensor([[1.0, -1.0]])
```

이제 인수 `X`(저희의 입력)와 `K`(저희의 커널)로
상호상관 연산을 수행할 준비가 되었습니다.
보시다시피, [**저희는 흰색에서 검정으로의 에지에 대해 $1$을,
검정에서 흰색으로의 에지에 대해 $-1$을 검출합니다**].
그 외 모든 출력은 $0$ 값을 갖습니다.

```{.python .input}
%%tab all
Y = corr2d(X, K)
Y
```

이제 저희는 전치된 이미지에 커널을 적용할 수 있습니다.
예상한 대로, 그것은 사라집니다. [**커널 `K`는 수직 에지만 검출합니다.**]

```{.python .input}
%%tab all
corr2d(d2l.transpose(X), K)
```

## 커널 학습하기

저희가 정확히 찾고 있는 것이 무엇인지 안다면
유한 차분 `[1, -1]`로 에지 검출기를 설계하는 것은 깔끔합니다.
하지만 더 큰 커널을 보고
연속적인 합성곱 계층을 고려할 때,
각 필터가 무엇을 해야 하는지 정확하게
수동으로 명시하는 것은 불가능할 수 있습니다.

이제 입력-출력 쌍만을 보고
[**`X`로부터 `Y`를 생성한 커널을 학습**]할 수 있는지
살펴봅시다.
저희는 먼저 합성곱 계층을 구성하고
그것의 커널을 무작위 텐서로 초기화합니다.
다음으로, 각 반복마다 제곱 오차를 사용하여
`Y`와 합성곱 계층의 출력을 비교합니다.
그런 다음 기울기를 계산하여 커널을 업데이트할 수 있습니다.
단순함을 위해,
다음에서 저희는
2차원 합성곱 계층에 대한 내장 클래스를 사용하고
편향은 무시합니다.

```{.python .input}
%%tab mxnet
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Learning rate

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
%%tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
%%tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Learning rate

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

```{.python .input}
%%tab jax
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv(1, kernel_size=(1, 2), use_bias=False, padding='VALID')

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 6, 8, 1))
Y = Y.reshape((1, 6, 7, 1))
lr = 3e-2  # Learning rate

params = conv2d.init(jax.random.PRNGKey(d2l.get_seed()), X)

def loss(params, X, Y):
    Y_hat = conv2d.apply(params, X)
    return ((Y_hat - Y) ** 2).sum()

for i in range(10):
    l, grads = jax.value_and_grad(loss)(params, X, Y)
    # Update the kernel
    params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l:.3f}')
```

10번의 반복 후에 오차가 작은 값으로 떨어졌다는 점에 주목하세요. 이제 [**저희가 학습한 커널 텐서를 살펴봅시다.**]

```{.python .input}
%%tab mxnet
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
%%tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

```{.python .input}
%%tab jax
params['params']['kernel'].reshape((1, 2))
```

실제로 학습된 커널 텐서는 저희가 앞서 정의한 커널 텐서 `K`와
놀라울 정도로 가깝습니다.

## 상호상관과 합성곱

상호상관과 합성곱 연산 사이의 대응에 관한
:numref:`sec_why-conv`의 저희의 관찰을 떠올려 보세요.
여기서는 2차원 합성곱 계층을 계속 고려해 봅시다.
만약 그러한 계층이 상호상관 대신
:eqref:`eq_2d-conv-discrete`에서 정의된 엄격한 합성곱 연산을
수행한다면 어떻게 될까요?
엄격한 *합성곱* 연산의 출력을 얻기 위해서는, 2차원 커널 텐서를 수평과 수직 모두로 뒤집은 다음, 입력 텐서와 *상호상관* 연산을 수행하기만 하면 됩니다.

딥러닝에서는 커널이 데이터로부터 학습되기 때문에,
그러한 계층이 엄격한 합성곱 연산이나
상호상관 연산 중 어느 것을 수행하든
합성곱 계층의 출력은 영향을 받지 않는다는 점에 주목할 만합니다.

이를 설명하기 위해, 합성곱 계층이 *상호상관*을 수행하고 :numref:`fig_correlation`의 커널을 학습한다고 가정해 봅시다. 이 커널을 행렬 $\mathbf{K}$로 표기하겠습니다.
다른 조건은 변경되지 않는다고 가정할 때,
이 계층이 대신 엄격한 *합성곱*을 수행할 때,
학습된 커널 $\mathbf{K}'$는 $\mathbf{K}'$가 수평과 수직
모두로 뒤집힌 후 $\mathbf{K}$와 같아질 것입니다.
즉,
합성곱 계층이
:numref:`fig_correlation`의 입력과 $\mathbf{K}'$에 대해
엄격한 *합성곱*을 수행할 때,
:numref:`fig_correlation`의 동일한 출력
(입력과 $\mathbf{K}$의 상호상관)이
얻어질 것입니다.

딥러닝 문헌의 표준 용어와 일관성을 유지하기 위해,
저희는 비록 엄격히 말하면 약간 다르지만,
상호상관 연산을 계속 합성곱이라고 부를 것입니다.
또한,
저희는 계층 표현이나 합성곱 커널을 표현하는 임의의 텐서의
항목(또는 구성 요소)을 지칭하기 위해 *원소(element)*라는 용어를 사용합니다.


## 특성 맵과 수용 영역

:numref:`subsec_why-conv-channels`에서 기술된 것처럼,
:numref:`fig_correlation`의 합성곱 계층 출력은
때때로 *특성 맵(feature map)*이라고 불리는데,
이는 후속 계층으로의 공간 차원(예: 너비와 높이)에서
학습된 표현(특성)으로 간주될 수 있기 때문입니다.
CNN에서는,
어떤 계층의 임의의 원소 $x$에 대해,
그것의 *수용 영역(receptive field)*은 순전파 동안
$x$의 계산에 영향을 미칠 수 있는
(이전 모든 계층의) 모든 원소를 지칭합니다.
수용 영역은
입력의 실제 크기보다 클 수 있다는 점에 주목하세요.

수용 영역을 설명하기 위해 :numref:`fig_correlation`을 계속 사용해 봅시다.
$2 \times 2$ 합성곱 커널이 주어졌을 때,
음영 처리된 출력 원소(값 $19$)의 수용 영역은
입력의 음영 처리된 부분의 네 원소입니다.
이제 $2 \times 2$ 출력을 $\mathbf{Y}$로 표기하고,
$\mathbf{Y}$를 입력으로 받아 단일 원소 $z$를 출력하는
추가적인 $2 \times 2$ 합성곱 계층을 가진 더 깊은 CNN을
고려해 봅시다.
이 경우,
$\mathbf{Y}$에서 $z$의 수용 영역은 $\mathbf{Y}$의 네 원소 모두를 포함하며,
한편
입력에서의 수용 영역은 아홉 입력 원소 모두를 포함합니다.
따라서,
특성 맵의 어떤 원소가
더 넓은 영역에 걸친 입력 특성을 검출하기 위해
더 큰 수용 영역이 필요할 때,
저희는 더 깊은 네트워크를 구축할 수 있습니다.


수용 영역은 신경생리학에서 그 이름이 유래되었습니다.
다양한 자극을 사용한 다양한 동물에 대한 일련의 실험
:cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968`은 그러한 자극에 대해 시각 피질이라고 불리는 것의 반응을
탐구했습니다. 대체로 그들은 하위 수준이 에지와 관련 형태에 반응한다는 것을
발견했습니다. 이후, :citet:`Field.1987`은 자연 이미지에 대해 이 효과를
합성곱 커널이라고밖에 부를 수 없는 것으로 보여주었습니다.
저희는 그 놀라운 유사성을 설명하기 위해 :numref:`field_visual`에 핵심 그림을 다시 게재합니다.

![:citet:`Field.1987`에서 가져온 그림과 캡션: 6개의 다른 채널을 가진 코딩의 예. (왼쪽) 각 채널과 연관된 6가지 유형의 센서 예. (오른쪽) (왼쪽)에 표시된 6개의 센서로 (가운데) 이미지의 합성곱. 개별 센서의 반응은 이러한 필터링된 이미지를 센서 크기에 비례하는 거리에서 샘플링하여 결정됩니다(점으로 표시). 이 도표는 짝수 대칭 센서의 반응만을 보여줍니다.](../img/field-visual.png)
:label:`field_visual`

알고 보면, 이 관계는 예를 들어 :citet:`Kuzovkin.Vicente.Petton.ea.2018`에서 보여준 것처럼, 이미지 분류 작업으로 훈련된 네트워크의 더 깊은 계층에 의해 계산된 특성에 대해서도 성립합니다. 합성곱은 생물학에서나 코드에서나 컴퓨터 비전을 위한 놀라울 정도로 강력한 도구임이 입증되었다고 말할 수 있을 것입니다. 따라서 그것들이 딥러닝의 최근 성공을 예고한 것은 (돌이켜 보면) 놀라운 일이 아닙니다.

## 요약

합성곱 계층에 필요한 핵심 계산은 상호상관 연산입니다. 저희는 그 값을 계산하는 데 단순한 중첩 for-루프만 있으면 된다는 것을 보았습니다. 다중 입력 및 다중 출력 채널이 있다면, 저희는 채널 간의 행렬(matrix)과 행렬(matrix) 연산을 수행하고 있는 것입니다. 보시다시피, 계산은 간단하고, 가장 중요하게는 매우 *국소적*입니다. 이는 상당한 하드웨어 최적화를 가능하게 하며, 컴퓨터 비전에서의 많은 최근 결과는 그 덕분에만 가능합니다. 결국, 이는 합성곱에 대한 최적화 시 칩 설계자가 메모리보다는 빠른 계산에 투자할 수 있다는 것을 의미합니다. 이는 다른 응용에 대해 최적의 설계로 이어지지 않을 수 있지만, 어디서나 볼 수 있고 저렴한 컴퓨터 비전의 문을 엽니다.

합성곱 자체에 관해서는, 예를 들어 에지와 선을 검출하거나, 이미지를 흐리게 하거나, 또는 선명하게 하는 등의 많은 목적으로 사용될 수 있습니다. 가장 중요하게는, 통계학자(또는 엔지니어)가 적합한 필터를 발명할 필요가 없다는 것입니다. 대신, 저희는 단순히 데이터로부터 그것들을 *학습*할 수 있습니다. 이는 특성 공학 휴리스틱을 증거 기반 통계로 대체합니다. 마지막으로, 그리고 꽤 기쁘게도, 이 필터들은 단지 심층 네트워크 구축에만 유리한 것이 아니라 뇌의 수용 영역과 특성 맵에도 대응합니다. 이는 저희가 올바른 방향에 있다는 자신감을 줍니다.

## 연습문제

1. 대각선 에지를 가진 이미지 `X`를 구성하세요.
    1. 이 절의 커널 `K`를 그것에 적용하면 어떻게 됩니까?
    1. `X`를 전치하면 어떻게 됩니까?
    1. `K`를 전치하면 어떻게 됩니까?
1. 일부 커널을 수동으로 설계하세요.
    1. 방향 벡터 $\mathbf{v} = (v_1, v_2)$가 주어졌을 때, $\mathbf{v}$에 직교하는 에지, 즉
       $(v_2, -v_1)$ 방향의 에지를 검출하는 에지 검출 커널을 도출하세요.
    1. 2차 도함수에 대한 유한 차분 연산자를 도출하세요. 그것과 연관된 합성곱 커널의 최소
       크기는 얼마입니까? 이미지의 어떤 구조가 그것에 가장 강하게 반응합니까?
    1. 흐림 커널을 어떻게 설계하시겠습니까? 왜 그러한 커널을 사용하고 싶을까요?
    1. $d$차 도함수를 얻기 위한 커널의 최소 크기는 얼마입니까?
1. 저희가 만든 `Conv2D` 클래스에 대해 자동으로 기울기를 찾으려고 할 때 어떤 종류의 오류 메시지가 보입니까?
1. 입력과 커널 텐서를 변경함으로써 상호상관 연산을 행렬 곱셈으로 어떻게 표현합니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/271)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17996)
:end_tab:
