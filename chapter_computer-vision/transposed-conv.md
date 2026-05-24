# 전치 합성곱
:label:`sec_transposed_conv`

지금까지 저희가 본 CNN 계층, 예를 들어 합성곱 계층(:numref:`sec_conv_layer`)과 풀링 계층(:numref:`sec_pooling`)은 일반적으로 입력의 공간 차원(높이와 너비)을 줄이거나(다운샘플링) 변경하지 않은 채로 유지합니다.
픽셀 수준에서 분류하는 시맨틱 분할에서, 입력과 출력의 공간 차원이 동일하면 편리할 것입니다.
예를 들어, 한 출력 픽셀의 채널 차원이 같은 공간적 위치에 있는 입력 픽셀에 대한 분류 결과를 가질 수 있습니다.


이를 달성하기 위해, 특히 공간 차원이 CNN 계층에 의해 줄어든 후에, 저희는 중간 특징 맵의 공간 차원을 늘릴(업샘플링) 수 있는 또 다른 유형의 CNN 계층을 사용할 수 있습니다.
이 절에서, 저희는 합성곱에 의한 다운샘플링 연산을 역으로 하기 위해 *분수 스트라이드 합성곱(fractionally-strided convolution)* :cite:`Dumoulin.Visin.2016`이라고도 불리는 *전치 합성곱*을 소개할 것입니다.

```{.python .input}
#@tab mxnet
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## 기본 연산

지금은 채널을 무시하고, 스트라이드가 1이고 패딩이 없는 기본 전치 합성곱 연산부터 시작해 봅시다.
$n_h \times n_w$ 입력 텐서와 $k_h \times k_w$ 커널이 주어졌다고 가정합니다.
커널 윈도우를 스트라이드 1로 각 행에서 $n_w$번, 각 열에서 $n_h$번 이동시키면 총 $n_h n_w$개의 중간 결과가 생성됩니다.
각 중간 결과는 0으로 초기화된 $(n_h + k_h - 1) \times (n_w + k_w - 1)$ 텐서입니다.
각 중간 텐서를 계산하기 위해, 입력 텐서의 각 원소는 커널과 곱해져서 결과로 나온 $k_h \times k_w$ 텐서가 각 중간 텐서의 한 부분을 대체합니다.
각 중간 텐서에서 대체된 부분의 위치는 계산에 사용된 입력 텐서의 원소의 위치에 해당한다는 점에 유의하세요.
마지막으로, 모든 중간 결과가 합산되어 출력을 생성합니다.

예제로, :numref:`fig_trans_conv`는 $2\times 2$ 입력 텐서에 대해 $2\times 2$ 커널을 사용한 전치 합성곱이 어떻게 계산되는지 설명합니다.


![$2\times 2$ 커널을 사용한 전치 합성곱. 음영 처리된 부분은 중간 텐서의 한 부분이자 계산에 사용된 입력 및 커널 텐서 원소입니다.](../img/trans_conv.svg)
:label:`fig_trans_conv`


저희는 입력 행렬 `X`와 커널 행렬 `K`에 대해 (**이 기본 전치 합성곱 연산**) `trans_conv`를 구현할 수 있습니다.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

커널을 통해 입력 원소를 *줄이는* 일반 합성곱(:numref:`sec_conv_layer`)과는 달리, 전치 합성곱은 커널을 통해 입력 원소를 *방송*하여, 입력보다 큰 출력을 생성합니다.
저희는 :numref:`fig_trans_conv`에서 입력 텐서 `X`와 커널 텐서 `K`를 구성해 기본 2차원 전치 합성곱 연산의 [**위 구현의 출력을 검증**]할 수 있습니다.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

또는, 입력 `X`와 커널 `K`가 모두 4차원 텐서일 때, 저희는 [**고수준 API를 사용해 같은 결과를 얻을**] 수 있습니다.

```{.python .input}
#@tab mxnet
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**패딩, 스트라이드, 다중 채널**]

패딩이 입력에 적용되는 일반 합성곱과는 달리, 전치 합성곱에서는 패딩이 출력에 적용됩니다.
예를 들어, 높이와 너비 양쪽 면에 패딩 숫자를 1로 지정하면, 첫 번째와 마지막 행과 열이 전치 합성곱 출력에서 제거됩니다.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

전치 합성곱에서, 스트라이드는 입력이 아니라 중간 결과(따라서 출력)에 대해 지정됩니다.
:numref:`fig_trans_conv`와 같은 입력 및 커널 텐서를 사용해, 스트라이드를 1에서 2로 바꾸면 중간 텐서의 높이와 너비가 모두 증가하고, 따라서 :numref:`fig_trans_conv_stride2`의 출력 텐서가 됩니다.


![스트라이드 2의 $2\times 2$ 커널을 사용한 전치 합성곱. 음영 처리된 부분은 중간 텐서의 한 부분이자 계산에 사용된 입력 및 커널 텐서 원소입니다.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`



다음 코드 조각은 :numref:`fig_trans_conv_stride2`에서 스트라이드 2에 대한 전치 합성곱 출력을 검증할 수 있습니다.

```{.python .input}
#@tab mxnet
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

다중 입력 및 출력 채널의 경우, 전치 합성곱은 일반 합성곱과 같은 방식으로 작동합니다.
입력이 $c_i$개의 채널을 가지고, 전치 합성곱이 각 입력 채널에 $k_h\times k_w$ 커널 텐서를 할당한다고 가정합니다.
다중 출력 채널이 지정되면, 저희는 각 출력 채널에 대해 $c_i\times k_h\times k_w$ 커널을 가지게 됩니다.


요컨대, 저희가 $\mathsf{X}$를 합성곱 계층 $f$에 입력해 $\mathsf{Y}=f(\mathsf{X})$를 출력하고, 출력 채널 수가 $\mathsf{X}$의 채널 수라는 점을 제외하고는 $f$와 동일한 하이퍼파라미터로 전치 합성곱 계층 $g$를 생성하면, $g(Y)$는 $\mathsf{X}$와 같은 형태를 가질 것입니다.
이는 다음 예제에서 설명될 수 있습니다.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**행렬 전치와의 연결**]
:label:`subsec-connection-to-mat-transposition`

전치 합성곱은 행렬 전치의 이름을 따서 명명되었습니다.
설명하자면, 먼저 행렬 곱셈을 사용해 합성곱을 어떻게 구현하는지 보겠습니다.
아래 예제에서, 저희는 $3\times 3$ 입력 `X`와 $2\times 2$ 합성곱 커널 `K`를 정의한 다음, `corr2d` 함수를 사용해 합성곱 출력 `Y`를 계산합니다.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

다음으로, 저희는 합성곱 커널 `K`를 많은 0을 포함하는 희소 가중치 행렬 `W`로 다시 작성합니다.
가중치 행렬의 형태는 ($4$, $9$)이며, 여기서 0이 아닌 원소는 합성곱 커널 `K`에서 옵니다.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

입력 `X`를 행별로 연결해 길이 9의 벡터를 얻습니다. 그러면 `W`와 벡터화된 `X`의 행렬 곱셈이 길이 4의 벡터를 제공합니다.
이를 형태 변경한 후, 저희는 위의 원래 합성곱 연산에서 같은 결과 `Y`를 얻을 수 있습니다. 저희는 방금 행렬 곱셈을 사용해 합성곱을 구현했습니다.

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

마찬가지로, 저희는 행렬 곱셈을 사용해 전치 합성곱을 구현할 수 있습니다.
다음 예제에서, 저희는 위의 일반 합성곱의 $2 \times 2$ 출력 `Y`를 전치 합성곱의 입력으로 가져옵니다.
행렬을 곱해 이 연산을 구현하기 위해, 저희는 새 형태 $(9, 4)$의 가중치 행렬 `W`를 전치하기만 하면 됩니다.

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

행렬을 곱해 합성곱을 구현하는 것을 생각해 보세요.
입력 벡터 $\mathbf{x}$와 가중치 행렬 $\mathbf{W}$가 주어졌을 때, 합성곱의 순전파 함수는 입력을 가중치 행렬과 곱해 벡터 $\mathbf{y}=\mathbf{W}\mathbf{x}$를 출력함으로써 구현될 수 있습니다.
역전파는 연쇄 법칙(chain rule)을 따르고 $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$이므로, 합성곱의 역전파 함수는 입력을 전치된 가중치 행렬 $\mathbf{W}^\top$과 곱함으로써 구현될 수 있습니다.
따라서, 전치 합성곱 계층은 합성곱 계층의 순전파 함수와 역전파 함수를 교환하기만 하면 됩니다. 그 순전파와 역전파 함수는 각각 입력 벡터를 $\mathbf{W}^\top$과 $\mathbf{W}$와 곱합니다.


## 요약

* 커널을 통해 입력 원소를 줄이는 일반 합성곱과는 달리, 전치 합성곱은 커널을 통해 입력 원소를 방송하여, 입력보다 큰 출력을 생성합니다.
* 저희가 $\mathsf{X}$를 합성곱 계층 $f$에 입력해 $\mathsf{Y}=f(\mathsf{X})$를 출력하고, 출력 채널 수가 $\mathsf{X}$의 채널 수라는 점을 제외하고는 $f$와 동일한 하이퍼파라미터로 전치 합성곱 계층 $g$를 생성하면, $g(Y)$는 $\mathsf{X}$와 같은 형태를 가질 것입니다.
* 저희는 행렬 곱셈을 사용해 합성곱을 구현할 수 있습니다. 전치 합성곱 계층은 합성곱 계층의 순전파 함수와 역전파 함수를 교환하기만 하면 됩니다.


## 연습문제

1. :numref:`subsec-connection-to-mat-transposition`에서, 합성곱 입력 `X`와 전치 합성곱 출력 `Z`는 같은 형태를 가집니다. 그들이 같은 값을 가지나요? 왜 그런가요?
1. 합성곱을 구현하기 위해 행렬 곱셈을 사용하는 것이 효율적인가요? 왜 그런가요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:
