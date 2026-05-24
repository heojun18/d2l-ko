```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 풀링
:label:`sec_pooling`

많은 경우 저희의 궁극적인 과제는 이미지에 관한 어떤 전역적인 질문을 던집니다.
예를 들어, *이미지에 고양이가 있는가?* 결과적으로, 저희의 최종 계층 유닛은
전체 입력에 민감해야 합니다.
정보를 점진적으로 집계하여 점점 더 거친 맵을 산출함으로써,
저희는 중간 처리 계층에서 합성곱 계층의 모든 이점을 유지하면서
궁극적으로 전역 표현을 학습한다는 이 목표를 달성합니다.
네트워크에서 더 깊이 들어갈수록,
각 은닉 노드가 민감한 (입력에 상대적인) 수용 영역은 더 커집니다.
공간 해상도를 줄이면
합성곱 커널이 더 큰 유효 영역을 덮으므로
이 과정이 가속화됩니다.

게다가, 에지와 같은 하위 수준 특성을 검출할 때
(:numref:`sec_conv_layer`에서 논의한 것처럼),
저희는 종종 저희의 표현이 이동에 대해 어느 정도 불변이기를 원합니다.
예를 들어, 흑백 사이에 뚜렷한 경계가 있는 이미지 `X`를 가져다가
전체 이미지를 오른쪽으로 한 픽셀 이동시키면,
즉 `Z[i, j] = X[i, j + 1]`로 하면,
새 이미지 `Z`에 대한 출력은 크게 다를 수 있습니다.
에지가 한 픽셀 이동되었을 것입니다.
실제로는, 객체가 정확히 동일한 장소에서 발생하는 경우는 거의 없습니다.
사실, 삼각대와 정지된 객체가 있더라도,
셔터 움직임으로 인한 카메라 진동으로
모든 것이 한 픽셀 정도 이동될 수 있습니다
(고급 카메라에는 이 문제를 해결하기 위한 특수 기능이 가득합니다).

이 절은 *풀링 계층(pooling layers)*을 소개합니다.
이는 합성곱 계층의 위치에 대한 민감도를 완화하고
표현을 공간적으로 다운샘플링하는
이중 목적을 수행합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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

## 최대 풀링 및 평균 풀링

합성곱 계층과 마찬가지로, *풀링* 연산자는
스트라이드에 따라 입력의 모든 영역에 걸쳐 미끄러지는
고정된 모양의 윈도우로 구성되며,
고정된 모양의 윈도우(때때로 *풀링 윈도우*라고 알려진)에 의해
횡단되는 각 위치에 대해 단일 출력을 계산합니다.
하지만, 합성곱 계층에서 입력과 커널의 상호상관 계산과 달리,
풀링 계층은 매개변수를 포함하지 않습니다 (*커널*이 없습니다).
대신, 풀링 연산자는 결정적이며,
일반적으로 풀링 윈도우 내의 원소들의 최댓값이나
평균값을 계산합니다.
이러한 연산은 각각 *최대 풀링(maximum pooling)* (줄여서 *맥스 풀링(max-pooling)*)과
*평균 풀링(average pooling)*이라고 불립니다.

*평균 풀링*은 본질적으로 CNN만큼이나 오래되었습니다. 그 아이디어는 이미지를
다운샘플링하는 것과 유사합니다. 저해상도 이미지를 위해 모든 두 번째(또는 세 번째)
픽셀의 값만 취하는 대신, 인접 픽셀에 대해 평균을 내어 여러 인접 픽셀의 정보를
결합하므로 더 나은 신호 대 잡음비를 가진
이미지를 얻을 수 있습니다. *맥스 풀링*은
:citet:`Riesenhuber.Poggio.1999`에서 객체 인식의 목적을 위해
정보 집계가 어떻게 계층적으로 집계될 수 있는지를 기술하기 위해 인지신경과학의 맥락에서
도입되었습니다. 음성 인식에는 이미 더 이른 버전이 있었습니다 :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`. 거의 모든 경우에, 맥스 풀링은 그렇게도 불리는데,
평균 풀링보다 선호됩니다.

두 경우 모두, 상호상관 연산자와 마찬가지로,
저희는 풀링 윈도우를 입력 텐서의 왼쪽 위에서 시작하여
왼쪽에서 오른쪽으로, 위에서 아래로 그것을 가로질러 미끄러지는 것으로
생각할 수 있습니다.
풀링 윈도우가 도달하는 각 위치에서,
맥스 풀링이 사용되는지 평균 풀링이 사용되는지에 따라
윈도우 내 입력 부분 텐서의 최댓값이나 평균값을 계산합니다.


![$2\times 2$ 모양의 풀링 윈도우를 가진 맥스 풀링. 음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 텐서 요소입니다: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling`의 출력 텐서는 높이 2와 너비 2를 가집니다.
네 원소는 각 풀링 윈도우에서의 최댓값으로부터 도출됩니다.

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

더 일반적으로, 저희는 해당 크기의 영역에 대해 집계함으로써
$p \times q$ 풀링 계층을 정의할 수 있습니다. 에지 검출 문제로 돌아가서,
저희는 합성곱 계층의 출력을 $2\times 2$ 맥스 풀링의 입력으로 사용합니다.
`X`를 합성곱 계층의 입력으로, `Y`를 풀링 계층의 출력으로 표기합시다.
`X[i, j]`, `X[i, j + 1]`, `X[i+1, j]`, `X[i+1, j + 1]`의 값이
다른지 여부에 관계없이,
풀링 계층은 항상 `Y[i, j] = 1`을 출력합니다.
즉, $2\times 2$ 맥스 풀링 계층을 사용하면,
합성곱 계층에 의해 인식된 패턴이 높이나 너비에서 한 원소를 넘지 않게
이동하는 경우 여전히 그것을 검출할 수 있습니다.

아래 코드에서, 저희는 `pool2d` 함수에서
(**풀링 계층의 순전파를 구현**)합니다.
이 함수는 :numref:`sec_conv_layer`의 `corr2d` 함수와 유사합니다.
하지만, 커널이 필요 없으며,
출력은 입력의 각 영역의 최댓값이나 평균값으로 계산됩니다.

```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab jax
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = jnp.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].max())
            elif mode == 'avg':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].mean())
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

저희는 [**2차원 맥스 풀링 계층의 출력을 검증하기 위해**] :numref:`fig_pooling`에서 입력 텐서 `X`를 구성할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

또한, 저희는 (**평균 풀링 계층**)으로 실험해 볼 수 있습니다.

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**패딩과 스트라이드**]

합성곱 계층과 마찬가지로, 풀링 계층도
출력 모양을 변경합니다.
그리고 이전과 마찬가지로, 저희는 입력을 패딩하고 스트라이드를 조정함으로써
원하는 출력 모양을 얻기 위해 연산을 조정할 수 있습니다.
저희는 딥러닝 프레임워크의 내장 2차원 맥스 풀링 계층을 통해
풀링 계층에서의 패딩과 스트라이드 사용을 시연할 수 있습니다.
저희는 먼저 예제의 수(배치 크기)와 채널 수가 모두 1인,
모양에 네 차원이 있는 입력 텐서 `X`를 구성합니다.

:begin_tab:`tensorflow`
다른 프레임워크와 달리, TensorFlow는
*channels-last* 입력을 선호하고 그것에 최적화되어 있다는 점에 주목하세요.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow, jax
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

풀링은 영역으로부터 정보를 집계하기 때문에, (**딥러닝 프레임워크는 풀링 윈도우 크기와 스트라이드를 일치시키는 것을 기본값으로 합니다.**) 예를 들어, 모양 `(3, 3)`의 풀링 윈도우를 사용하면 기본적으로 `(3, 3)`의 스트라이드 모양을 얻습니다.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab jax
# Pooling has no model parameters, hence it needs no initialization
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

말할 필요도 없이, 필요한 경우 프레임워크 기본값을 재정의하기 위해 [**스트라이드와 패딩을 수동으로 지정할 수 있습니다**].

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

물론, 저희는 아래 예제가 보여주는 것처럼 각각 임의의 높이와 너비를 가진 임의의 직사각형 풀링 윈도우를 지정할 수 있습니다.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
%%tab jax

X_padded = jnp.pad(X, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(2, 3), strides=(2, 3), padding='VALID')
```

## 다중 채널

다중 채널 입력 데이터를 처리할 때,
[**풀링 계층은 합성곱 계층에서처럼 채널에 걸쳐 입력을 합산하기보다,
각 입력 채널을 별도로 풀링**]합니다.
이는 풀링 계층의 출력 채널 수가
입력 채널 수와 같음을 의미합니다.
아래에서, 저희는 채널 차원에서 텐서 `X`와 `X + 1`을 연결하여
두 채널을 가진 입력을 구성할 것입니다.

:begin_tab:`tensorflow`
이는 channels-last 구문 때문에 TensorFlow에서는 마지막 차원을 따라
연결해야 한다는 점에 주목하세요.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# Concatenate along `dim=3` due to channels-last syntax
X = d2l.concat([X, X + 1], 3)
X
```

보시다시피, 풀링 후에도 출력 채널 수는 여전히 두 개입니다.

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

:begin_tab:`tensorflow`
TensorFlow 풀링의 출력은 언뜻 보기에는 다르게 나타나지만,
수치적으로는 MXNet 및 PyTorch와 동일한 결과가 제시된다는 점에 주목하세요.
차이는 차원성에 있으며, 출력을 수직으로 읽으면 다른 구현과
동일한 출력이 산출됩니다.
:end_tab:

## 요약

풀링은 매우 단순한 연산입니다. 이름이 나타내는 그대로의 일을 합니다. 값들의 윈도우에 대해 결과를 집계하는 것입니다. 스트라이드와 패딩과 같은 모든 합성곱 의미론은 이전과 같은 방식으로 적용됩니다. 풀링은 채널에 무관함, 즉 채널 수를 변경하지 않은 채로 두고 각 채널에 별도로 적용된다는 점에 주목하세요. 마지막으로, 두 가지 인기 있는 풀링 선택 중 맥스 풀링은 출력에 어느 정도의 불변성을 부여하므로 평균 풀링보다 선호됩니다. 인기 있는 선택은 출력의 공간 해상도를 4분의 1로 만들기 위해 $2 \times 2$의 풀링 윈도우 크기를 고르는 것입니다.

풀링 이외에도 해상도를 줄이는 더 많은 방법이 있다는 점에 주목하세요. 예를 들어, 확률적 풀링(stochastic pooling) :cite:`Zeiler.Fergus.2013`과 분수 맥스 풀링(fractional max-pooling) :cite:`Graham.2014`에서 집계는 무작위화와 결합됩니다. 이는 어떤 경우에는 정확도를 약간 향상시킬 수 있습니다. 마지막으로, 저희가 어텐션 메커니즘에서 나중에 볼 것처럼, 예를 들어 쿼리와 표현 벡터 사이의 정렬을 사용하는 등 출력에 걸쳐 집계하는 더 정교한 방법이 있습니다.


## 연습문제

1. 합성곱을 통해 평균 풀링을 구현하세요.
1. 맥스 풀링이 합성곱만으로 구현될 수 없음을 증명하세요.
1. 맥스 풀링은 ReLU 연산을 사용하여, 즉 $\textrm{ReLU}(x) = \max(0, x)$로 달성될 수 있습니다.
    1. ReLU 연산만 사용하여 $\max (a, b)$를 표현하세요.
    1. 이것을 사용하여 합성곱과 ReLU 계층을 통해 맥스 풀링을 구현하세요.
    1. $2 \times 2$ 합성곱에 얼마나 많은 채널과 계층이 필요합니까? $3 \times 3$ 합성곱에는 얼마나 많이 필요합니까?
1. 풀링 계층의 계산 비용은 얼마입니까? 풀링 계층의 입력이 $c\times h\times w$ 크기이고, 풀링 윈도우가 $(p_\textrm{h}, p_\textrm{w})$의 패딩과 $(s_\textrm{h}, s_\textrm{w})$의 스트라이드를 가진 $p_\textrm{h}\times p_\textrm{w}$의 모양이라고 가정하세요.
1. 왜 맥스 풀링과 평균 풀링이 다르게 작동할 것으로 기대합니까?
1. 별도의 최소 풀링 계층이 필요합니까? 그것을 다른 연산으로 대체할 수 있습니까?
1. 저희는 풀링을 위해 소프트맥스 연산을 사용할 수 있을 것입니다. 왜 그것이 그렇게 인기가 없을 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/274)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17999)
:end_tab:

