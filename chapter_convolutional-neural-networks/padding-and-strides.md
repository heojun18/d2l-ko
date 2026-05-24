```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 패딩과 스트라이드
:label:`sec_padding`

:numref:`fig_correlation`에서의 합성곱 예제를 떠올려 보세요.
입력은 높이와 너비가 모두 3이었고
합성곱 커널은 높이와 너비가 모두 2였으며,
$2\times2$ 차원의 출력 표현을 산출했습니다.
입력 모양이 $n_\textrm{h}\times n_\textrm{w}$이고
합성곱 커널 모양이 $k_\textrm{h}\times k_\textrm{w}$라고 가정할 때,
출력 모양은 $(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1)$이 됩니다.
저희는 합성곱을 적용할 픽셀이 다 떨어질 때까지만
합성곱 커널을 이동시킬 수 있습니다.

다음에서 저희는 출력 크기를 더 잘 제어할 수 있는
패딩과 스트라이드 합성곱을 포함한
여러 기법을 탐색할 것입니다.
동기로서, 커널은 일반적으로 너비와 높이가 $1$보다 크기 때문에,
많은 연속적인 합성곱을 적용한 후에는
입력보다 상당히 작은 출력을 얻게 되는 경향이 있다는 점에
주목하세요.
$240 \times 240$ 픽셀 이미지로 시작한다면,
$5 \times 5$ 합성곱의 열 개 계층은 이미지를
$200 \times 200$ 픽셀로 줄여,
이미지의 $30 \%$를 잘라내고 그와 함께
원본 이미지 경계의 흥미로운 정보를 모두 지워버립니다.
*패딩*은 이 문제를 처리하기 위한 가장 인기 있는 도구입니다.
다른 경우에는, 원본 입력 해상도가 다루기 어렵다고 판단되는 경우 등,
차원성을 극적으로 줄이고 싶을 수도 있습니다.
*스트라이드 합성곱(strided convolution)*은 이러한 경우에 도움이 될 수 있는 인기 있는 기법입니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 패딩

위에서 기술한 대로, 합성곱 계층을 적용할 때 까다로운 한 가지 문제는
이미지 둘레의 픽셀을 잃는 경향이 있다는 것입니다. 합성곱 커널 크기와 이미지 내 위치에 대한 함수로서의 픽셀 활용을 묘사한 :numref:`img_conv_reuse`를 고려해 보세요. 모서리의 픽셀들은 거의 사용되지 않습니다.

![각각 $1 \times 1$, $2 \times 2$, $3 \times 3$ 크기의 합성곱에 대한 픽셀 활용.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

저희는 일반적으로 작은 커널을 사용하기 때문에,
주어진 임의의 합성곱에 대해
몇 개의 픽셀만 잃을 수 있지만,
많은 연속적인 합성곱 계층을 적용함에 따라
이는 누적될 수 있습니다.
이 문제에 대한 한 가지 직관적인 해결책은
저희 입력 이미지의 경계 주변에 채움용 픽셀을 추가하여,
이미지의 유효 크기를 늘리는 것입니다.
일반적으로, 저희는 추가 픽셀의 값을 0으로 설정합니다.
:numref:`img_conv_pad`에서, 저희는 $3 \times 3$ 입력을 패딩하여
그 크기를 $5 \times 5$로 늘립니다.
대응하는 출력은 그러면 $4 \times 4$ 행렬로 늘어납니다.
음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times0+0\times1+0\times2+0\times3=0$.

![패딩을 사용한 2차원 상호상관.](../img/conv-pad.svg)
:label:`img_conv_pad`

일반적으로, 총 $p_\textrm{h}$ 행의 패딩
(대략 절반은 위쪽, 절반은 아래쪽)과
총 $p_\textrm{w}$ 열의 패딩
(대략 절반은 왼쪽, 절반은 오른쪽)을 추가하면,
출력 모양은 다음과 같이 됩니다.

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$

이는 출력의 높이와 너비가
각각 $p_\textrm{h}$와 $p_\textrm{w}$만큼 늘어남을 의미합니다.

많은 경우, 저희는 입력과 출력에 동일한 높이와 너비를 주기 위해
$p_\textrm{h}=k_\textrm{h}-1$과 $p_\textrm{w}=k_\textrm{w}-1$로 설정하고 싶을 것입니다.
이는 네트워크를 구성할 때 각 계층의 출력 모양을 예측하는 것을
더 쉽게 만들 것입니다.
여기서 $k_\textrm{h}$가 홀수라고 가정하면,
저희는 높이의 양쪽에 각각 $p_\textrm{h}/2$ 행을 패딩할 것입니다.
$k_\textrm{h}$가 짝수라면, 한 가지 가능성은
입력의 위쪽에 $\lceil p_\textrm{h}/2\rceil$ 행을,
아래쪽에 $\lfloor p_\textrm{h}/2\rfloor$ 행을 패딩하는 것입니다.
저희는 너비의 양쪽도 같은 방식으로 패딩할 것입니다.

CNN은 일반적으로 1, 3, 5, 7과 같이 홀수의 높이와 너비 값을 가진
합성곱 커널을 사용합니다.
홀수 커널 크기를 선택하는 것은
위쪽과 아래쪽에 동일한 수의 행으로,
그리고 왼쪽과 오른쪽에 동일한 수의 열로 패딩하면서
차원성을 유지할 수 있다는 이점이 있습니다.

게다가, 차원성을 정확히 유지하기 위해 홀수 커널과 패딩을 사용하는
이 관행은 사무적 이점도 제공합니다.
임의의 2차원 텐서 `X`에 대해,
커널의 크기가 홀수이고
모든 측면에서 패딩 행과 열의 수가 같을 때,
입력과 동일한 높이와 너비를 가진 출력을 생성하므로,
저희는 출력 `Y[i, j]`가 `X[i, j]`를 중심으로 한
윈도우로 입력과 합성곱 커널의 상호상관에 의해
계산된다는 것을 압니다.

다음 예제에서, 저희는 높이와 너비가 3인 2차원 합성곱 계층을 생성하고
(**모든 측면에 1픽셀의 패딩을 적용**)합니다.
높이와 너비가 8인 입력이 주어졌을 때,
저희는 출력의 높이와 너비도 8임을 발견합니다.

```{.python .input}
%%tab mxnet
# We define a helper function to calculate convolutions. It initializes 
# the convolutional layer weights and performs corresponding dimensionality 
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# We define a helper function to calculate convolutions. It initializes the
# convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# We define a helper function to calculate convolutions. It initializes
# the convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return tf.reshape(Y, Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# We define a helper function to calculate convolutions. It initializes
# the convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, X.shape, 1) indicates that batch size and the number of channels are both 1
    key = jax.random.PRNGKey(d2l.get_seed())
    X = X.reshape((1,) + X.shape + (1,))
    Y, _ = conv2d.init_with_output(key, X)
    # Strip the dimensions: examples and channels
    return Y.reshape(Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

합성곱 커널의 높이와 너비가 다를 때,
저희는 [**높이와 너비에 다른 패딩 수를 설정**]함으로써
출력과 입력이 동일한 높이와 너비를 가지도록 할 수 있습니다.

```{.python .input}
%%tab mxnet
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
# We use a convolution kernel with height 5 and width 3. The padding on either
# side of the height and width are 2 and 1, respectively
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = nn.Conv(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## 스트라이드

상호상관을 계산할 때,
저희는 입력 텐서의 왼쪽 위 모서리에서 합성곱 윈도우로 시작하여,
그 다음 모든 위치에 걸쳐 아래쪽과 오른쪽 모두로 그것을 미끄러뜨립니다.
이전 예제에서, 저희는 기본적으로 한 번에 한 원소씩 미끄러뜨렸습니다.
하지만 때로는, 계산 효율성을 위해서나
다운샘플링하고 싶기 때문에,
저희는 윈도우를 한 번에 두 원소 이상 이동시켜
중간 위치를 건너뜁니다. 이는 합성곱 커널이
큰 경우, 기저 이미지의 큰 영역을 포착하므로 특히 유용합니다.

저희는 슬라이드당 횡단되는 행과 열의 수를 *스트라이드(stride)*라고 부릅니다.
지금까지, 저희는 높이와 너비 모두에 대해 1의 스트라이드를 사용해 왔습니다.
때로는, 저희는 더 큰 스트라이드를 사용하고 싶을 수도 있습니다.
:numref:`img_conv_stride`는 수직으로 3, 수평으로 2의 스트라이드를 가진
2차원 상호상관 연산을 보여줍니다.
음영 처리된 부분은 출력 요소와 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
첫 번째 열의 두 번째 원소가 생성될 때,
합성곱 윈도우가 세 행 아래로 미끄러진다는 것을 볼 수 있습니다.
첫 번째 행의 두 번째 원소가 생성될 때
합성곱 윈도우는 오른쪽으로 두 열 미끄러집니다.
합성곱 윈도우가 입력에서 계속해서 오른쪽으로 두 열 미끄러지면,
입력 원소가 윈도우를 채울 수 없기 때문에 출력이 없습니다
(다른 열의 패딩을 추가하지 않는 한).

![각각 높이와 너비에 대해 3과 2의 스트라이드를 가진 상호상관.](../img/conv-stride.svg)
:label:`img_conv_stride`

일반적으로, 높이에 대한 스트라이드가 $s_\textrm{h}$이고
너비에 대한 스트라이드가 $s_\textrm{w}$일 때, 출력 모양은 다음과 같습니다.

$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$

만약 저희가 $p_\textrm{h}=k_\textrm{h}-1$과 $p_\textrm{w}=k_\textrm{w}-1$로 설정한다면,
출력 모양은
$\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$로 단순화될 수 있습니다.
한 단계 더 나아가, 입력 높이와 너비가
높이와 너비의 스트라이드로 나누어떨어진다면,
출력 모양은 $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$이 될 것입니다.

아래에서, 저희는 [**높이와 너비 모두에 대해 스트라이드를 2로 설정**]하여,
입력 높이와 너비를 절반으로 만듭니다.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 3), padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

(**조금 더 복잡한 예제**)를 살펴봅시다.

```{.python .input}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## 요약 및 논의

패딩은 출력의 높이와 너비를 늘릴 수 있습니다. 이는 출력의 바람직하지 않은 수축을 피하기 위해 출력에 입력과 동일한 높이와 너비를 부여하는 데 자주 사용됩니다. 게다가, 그것은 모든 픽셀이 동등하게 자주 사용되도록 보장합니다. 일반적으로 저희는 입력 높이와 너비의 양쪽에 대칭 패딩을 선택합니다. 이 경우 저희는 $(p_\textrm{h}, p_\textrm{w})$ 패딩이라고 지칭합니다. 가장 흔하게는 $p_\textrm{h} = p_\textrm{w}$로 설정하며, 이 경우 저희는 단순히 패딩 $p$를 선택했다고 말합니다.

스트라이드에도 비슷한 관례가 적용됩니다. 수평 스트라이드 $s_\textrm{h}$와 수직 스트라이드 $s_\textrm{w}$가 일치할 때, 저희는 단순히 스트라이드 $s$에 대해 이야기합니다. 스트라이드는 출력의 해상도를 줄일 수 있으며, 예를 들어 $n > 1$일 때 출력의 높이와 너비를 입력의 높이와 너비의 $1/n$으로만 줄입니다. 기본적으로, 패딩은 0이고 스트라이드는 1입니다.

지금까지 저희가 논의한 모든 패딩은 단순히 이미지를 0으로 확장했습니다. 이것을 달성하기는 쉽기 때문에 상당한 계산상의 이점이 있습니다. 게다가, 연산자는 추가 메모리를 할당할 필요 없이 이 패딩을 암시적으로 활용하도록 설계될 수 있습니다. 동시에, 그것은 CNN이 "여백"이 어디 있는지를 단순히 학습함으로써 이미지 내에서 암시적 위치 정보를 인코딩할 수 있도록 허용합니다. 0-패딩의 많은 대안이 있습니다. :citet:`Alsallakh.Kokhlikyan.Miglani.ea.2020`은 이러한 것들에 대한 광범위한 개요를 제공했습니다(다만 아티팩트가 발생하지 않는 한 0이 아닌 패딩을 언제 사용해야 하는지에 대한 명확한 사례는 없습니다).


## 연습문제

1. 이 절의 마지막 코드 예제에서 커널 크기 $(3, 5)$, 패딩 $(0, 1)$, 스트라이드 $(3, 4)$가 주어졌을 때,
   실험 결과와 일치하는지 확인하기 위해 출력 모양을 계산하세요.
1. 오디오 신호에서 2의 스트라이드는 무엇에 대응합니까?
1. 미러 패딩, 즉 경계 값이 텐서를 확장하기 위해 단순히 거울처럼 비춰지는 패딩을 구현하세요.
1. 1보다 큰 스트라이드의 계산적 이점은 무엇입니까?
1. 1보다 큰 스트라이드의 통계적 이점은 무엇일 수 있습니까?
1. $\frac{1}{2}$의 스트라이드를 어떻게 구현하시겠습니까? 그것은 무엇에 대응합니까? 언제 이것이 유용할까요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/272)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17997)
:end_tab:
