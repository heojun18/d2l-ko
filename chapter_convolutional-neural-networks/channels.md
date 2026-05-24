```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 다중 입력 및 다중 출력 채널
:label:`sec_channels`

저희가 :numref:`subsec_why-conv-channels`에서 각 이미지를 구성하는 다중 채널
(예를 들어, 컬러 이미지는 빨강, 초록, 파랑의 양을 나타내기 위해 표준 RGB 채널을 가집니다)과
다중 채널에 대한 합성곱 계층을 기술했지만,
지금까지는 단지 단일 입력과 단일 출력 채널로 작업함으로써
저희의 모든 수치 예제를 단순화했습니다.
이는 저희의 입력, 합성곱 커널, 출력 각각을
2차원 텐서로 생각할 수 있도록 해주었습니다.

저희가 채널을 더할 때,
저희의 입력과 은닉 표현은
모두 3차원 텐서가 됩니다.
예를 들어, 각 RGB 입력 이미지는 $3\times h\times w$ 모양을 가집니다.
저희는 크기가 3인 이 축을 *채널* 차원이라고 부릅니다. 채널 개념은
CNN 자체만큼이나 오래되었습니다. 예를 들어 LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`도 그것을 사용합니다.
이 절에서, 저희는 다중 입력 및 다중 출력 채널을 가진
합성곱 커널을 더 깊이 살펴봅니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 다중 입력 채널

입력 데이터가 다중 채널을 포함할 때,
저희는 입력 데이터와 동일한 수의 입력 채널을 가진
합성곱 커널을 구성해야 합니다.
그래야 그것이 입력 데이터와 상호상관을 수행할 수 있습니다.
입력 데이터의 채널 수가 $c_\textrm{i}$라고 가정하면,
합성곱 커널의 입력 채널 수도 $c_\textrm{i}$여야 합니다. 만약 저희의 합성곱 커널의 윈도우 모양이 $k_\textrm{h}\times k_\textrm{w}$라면,
$c_\textrm{i}=1$일 때, 저희는 합성곱 커널을
단순히 $k_\textrm{h}\times k_\textrm{w}$ 모양의 2차원 텐서로 생각할 수 있습니다.

하지만, $c_\textrm{i}>1$일 때, 저희는 *모든* 입력 채널에 대해
$k_\textrm{h}\times k_\textrm{w}$ 모양의 텐서를 포함하는 커널이 필요합니다. 이 $c_\textrm{i}$개의 텐서를 함께 연결하면
$c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ 모양의 합성곱 커널이 산출됩니다.
입력과 합성곱 커널이 각각 $c_\textrm{i}$개의 채널을 가지므로,
저희는 각 채널에 대해 입력의 2차원 텐서와
합성곱 커널의 2차원 텐서에
상호상관 연산을 수행하고, $c_\textrm{i}$개의 결과를 함께 더해
(채널에 대해 합산하여)
2차원 텐서를 산출할 수 있습니다.
이것이 다중 채널 입력과 다중 입력 채널 합성곱 커널 사이의
2차원 상호상관의 결과입니다.

:numref:`fig_conv_multi_in`은 두 입력 채널을 가진 2차원 상호상관의
예제를 제공합니다.
음영 처리된 부분은 첫 번째 출력 요소와 출력 계산에 사용된
입력 및 커널 텐서 요소입니다:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![두 입력 채널을 가진 상호상관 계산.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


여기서 무슨 일이 일어나고 있는지 정말로 이해하기 위해,
저희는 (**다중 입력 채널을 가진 상호상관 연산을 직접 구현**)할 수 있습니다.
저희가 하는 모든 일은 채널당 상호상관 연산을 수행하고
그 다음 결과를 합산하는 것뿐임에 주목하세요.

```{.python .input}
%%tab mxnet, pytorch, jax
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
%%tab tensorflow
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

저희는 상호상관 연산의 (**출력을 검증하기 위해**)
:numref:`fig_conv_multi_in`의 값에 해당하는
입력 텐서 `X`와 커널 텐서 `K`를 구성할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## 다중 출력 채널
:label:`subsec_multi-output-channels`

입력 채널의 수에 관계없이,
지금까지 저희는 항상 하나의 출력 채널로 끝났습니다.
하지만, :numref:`subsec_why-conv-channels`에서 논의한 것처럼,
각 계층에서 다중 채널을 가지는 것이 필수적임이 드러납니다.
가장 인기 있는 신경망 아키텍처에서,
저희는 실제로 신경망의 더 깊은 곳으로 갈수록
채널 차원을 늘리며,
일반적으로 다운샘플링하여 공간 해상도를
더 큰 *채널 깊이(channel depth)*와 맞바꿉니다.
직관적으로, 각 채널을 다른 특성 집합에 반응하는 것으로
생각할 수 있습니다.
현실은 이것보다 조금 더 복잡합니다. 순진한 해석은
표현이 픽셀별 또는 채널별로 독립적으로 학습된다고 시사할 수 있습니다.
대신, 채널은 공동으로 유용하도록 최적화됩니다.
이는 단일 채널을 에지 검출기에 매핑하기보다는, 채널 공간에서 어떤 방향이 에지 검출에
대응한다는 것을 단순히 의미할 수 있다는 것입니다.

입력 채널 수와 출력 채널 수를 각각 $c_\textrm{i}$와 $c_\textrm{o}$로 표기하고,
커널의 높이와 너비를 $k_\textrm{h}$와 $k_\textrm{w}$로 표기합시다.
다중 채널을 가진 출력을 얻기 위해,
저희는 *모든* 출력 채널에 대해
$c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$ 모양의
커널 텐서를 생성할 수 있습니다.
저희는 그것들을 출력 채널 차원에서 연결하여,
합성곱 커널의 모양이
$c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$이 되도록 합니다.
상호상관 연산에서,
각 출력 채널에서의 결과는 해당 출력 채널에 대응하는
합성곱 커널로부터 계산되며,
입력 텐서의 모든 채널로부터 입력을 받습니다.

저희는 아래에 보인 대로 [**다중 채널의 출력을 계산하기 위한**]
상호상관 함수를 구현합니다.

```{.python .input}
%%tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

저희는 `K`에 대한 커널 텐서를 `K+1`과 `K+2`와 연결하여
세 출력 채널을 가진 사소한 합성곱 커널을 구성합니다.

```{.python .input}
%%tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

아래에서, 저희는 입력 텐서 `X`와 커널 텐서 `K`에
상호상관 연산을 수행합니다.
이제 출력은 세 채널을 포함합니다.
첫 번째 채널의 결과는
이전 입력 텐서 `X`와 다중 입력 채널,
단일 출력 채널 커널의 결과와 일치합니다.

```{.python .input}
%%tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ 합성곱 계층
:label:`subsec_1x1`

처음에는 [**$1 \times 1$ 합성곱**], 즉 $k_\textrm{h} = k_\textrm{w} = 1$이
별 의미가 없어 보입니다.
결국, 합성곱은 인접한 픽셀들을 상관시킵니다.
$1 \times 1$ 합성곱은 분명히 그렇게 하지 않습니다.
그럼에도 불구하고, 그것들은 때때로 복잡한 심층 네트워크의
설계에 포함되는 인기 있는 연산입니다 :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`.
그것이 실제로 무엇을 하는지 좀 더 자세히 살펴봅시다.

최소 윈도우가 사용되기 때문에,
$1\times 1$ 합성곱은 더 큰 합성곱 계층이
높이 및 너비 차원에서 인접한 원소들 사이의 상호작용으로
구성된 패턴을 인식하는 능력을 잃습니다.
$1\times 1$ 합성곱의 유일한 계산은
채널 차원에서 발생합니다.

:numref:`fig_conv_1x1`은 3개의 입력 채널과 2개의 출력 채널을 가진
$1\times 1$ 합성곱 커널을 사용한 상호상관 계산을
보여줍니다.
입력과 출력이 동일한 높이와 너비를 가진다는 점에 주목하세요.
출력의 각 원소는 입력 이미지의
*동일한 위치에* 있는 원소들의 선형 결합으로부터
도출됩니다.
$1\times 1$ 합성곱 계층을, $c_\textrm{i}$개의 대응하는 입력 값을 $c_\textrm{o}$개의 출력 값으로 변환하기 위해
모든 단일 픽셀 위치에 적용되는 완전 연결 계층을
구성하는 것으로 생각할 수 있습니다.
이것이 여전히 합성곱 계층이기 때문에,
가중치는 픽셀 위치에 걸쳐 묶여 있습니다.
따라서 $1\times 1$ 합성곱 계층은 $c_\textrm{o}\times c_\textrm{i}$개의 가중치를
필요로 합니다(편향을 더한 것). 또한 합성곱 계층 뒤에는 일반적으로
비선형성이 따른다는 점에 주목하세요. 이는 $1 \times 1$ 합성곱이 단순히 다른 합성곱에
접혀들지 않도록 보장합니다.

![3개의 입력 채널과 2개의 출력 채널을 가진 $1\times 1$ 합성곱 커널을 사용하는 상호상관 계산. 입력과 출력은 동일한 높이와 너비를 가집니다.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

이것이 실제로 작동하는지 확인해 봅시다.
저희는 완전 연결 계층을 사용하여
$1 \times 1$ 합성곱을 구현합니다.
유일한 것은 행렬 곱셈 전후에
데이터 모양에 약간의 조정이 필요하다는 것입니다.

```{.python .input}
%%tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

$1\times 1$ 합성곱을 수행할 때,
위 함수는 이전에 구현된 상호상관 함수 `corr2d_multi_in_out`와 동등합니다.
이를 일부 샘플 데이터로 확인해 봅시다.

```{.python .input}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

```{.python .input}
%%tab jax
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (3, 3, 3)) + 0 * 1
K = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 3, 1, 1)) + 0 * 1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## 논의

채널은 양쪽의 장점을 결합할 수 있도록 해줍니다. 즉, 상당한 비선형성을 허용하는 MLP와 특성의 *국소적인* 분석을 허용하는 합성곱입니다. 특히, 채널은 CNN이 에지와 형태 검출기와 같은 다중 특성을 동시에 추론할 수 있도록 해줍니다. 또한 이동 불변성과 국소성으로부터 발생하는 극적인 매개변수 감소와 컴퓨터 비전에서 표현력 있고 다양한 모델의 필요성 사이의 실용적인 트레이드오프를 제공합니다.

하지만, 이 유연성에는 대가가 따른다는 점에 주목하세요. $(h \times w)$ 크기의 이미지가 주어졌을 때, $k \times k$ 합성곱을 계산하는 비용은 $\mathcal{O}(h \cdot w \cdot k^2)$입니다. 각각 $c_\textrm{i}$개와 $c_\textrm{o}$개의 입력 및 출력 채널에 대해 이는 $\mathcal{O}(h \cdot w \cdot k^2 \cdot c_\textrm{i} \cdot c_\textrm{o})$로 증가합니다. $5 \times 5$ 커널과 각각 $128$개의 입력 및 출력 채널을 가진 $256 \times 256$ 픽셀 이미지의 경우, 이는 530억 개 이상의 연산에 해당합니다(저희는 곱셈과 덧셈을 별도로 셉니다). 나중에 저희는 비용을 줄이기 위한 효과적인 전략을 마주할 것입니다. 예를 들어, 채널별 연산을 블록 대각이 되도록 요구하여, ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`와 같은 아키텍처로 이어지는 것입니다.

## 연습문제

1. 각각 크기 $k_1$과 $k_2$의 두 합성곱 커널이 있다고 가정하세요
   (그 사이에 비선형성이 없음).
    1. 그 연산의 결과가 단일 합성곱으로 표현될 수 있음을 증명하세요.
    1. 등가의 단일 합성곱의 차원성은 무엇입니까?
    1. 그 역도 참입니까? 즉, 합성곱을 항상 두 개의 더 작은 합성곱으로 분해할 수 있습니까?
1. $c_\textrm{i}\times h\times w$ 모양의 입력과 $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$
   모양의 합성곱 커널, 그리고 $(p_\textrm{h}, p_\textrm{w})$의 패딩과 $(s_\textrm{h}, s_\textrm{w})$의 스트라이드를 가정하세요.
    1. 순전파의 계산 비용(곱셈과 덧셈)은 얼마입니까?
    1. 메모리 사용량은 얼마입니까?
    1. 역방향 계산의 메모리 사용량은 얼마입니까?
    1. 역전파의 계산 비용은 얼마입니까?
1. 입력 채널 수 $c_\textrm{i}$와 출력 채널 수 $c_\textrm{o}$를 모두 두 배로 늘리면 계산 횟수는
   어떤 인자만큼 증가합니까? 패딩을 두 배로 늘리면 어떻게 됩니까?
1. 이 절의 마지막 예제에서 변수 `Y1`과 `Y2`는 정확히 같습니까? 왜 그렇습니까?
1. 합성곱 윈도우가 $1 \times 1$이 아닐 때에도 합성곱을 행렬 곱셈으로 표현하세요.
1. 여러분의 과제는 $k \times k$ 커널로 빠른 합성곱을 구현하는 것입니다. 알고리즘 후보 중 하나는
   소스에 걸쳐 수평으로 스캔하여, $k$-너비 스트립을 읽고 한 번에 하나의 값으로 $1$-너비 출력 스트립을
   계산하는 것입니다. 대안은 $k + \Delta$ 너비 스트립을 읽고 $\Delta$-너비
   출력 스트립을 계산하는 것입니다. 왜 후자가 더 바람직합니까? $\Delta$를 얼마나 크게 선택해야 하는지에 대한 한계가 있습니까?
1. $c \times c$ 행렬이 있다고 가정하세요.
    1. 행렬이 $b$개의 블록으로 나뉠 때 블록 대각 행렬과 곱하는 것이 얼마나 더 빠릅니까?
    1. $b$개의 블록을 갖는 것의 단점은 무엇입니까? 적어도 부분적으로 어떻게 그것을 고칠 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/273)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17998)
:end_tab:

