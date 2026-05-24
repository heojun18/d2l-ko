```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 셀프 어텐션과 위치 인코딩
:label:`sec_self-attention-and-positional-encoding`

딥러닝에서 저희는 종종 시퀀스를 인코딩하기 위해 CNN이나 RNN을 사용합니다.
이제 어텐션 메커니즘을 염두에 두고, 토큰 시퀀스를 어텐션 메커니즘에 공급하여 매 단계마다 각 토큰이 자신만의 쿼리, 키, 값을 갖도록 한다고 상상해 보십시오.
여기서 다음 계층에서 토큰의 표현 값을 계산할 때, 토큰은 (그것의 쿼리 벡터를 통해) 다른 어떤 토큰에도 (그들의 키 벡터에 기반해 매칭함으로써) 주의를 기울일 수 있습니다.
쿼리-키 호환성 점수의 전체 집합을 사용해, 저희는 각 토큰에 대해 다른 토큰들에 대한 적절한 가중합을 만듦으로써 표현을 계산할 수 있습니다.
모든 토큰이 서로의 토큰에 주의를 기울이기 때문에(디코더 단계가 인코더 단계에 주의를 기울이는 경우와 달리), 그러한 아키텍처는 일반적으로 *셀프 어텐션(self-attention)* 모델 :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017` 로 묘사되며, 다른 곳에서는 *내부 어텐션(intra-attention)* 모델 :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017` 로 묘사됩니다.
이 절에서 저희는 시퀀스 순서에 대한 추가 정보를 사용하는 것을 포함하여, 셀프 어텐션을 사용한 시퀀스 인코딩을 논의할 것입니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## [**셀프 어텐션**]

임의의 $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$) 인 입력 토큰 시퀀스 $\mathbf{x}_1, \ldots, \mathbf{x}_n$ 이 주어졌을 때, 그것의 셀프 어텐션은 같은 길이의 시퀀스 $\mathbf{y}_1, \ldots, \mathbf{y}_n$ 을 출력하는데, 여기서

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

이는 :eqref:`eq_attention_pooling` 의 어텐션 풀링 정의에 따른 것입니다.
멀티헤드 어텐션을 사용해, 다음 코드 스니펫은 (배치 크기, 시간 단계 수 또는 토큰 단위 시퀀스 길이, $d$) 모양의 텐서의 셀프 어텐션을 계산합니다.
출력 텐서는 같은 모양을 가집니다.

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, X, X, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## CNN, RNN, 셀프 어텐션 비교
:label:`subsec_cnn-rnn-self-attention`

$n$ 개 토큰의 시퀀스를 같은 길이의 다른 시퀀스에 매핑하기 위한 아키텍처를 비교해 봅시다. 여기서 각 입력 또는 출력 토큰은 $d$ 차원 벡터로 표현됩니다.
구체적으로 저희는 CNN, RNN, 셀프 어텐션을 고려할 것입니다.
저희는 그들의 계산 복잡도, 순차 연산, 최대 경로 길이를 비교할 것입니다.
순차 연산은 병렬 계산을 방해하는 반면, 시퀀스 위치의 어떤 조합 간의 더 짧은 경로는 시퀀스 내의 장거리 의존성을 학습하기 더 쉽게 한다는 점에 유의하십시오 :cite:`Hochreiter.Bengio.Frasconi.ea.2001`.


![CNN(패딩 토큰은 생략됨), RNN, 셀프 어텐션 아키텍처 비교.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`



어떤 텍스트 시퀀스든 "1차원 이미지"로 간주합시다. 마찬가지로 1차원 CNN은 텍스트의 $n$-그램 같은 국소 특성을 처리할 수 있습니다.
길이 $n$ 의 시퀀스가 주어졌을 때, 커널 크기가 $k$ 이고 입력 및 출력 채널 수가 모두 $d$ 인 합성곱 계층을 고려합시다.
합성곱 계층의 계산 복잡도는 $\mathcal{O}(knd^2)$ 입니다.
:numref:`fig_cnn-rnn-self-attention` 가 보여주듯이, CNN은 계층적이므로 $\mathcal{O}(1)$ 순차 연산이 있고 최대 경로 길이는 $\mathcal{O}(n/k)$ 입니다.
예를 들어 $\mathbf{x}_1$ 과 $\mathbf{x}_5$ 는 :numref:`fig_cnn-rnn-self-attention` 의 커널 크기 3을 가진 2층 CNN의 수용 영역(receptive field) 내에 있습니다.

RNN의 은닉 상태를 업데이트할 때, $d \times d$ 가중치 행렬과 $d$ 차원 은닉 상태의 곱셈은 $\mathcal{O}(d^2)$ 의 계산 복잡도를 가집니다.
시퀀스 길이가 $n$ 이므로, 순환 계층의 계산 복잡도는 $\mathcal{O}(nd^2)$ 입니다.
:numref:`fig_cnn-rnn-self-attention` 에 따르면, 병렬화될 수 없는 $\mathcal{O}(n)$ 순차 연산이 있고 최대 경로 길이도 $\mathcal{O}(n)$ 입니다.

셀프 어텐션에서, 쿼리, 키, 값은 모두 $n \times d$ 행렬입니다.
:eqref:`eq_softmax_QK_V` 의 스케일드 내적 어텐션을 고려합시다. 여기서 $n \times d$ 행렬이 $d \times n$ 행렬과 곱해진 다음, 출력 $n \times n$ 행렬이 $n \times d$ 행렬과 곱해집니다.
결과적으로 셀프 어텐션은 $\mathcal{O}(n^2d)$ 계산 복잡도를 가집니다.
:numref:`fig_cnn-rnn-self-attention` 에서 볼 수 있듯이, 각 토큰은 셀프 어텐션을 통해 다른 어떤 토큰과도 직접 연결됩니다.
따라서 계산은 $\mathcal{O}(1)$ 순차 연산으로 병렬일 수 있고 최대 경로 길이도 $\mathcal{O}(1)$ 입니다.

종합하면, CNN과 셀프 어텐션 모두 병렬 계산의 이점을 누리고 셀프 어텐션은 가장 짧은 최대 경로 길이를 가집니다.
그러나 시퀀스 길이에 대한 이차 계산 복잡도는 매우 긴 시퀀스에 대해 셀프 어텐션을 엄청나게 느리게 만듭니다.




## [**위치 인코딩**]
:label:`subsec_positional-encoding`


시퀀스의 토큰을 하나씩 반복적으로 처리하는 RNN과 달리, 셀프 어텐션은 병렬 계산을 위해 순차 연산을 버립니다.
셀프 어텐션 자체는 시퀀스의 순서를 보존하지 않는다는 점에 유의하십시오.
모델이 입력 시퀀스가 어떤 순서로 도착했는지 아는 것이 정말로 중요하다면 저희는 어떻게 해야 할까요?

토큰의 순서에 대한 정보를 보존하기 위한 지배적인 접근법은 이를 각 토큰과 연관된 추가 입력으로 모델에 표현하는 것입니다.
이러한 입력은 *위치 인코딩(positional encoding)* 이라고 불리며, 학습되거나 *사전에(a priori)* 고정될 수 있습니다.
이제 저희는 사인과 코사인 함수에 기반한 고정 위치 인코딩의 간단한 방식을 설명합니다 :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

입력 표현 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 가 시퀀스의 $n$ 토큰에 대한 $d$ 차원 임베딩을 포함한다고 가정합시다.
위치 인코딩은 같은 모양의 위치 임베딩 행렬 $\mathbf{P} \in \mathbb{R}^{n \times d}$ 를 사용해 $\mathbf{X} + \mathbf{P}$ 를 출력하는데, 그 $i^\textrm{th}$ 행과 $(2j)^\textrm{th}$ 또는 $(2j + 1)^\textrm{th}$ 열의 원소는 다음과 같습니다.

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

언뜻 보기에 이 삼각 함수 설계는 이상해 보입니다.
이 설계에 대한 설명을 제공하기 전에, 먼저 다음 `PositionalEncoding` 클래스에서 구현해 봅시다.

```{.python .input}
%%tab mxnet
class PositionalEncoding(nn.Block):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
%%tab pytorch
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
%%tab tensorflow
class PositionalEncoding(tf.keras.layers.Layer):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

```{.python .input}
%%tab jax
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        # Create a long enough P
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(
            -1, 1) / jnp.power(10000, jnp.arange(
            0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X))

    @nn.compact
    def __call__(self, X, training=False):
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'P', self.P)
        X = X + self.P[:, :X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)
```

위치 임베딩 행렬 $\mathbf{P}$ 에서, [**행은 시퀀스 내의 위치에 해당하고 열은 서로 다른 위치 인코딩 차원을 나타냅니다**].
아래 예시에서, 위치 임베딩 행렬의 $6^{\textrm{th}}$ 와 $7^{\textrm{th}}$ 열이 $8^{\textrm{th}}$ 와 $9^{\textrm{th}}$ 열보다 더 높은 주파수를 가지는 것을 볼 수 있습니다.
$6^{\textrm{th}}$ 와 $7^{\textrm{th}}$ 열 사이의 오프셋($8^{\textrm{th}}$ 와 $9^{\textrm{th}}$ 도 같음)은 사인과 코사인 함수의 교대 때문입니다.

```{.python .input}
%%tab mxnet
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

```{.python .input}
%%tab jax
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
params = pos_encoding.init(d2l.get_key(), d2l.zeros((1, num_steps, encoding_dim)))
X, inter_vars = pos_encoding.apply(params, d2l.zeros((1, num_steps, encoding_dim)),
                                   mutable='intermediates')
P = inter_vars['intermediates']['P'][0]  # retrieve intermediate value P
P = P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### 절대 위치 정보

인코딩 차원에 따라 단조 감소하는 주파수가 절대 위치 정보와 어떻게 관련되는지 보기 위해, $0, 1, \ldots, 7$ 의 [**이진 표현**]을 출력해 봅시다.
보시다시피, 가장 낮은 비트, 두 번째로 낮은 비트, 세 번째로 낮은 비트는 각각 모든 숫자, 두 개 숫자마다, 네 개 숫자마다 교대합니다.

```{.python .input}
%%tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

이진 표현에서, 더 높은 비트는 더 낮은 비트보다 더 낮은 주파수를 가집니다.
유사하게, 아래의 히트맵에서 입증된 것처럼, [**위치 인코딩은 삼각 함수를 사용해 인코딩 차원에 따라 주파수를 감소**]시킵니다.
출력이 부동소수점 수이기 때문에, 그러한 연속적인 표현은 이진 표현보다 공간 효율적입니다.

```{.python .input}
%%tab mxnet
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab jax
P = jnp.expand_dims(jnp.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 상대 위치 정보

절대 위치 정보를 포착하는 것 외에도, 위의 위치 인코딩은 모델이 상대 위치에 의해 주의를 기울이는 것을 쉽게 학습할 수 있게 합니다.
이는 어떤 고정된 위치 오프셋 $\delta$ 에 대해, 위치 $i + \delta$ 에서의 위치 인코딩이 위치 $i$ 에서의 인코딩의 선형 사영으로 표현될 수 있기 때문입니다.


이 사영은 수학적으로 설명될 수 있습니다.
$\omega_j = 1/10000^{2j/d}$ 로 표기할 때, :eqref:`eq_positional-encoding-def` 의 어떤 쌍 $(p_{i, 2j}, p_{i, 2j+1})$ 도 어떤 고정된 오프셋 $\delta$ 에 대해 $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$ 으로 선형 사영될 수 있습니다.

$$\begin{aligned}
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

여기서 $2\times 2$ 사영 행렬은 어떤 위치 인덱스 $i$ 에도 의존하지 않습니다.

## 요약

셀프 어텐션에서 쿼리, 키, 값은 모두 같은 곳에서 나옵니다.
CNN과 셀프 어텐션 모두 병렬 계산의 이점을 누리고 셀프 어텐션은 가장 짧은 최대 경로 길이를 가집니다.
그러나 시퀀스 길이에 대한 이차 계산 복잡도는 매우 긴 시퀀스에 대해 셀프 어텐션을 엄청나게 느리게 만듭니다.
시퀀스 순서 정보를 사용하기 위해, 저희는 입력 표현에 위치 인코딩을 추가함으로써 절대 또는 상대 위치 정보를 주입할 수 있습니다.

## 연습문제

1. 위치 인코딩을 가진 셀프 어텐션 계층을 쌓아 시퀀스를 표현하는 깊은 아키텍처를 설계한다고 가정합시다. 가능한 문제는 무엇일까요?
1. 학습 가능한 위치 인코딩 방법을 설계할 수 있습니까?
1. 셀프 어텐션에서 비교되는 쿼리와 키 간의 서로 다른 오프셋에 따라 서로 다른 학습된 임베딩을 할당할 수 있을까요? 힌트: 상대 위치 임베딩 :cite:`shaw2018self,huang2018music` 을 참조할 수 있습니다.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3870)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18030)
:end_tab:
