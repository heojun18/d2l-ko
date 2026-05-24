```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 어텐션 스코어링 함수
:label:`sec_attention-scoring-functions`


:numref:`sec_attention-pooling` 에서 저희는 쿼리와 키 간의 상호작용을 모델링하기 위해 가우시안 커널을 포함한 여러 거리 기반 커널을 사용했습니다. 알고 보면 거리 함수는 내적보다 계산이 약간 더 비쌉니다. 따라서 음이 아닌 어텐션 가중치를 보장하는 소프트맥스 연산과 함께, 많은 작업이 :eqref:`eq_softmax_attention` 및 :numref:`fig_attention_output` 에서 더 계산하기 간단한 *어텐션 스코어링 함수* $a$ 에 집중되어 왔습니다.

![어텐션 스코어링 함수 $\mathit{a}$ 와 소프트맥스 연산으로 가중치가 계산되는, 값의 가중 평균으로서의 어텐션 풀링 출력 계산.](../img/attention-output.svg)
:label:`fig_attention_output`

```{.python .input}
%%tab mxnet
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
```

## [**내적 어텐션**]


잠시 가우시안 커널에서 (지수화 없이) 어텐션 함수를 다시 살펴봅시다.

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.
$$

첫째, 마지막 항이 $\mathbf{q}$ 에만 의존한다는 점을 주목하십시오. 따라서 그것은 모든 $(\mathbf{q}, \mathbf{k}_i)$ 쌍에 대해 동일합니다. :eqref:`eq_softmax_attention` 에서 행해진 것처럼 어텐션 가중치를 $1$ 로 정규화하면 이 항이 완전히 사라짐을 보장합니다. 둘째, (나중에 논의될) 배치 정규화와 층 정규화 모두 잘 경계지어진, 그리고 종종 일정한 노름 $\|\mathbf{k}_i\|$ 를 가진 활성화 값을 산출한다는 점을 주목하십시오. 예를 들어 키 $\mathbf{k}_i$ 가 층 정규화에 의해 생성되었을 때마다 그러한 경우입니다. 따라서 저희는 결과에 큰 변화 없이 $a$ 의 정의에서 그것을 떨어뜨릴 수 있습니다.

마지막으로, 지수 함수의 인수의 크기 정도를 통제 하에 둘 필요가 있습니다. 쿼리 $\mathbf{q} \in \mathbb{R}^d$ 와 키 $\mathbf{k}_i \in \mathbb{R}^d$ 의 모든 원소가 독립적이고 동일하게 추출된 평균 0, 분산 1의 확률 변수라고 가정합시다. 두 벡터 간의 내적은 평균이 0이고 분산이 $d$ 입니다. 벡터 길이와 무관하게 내적의 분산이 여전히 $1$ 로 유지되도록 보장하기 위해, 저희는 *스케일드 내적 어텐션(scaled dot product attention)* 스코어링 함수를 사용합니다. 즉, 저희는 내적을 $1/\sqrt{d}$ 로 재스케일합니다. 따라서 저희는 트랜스포머 :cite:`Vaswani.Shazeer.Parmar.ea.2017` 등에서 사용되는 첫 번째 흔히 사용되는 어텐션 함수에 도달합니다.

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

어텐션 가중치 $\alpha$ 는 여전히 정규화가 필요하다는 점에 유의하십시오. 저희는 소프트맥스 연산을 사용하여 :eqref:`eq_softmax_attention` 을 통해 이를 더 단순화할 수 있습니다.

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$
:eqlabel:`eq_attn-scoring-alpha`

알고 보면 인기 있는 모든 어텐션 메커니즘은 소프트맥스를 사용하므로, 저희는 이 장의 나머지 부분에서 그것에 한정하겠습니다.

## 편의 함수

저희는 어텐션 메커니즘을 효율적으로 배치하기 위해 몇 가지 함수가 필요합니다. 여기에는 가변 길이 문자열을 다루기 위한 도구(자연어 처리에서 일반적)와 미니배치에서 효율적으로 평가하기 위한 도구(배치 행렬 곱셈)가 포함됩니다.


### [**마스킹된 소프트맥스 연산**]

어텐션 메커니즘의 가장 인기 있는 응용 중 하나는 시퀀스 모델입니다. 따라서 저희는 서로 다른 길이의 시퀀스를 다룰 수 있어야 합니다. 어떤 경우에는 그러한 시퀀스들이 같은 미니배치에 들어갈 수 있는데, 더 짧은 시퀀스에 대해서는 더미 토큰으로 패딩이 필요합니다(예시는 :numref:`sec_machine_translation` 을 참조하십시오). 이러한 특수 토큰은 의미를 가지지 않습니다. 예를 들어, 저희가 다음 세 문장을 가진다고 가정합시다.

```
Dive  into  Deep    Learning 
Learn to    code    <blank>
Hello world <blank> <blank>
```


저희의 어텐션 모델에 빈칸을 원하지 않으므로, 저희는 실제 문장의 길이가 얼마나 길든, $l \leq n$ 에 대해 단순히 $\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ 를 $\sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$ 로 제한하면 됩니다. 이것은 매우 흔한 문제이므로 이름이 있습니다: *마스킹된 소프트맥스 연산(masked softmax operation)*.

구현해 봅시다. 사실 구현은 $i > l$ 에 대해 $\mathbf{v}_i$ 의 값을 0으로 설정함으로써 약간 속임수를 씁니다. 또한 어텐션 가중치를 $-10^{6}$ 같은 큰 음수로 설정하여 그래디언트와 값에 대한 기여를 실제로 사라지게 만듭니다. 이는 선형 대수 커널과 연산자가 GPU에 대해 크게 최적화되어 있어서, 조건문(if then else 문)이 있는 코드를 가지는 것보다 계산에서 약간 낭비하는 것이 더 빠르기 때문에 그렇게 합니다.

```{.python .input}
%%tab mxnet
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
%%tab pytorch
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
%%tab tensorflow
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)
    
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0    
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```{.python .input}
%%tab jax
def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)
```

[**이 함수가 어떻게 작동하는지 보여주기**] 위해, 유효 길이가 각각 $2$ 와 $3$ 인 크기 $2 \times 4$ 의 두 예시의 미니배치를 고려합니다. 마스킹된 소프트맥스 연산의 결과로, 각 벡터 쌍에 대해 유효 길이를 넘는 값은 모두 0으로 마스킹됩니다.

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)), jnp.array([2, 3]))
```

저희가 모든 예시의 두 벡터 각각에 대해 유효 길이를 지정하기 위해 더 세밀한 제어가 필요하다면, 단순히 유효 길이의 2차원 텐서를 사용합니다. 이는 다음을 산출합니다.

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform((2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)),
               jnp.array([[1, 3], [2, 4]]))
```

### 배치 행렬 곱셈
:label:`subsec_batch_dot`

또 다른 흔히 사용되는 연산은 배치 행렬을 서로 곱하는 것입니다. 이는 쿼리, 키, 값의 미니배치를 가질 때 유용합니다. 보다 구체적으로, 다음을 가정합시다.

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \\
    \mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}.
$$

그러면 배치 행렬 곱셈(BMM)은 원소별 곱셈을 계산합니다.

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}.$$
:eqlabel:`eq_batch-matrix-mul`

딥러닝 프레임워크에서 이것을 실제로 봅시다.

```{.python .input}
%%tab mxnet
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(npx.batch_dot(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab pytorch
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab tensorflow
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(tf.matmul(Q, K).numpy(), (2, 3, 6))
```

```{.python .input}
%%tab jax
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(jax.lax.batch_matmul(Q, K), (2, 3, 6))
```

## [**스케일드 내적 어텐션**]

:eqref:`eq_dot_product_attention` 에서 소개된 내적 어텐션으로 돌아갑시다.
일반적으로 이는 쿼리와 키가 같은 벡터 길이, 가령 $d$ 를 가져야 함을 요구합니다. 다만 이는 $\mathbf{q}^\top \mathbf{k}$ 를 $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ 로 대체함으로써 쉽게 해결할 수 있는데, 여기서 $\mathbf{M}$ 은 두 공간 간 변환을 위해 적절히 선택된 행렬입니다. 지금은 차원이 일치한다고 가정합니다.

실제로 저희는 효율성을 위해 종종 미니배치를 생각합니다. 예를 들어 길이가 $d$ 인 쿼리와 키, 길이가 $v$ 인 값에 대해 $n$ 개의 쿼리와 $m$ 개의 키-값 쌍에 대한 어텐션을 계산하는 것입니다. 따라서 쿼리 $\mathbf Q\in\mathbb R^{n\times d}$, 키 $\mathbf K\in\mathbb R^{m\times d}$, 값 $\mathbf V\in\mathbb R^{m\times v}$ 의 스케일드 내적 어텐션은 다음과 같이 쓸 수 있습니다.

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

이를 미니배치에 적용할 때는, :eqref:`eq_batch-matrix-mul` 에서 소개된 배치 행렬 곱셈이 필요하다는 점에 유의하십시오. 다음 스케일드 내적 어텐션 구현에서, 저희는 모델 정규화를 위해 드롭아웃을 사용합니다.

```{.python .input}
%%tab mxnet
class DotProductAttention(nn.Block):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of keys
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class DotProductAttention(tf.keras.layers.Layer):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    dropout: float

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None,
                 training=False):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.swapaxes(1, 2)
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights
```

[**`DotProductAttention` 클래스가 어떻게 작동하는지 보여주기**] 위해, 저희는 가산 어텐션에 대한 앞선 토이 예시의 동일한 키, 값, 유효 길이를 사용합니다. 저희 예시의 목적상 미니배치 크기가 $2$ 이고, 총 $10$ 개의 키와 값이 있으며, 값의 차원이 $4$ 라고 가정합니다. 마지막으로, 관측치당 유효 길이가 각각 $2$ 와 $6$ 이라고 가정합니다. 이를 고려할 때, 저희는 출력이 $2 \times 1 \times 4$ 텐서, 즉 미니배치의 예시당 한 행이 될 것으로 예상합니다.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
keys = tf.random.normal(shape=(2, 10, 2))
values = tf.random.normal(shape=(2, 10, 4))
valid_lens = tf.constant([2, 6])

attention = DotProductAttention(dropout=0.5)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 2))
keys = jax.random.normal(d2l.get_key(), (2, 10, 2))
values = jax.random.normal(d2l.get_key(), (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

(유효 길이를 $2$ 와 $6$ 으로 설정했기 때문에) 두 번째와 여섯 번째 열을 넘는 어떤 것에 대해서도 어텐션 가중치가 실제로 사라지는지 확인해 봅시다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## [**가산 어텐션**]
:label:`subsec_additive-attention`

쿼리 $\mathbf{q}$ 와 키 $\mathbf{k}$ 가 서로 다른 차원의 벡터일 때, 저희는 $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ 를 통해 불일치를 다루기 위해 행렬을 사용하거나, 스코어링 함수로 가산 어텐션을 사용할 수 있습니다. 또 다른 이점은 그 이름이 시사하듯이 어텐션이 가산적이라는 것입니다. 이는 약간의 계산상 절약으로 이어질 수 있습니다.
쿼리 $\mathbf{q} \in \mathbb{R}^q$ 와 키 $\mathbf{k} \in \mathbb{R}^k$ 가 주어졌을 때, *가산 어텐션(additive attention)* 스코어링 함수 :cite:`Bahdanau.Cho.Bengio.2014` 는 다음과 같이 주어집니다.

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

여기서 $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$, $\mathbf w_v\in\mathbb R^{h}$ 는 학습 가능한 매개변수입니다. 이 항은 다음으로 음이 아님과 정규화 모두를 보장하기 위해 소프트맥스에 공급됩니다.
:eqref:`eq_additive-attn` 의 동등한 해석은 쿼리와 키가 연결되어(concatenated) 단일 은닉층을 가진 MLP에 공급된다는 것입니다.
활성화 함수로 $\tanh$ 를 사용하고 편향 항을 비활성화하면서, 저희는 가산 어텐션을 다음과 같이 구현합니다.

```{.python .input}
%%tab mxnet
class AdditiveAttention(nn.Block):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Use flatten=False to only transform the last axis so that the
        # shapes for the other axes are kept the same
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1,
        # no. of key-value pairs, num_hiddens). Sum them up with
        # broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores:
        # (batch_size, no. of queries, no. of key-value pairs)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class AdditiveAttention(tf.keras.layers.Layer):  #@save
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class AdditiveAttention(nn.Module):  #@save
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = jnp.expand_dims(queries, axis=2) + jnp.expand_dims(keys, axis=1)
        features = nn.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return dropout_layer(attention_weights)@values, attention_weights
```

[**`AdditiveAttention` 이 어떻게 작동하는지 살펴봅시다**]. 저희의 토이 예시에서 저희는 각각 크기 $(2, 1, 20)$, $(2, 10, 2)$, $(2, 10, 4)$ 의 쿼리, 키, 값을 선택합니다. 이는 이제 쿼리가 $20$ 차원이라는 점을 제외하면 `DotProductAttention` 에 대한 저희의 선택과 동일합니다. 마찬가지로 미니배치의 시퀀스에 대한 유효 길이로 $(2, 6)$ 을 선택합니다.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 20))

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 20))
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

어텐션 함수를 검토할 때 저희는 `DotProductAttention` 과 정성적으로 매우 유사한 행동을 봅니다. 즉, 선택된 유효 길이 $(2, 6)$ 이내의 항들만이 0이 아닙니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## 요약

이 절에서 저희는 두 가지 핵심 어텐션 스코어링 함수, 즉 내적과 가산 어텐션을 소개했습니다. 그것들은 가변 길이의 시퀀스에 걸쳐 집계하기 위한 효과적인 도구입니다. 특히 내적 어텐션은 현대 트랜스포머 아키텍처의 주축입니다. 쿼리와 키가 서로 다른 길이의 벡터일 때, 저희는 가산 어텐션 스코어링 함수를 대신 사용할 수 있습니다. 이러한 계층의 최적화는 최근 몇 년간 발전의 주요 영역 중 하나입니다. 예를 들어 [NVIDIA의 트랜스포머 라이브러리](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) 와 Megatron :cite:`shoeybi2019megatron` 은 결정적으로 어텐션 메커니즘의 효율적인 변형에 의존합니다. 저희는 나중에 트랜스포머를 검토하면서 이를 좀 더 자세히 파고들 것입니다.

## 연습문제

1. `DotProductAttention` 코드를 수정하여 거리 기반 어텐션을 구현하십시오. 효율적인 구현을 위해서는 키의 제곱 노름 $\|\mathbf{k}_i\|^2$ 만 필요하다는 점에 유의하십시오.
1. 차원을 조정하기 위한 행렬을 사용하여 서로 다른 차원의 쿼리와 키를 허용하도록 내적 어텐션을 수정하십시오.
1. 계산 비용이 키, 쿼리, 값의 차원과 그 수에 따라 어떻게 스케일됩니까? 메모리 대역폭 요구사항은 어떻습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1064)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3867)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18027)
:end_tab:
