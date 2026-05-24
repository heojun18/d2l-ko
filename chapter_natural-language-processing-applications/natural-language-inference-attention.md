# 자연어 추론: 어텐션 사용하기
:label:`sec_natural-language-inference-attention`

저희는 :numref:`sec_natural-language-inference-and-dataset`에서 자연어 추론 작업과 SNLI 데이터셋을 소개했습니다. 복잡하고 심층적인 아키텍처에 기반한 많은 모델들을 고려하여, :citet:`Parikh.Tackstrom.Das.ea.2016`은 어텐션 메커니즘으로 자연어 추론을 해결할 것을 제안하였고 이를 "분해 가능한 어텐션 모델(decomposable attention model)"이라고 불렀습니다.
이는 순환 또는 합성곱 레이어가 없는 모델을 만들어, 훨씬 적은 파라미터로 당시 SNLI 데이터셋에서 최고의 결과를 달성하였습니다.
이 절에서는 :numref:`fig_nlp-map-nli-attention`에 나타난 것처럼 자연어 추론을 위한 이 어텐션 기반 방법(MLP와 함께)을 설명하고 구현할 것입니다.

![이 절에서는 자연어 추론을 위해 사전 학습된 GloVe를 어텐션과 MLP에 기반한 아키텍처에 입력합니다.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`


## 모델

전제와 가설에서 토큰의 순서를 보존하는 것보다 더 간단하게,
저희는 한 텍스트 시퀀스의 토큰을 다른 시퀀스의 모든 토큰에 정렬하고 그 반대도 마찬가지로 할 수 있습니다.
그런 다음 이러한 정보를 비교하고 집계하여 전제와 가설 간의
논리적 관계를 예측합니다.
기계 번역에서 원본 문장과 대상 문장 간의 토큰 정렬과 유사하게,
전제와 가설 간의 토큰 정렬은
어텐션 메커니즘으로 깔끔하게 수행될 수 있습니다.

![어텐션 메커니즘을 사용한 자연어 추론.](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention`은 어텐션 메커니즘을 사용한 자연어 추론 방법을 나타냅니다.
높은 수준에서, 이는 공동으로 학습되는 세 가지 단계로 구성됩니다: 어텐딩(attending), 비교(comparing), 그리고 집계(aggregating).
다음에서 단계별로 이들을 설명할 것입니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### 어텐딩(Attending)

첫 번째 단계는 한 텍스트 시퀀스의 토큰을 다른 시퀀스의 각 토큰에 정렬하는 것입니다.
전제가 "i do need sleep"이고 가설이 "i am tired"라고 가정합시다.
의미적 유사성으로 인해,
저희는 가설의 "i"를 전제의 "i"와 정렬하고,
가설의 "tired"를 전제의 "sleep"과 정렬하기를 원할 수 있습니다.
마찬가지로, 전제의 "i"를 가설의 "i"와 정렬하고,
전제의 "need"와 "sleep"을 가설의 "tired"와 정렬하기를 원할 수 있습니다.
이러한 정렬은 가중 평균을 사용한 *부드러운(soft)* 정렬임에 유의하세요.
이상적으로 큰 가중치가 정렬될 토큰들과 연관됩니다.
설명의 편의를 위해, :numref:`fig_nli_attention`은 이러한 정렬을 *경직된(hard)* 방식으로 보여줍니다.

이제 어텐션 메커니즘을 사용한 부드러운 정렬에 대해 좀 더 자세히 설명하겠습니다.
전제와 가설을 각각
$\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$과
$\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$로 표기하며,
토큰의 개수는 각각 $m$과 $n$입니다.
여기서 $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$)는 $d$차원 단어 벡터입니다.
부드러운 정렬을 위해, 다음과 같이 어텐션 가중치 $e_{ij} \in \mathbb{R}$을 계산합니다.

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

여기서 함수 $f$는 다음 `mlp` 함수에서 정의된 MLP입니다.
$f$의 출력 차원은 `mlp`의 `num_hiddens` 인자로 지정됩니다.

```{.python .input}
#@tab mxnet
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

강조해야 할 점은, :eqref:`eq_nli_e`에서
$f$는 $\mathbf{a}_i$와 $\mathbf{b}_j$ 쌍을 함께 입력으로 받는 대신 별도로 입력을 받는다는 것입니다.
이 *분해(decomposition)* 트릭은 $mn$번의 적용(이차 복잡도)이 아닌 $m + n$번의 적용(선형 복잡도)만으로 $f$를 적용하게 합니다.


:eqref:`eq_nli_e`의 어텐션 가중치를 정규화하여,
저희는 전제에서 인덱스 $i$인 토큰과 부드럽게 정렬된 가설의 표현을 얻기 위해
가설의 모든 토큰 벡터의 가중 평균을 계산합니다.

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

마찬가지로, 가설에서 인덱스 $j$인 각 토큰에 대해 전제 토큰의 부드러운 정렬을 계산합니다.

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

아래에서 입력 전제 `A`와의 가설(`beta`)의 부드러운 정렬 및 입력 가설 `B`와의 전제(`alpha`)의 부드러운 정렬을 계산하기 위해 `Attend` 클래스를 정의합니다.

```{.python .input}
#@tab mxnet
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### 비교(Comparing)

다음 단계에서, 저희는 한 시퀀스의 토큰을 그 토큰과 부드럽게 정렬된 다른 시퀀스와 비교합니다.
부드러운 정렬에서는 한 시퀀스의 모든 토큰이 아마도 서로 다른 어텐션 가중치를 가지면서 다른 시퀀스의 토큰과 비교될 것임에 유의하세요.
설명의 편의를 위해, :numref:`fig_nli_attention`은 토큰들을 정렬된 토큰들과 *경직된(hard)* 방식으로 쌍을 이루어 보여줍니다.
예를 들어, 어텐딩 단계에서 전제의 "need"와 "sleep"이 모두 가설의 "tired"와 정렬된다고 결정되었다면, "tired--need sleep" 쌍이 비교될 것입니다.

비교 단계에서는, 한 시퀀스의 토큰과 다른 시퀀스의 정렬된 토큰을 연결(연산자 $[\cdot, \cdot]$)한 것을 함수 $g$(MLP)에 입력합니다.

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`


:eqref:`eq_nli_v_ab`에서, $\mathbf{v}_{A,i}$는 전제의 토큰 $i$와 토큰 $i$와 부드럽게 정렬된 모든 가설 토큰 간의 비교이며,
$\mathbf{v}_{B,j}$는 가설의 토큰 $j$와 토큰 $j$와 부드럽게 정렬된 모든 전제 토큰 간의 비교입니다.
다음 `Compare` 클래스는 그러한 비교 단계를 정의합니다.

```{.python .input}
#@tab mxnet
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### 집계(Aggregating)

두 세트의 비교 벡터 $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$)와 $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$)를 가지고,
마지막 단계에서 저희는 논리적 관계를 추론하기 위해 이러한 정보를 집계할 것입니다.
저희는 두 세트를 모두 합하는 것으로 시작합니다.

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

다음으로, 논리적 관계의 분류 결과를 얻기 위해 두 요약 결과를 연결한 것을 함수 $h$(MLP)에 입력합니다.

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

집계 단계는 다음 `Aggregate` 클래스에서 정의됩니다.

```{.python .input}
#@tab mxnet
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### 모두 통합하기

어텐딩, 비교, 집계 단계를 함께 결합하여,
저희는 이 세 단계를 공동으로 학습하기 위한 분해 가능한 어텐션 모델을 정의합니다.

```{.python .input}
#@tab mxnet
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## 모델 학습 및 평가

이제 SNLI 데이터셋에서 정의된 분해 가능한 어텐션 모델을 학습하고 평가할 것입니다.
저희는 데이터셋을 읽는 것부터 시작합니다.


### 데이터셋 읽기

저희는 :numref:`sec_natural-language-inference-and-dataset`에서 정의된 함수를 사용하여 SNLI 데이터셋을 다운로드하고 읽습니다. 배치 크기와 시퀀스 길이는 각각 $256$과 $50$으로 설정됩니다.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### 모델 생성하기

저희는 입력 토큰을 표현하기 위해 사전 학습된 100차원 GloVe 임베딩을 사용합니다.
따라서 :eqref:`eq_nli_e`에서 벡터 $\mathbf{a}_i$와 $\mathbf{b}_j$의 차원을 100으로 미리 정의합니다.
:eqref:`eq_nli_e`의 함수 $f$와 :eqref:`eq_nli_v_ab`의 함수 $g$의 출력 차원은 200으로 설정됩니다.
그런 다음 모델 인스턴스를 생성하고, 파라미터를 초기화한 다음,
입력 토큰의 벡터를 초기화하기 위해 GloVe 임베딩을 로드합니다.

```{.python .input}
#@tab mxnet
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### 모델 학습 및 평가

텍스트 시퀀스(또는 이미지)와 같은 단일 입력을 받는 :numref:`sec_multi_gpu`의 `split_batch` 함수와 대조적으로,
저희는 미니배치에서 전제 및 가설과 같은 여러 입력을 받기 위한 `split_batch_multi_inputs` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

이제 SNLI 데이터셋에서 모델을 학습하고 평가할 수 있습니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### 모델 사용하기

마지막으로, 전제와 가설 쌍 간의 논리적 관계를 출력하기 위해 예측 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

저희는 학습된 모델을 사용하여 샘플 문장 쌍에 대한 자연어 추론 결과를 얻을 수 있습니다.

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## 요약

* 분해 가능한 어텐션 모델은 전제와 가설 간의 논리적 관계를 예측하기 위한 세 단계, 즉 어텐딩, 비교, 집계로 구성됩니다.
* 어텐션 메커니즘을 사용하여, 한 텍스트 시퀀스의 토큰을 다른 시퀀스의 모든 토큰에 정렬할 수 있으며 그 반대도 마찬가지입니다. 이러한 정렬은 가중 평균을 사용한 부드러운 정렬로, 이상적으로는 큰 가중치가 정렬될 토큰들과 연관됩니다.
* 분해 트릭은 어텐션 가중치를 계산할 때 이차 복잡도가 아닌 더 바람직한 선형 복잡도로 이어집니다.
* 저희는 자연어 추론과 같은 다운스트림 자연어 처리 작업의 입력 표현으로 사전 학습된 단어 벡터를 사용할 수 있습니다.


## 연습문제

1. 다른 하이퍼파라미터 조합으로 모델을 학습해 보세요. 테스트 세트에서 더 나은 정확도를 얻을 수 있겠습니까?
1. 자연어 추론을 위한 분해 가능한 어텐션 모델의 주요 단점은 무엇입니까?
1. 어떤 문장 쌍에 대해서든 의미적 유사도 수준(예: 0과 1 사이의 연속적인 값)을 얻고 싶다고 가정합시다. 데이터셋을 어떻게 수집하고 레이블링해야 합니까? 어텐션 메커니즘을 가진 모델을 설계할 수 있겠습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1530)
:end_tab:
