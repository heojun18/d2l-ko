# 트랜스포머로부터의 양방향 인코더 표현 (BERT)
:label:`sec_bert`

저희는 자연어 이해를 위한 몇 가지 단어 임베딩 모델을 소개했습니다.
사전 학습 후, 출력은 행렬로 생각할 수 있으며,
여기서 각 행은 미리 정의된 어휘의 한 단어를 표현하는 벡터입니다.
사실, 이러한 단어 임베딩 모델은 모두 *문맥 독립적(context-independent)* 입니다.
이 특성을 설명하는 것부터 시작합시다.


## 문맥 독립에서 문맥 민감으로

:numref:`sec_word2vec_pretraining`와 :numref:`sec_synonyms`의 실험을 떠올려 보십시오.
예를 들어, word2vec과 GloVe는 모두 단어의 문맥(있을 경우)과 관계없이 같은 단어에 같은 사전 학습된 벡터를 할당합니다.
형식적으로, 임의의 토큰 $x$에 대한 문맥 독립 표현은
$x$만을 입력으로 받는 함수 $f(x)$입니다.
자연어에서 다의어와 복잡한 의미의 풍부함을 고려할 때,
문맥 독립 표현은 명백한 한계를 가집니다.
예를 들어, "crane"이라는 단어는
"a crane is flying"과 "a crane driver came"이라는 문맥에서 완전히 다른 의미를 가집니다.
따라서, 같은 단어가 문맥에 따라 다른 표현을 할당받을 수도 있습니다.

이는 *문맥 민감(context-sensitive)* 단어 표현의 개발을 동기 부여하며,
여기서 단어의 표현은 그 문맥에 의존합니다.
따라서, 토큰 $x$의 문맥 민감 표현은 $x$와 그 문맥 $c(x)$ 모두에 의존하는
함수 $f(x, c(x))$입니다.
인기 있는 문맥 민감 표현으로는
TagLM(language-model-augmented sequence tagger) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe(Context Vectors) :cite:`McCann.Bradbury.Xiong.ea.2017`,
ELMo(Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`가 있습니다.

예를 들어, 전체 시퀀스를 입력으로 받음으로써,
ELMo는 입력 시퀀스의 각 단어에 표현을 할당하는 함수입니다.
구체적으로, ELMo는 사전 학습된 양방향 LSTM의 모든 중간 레이어 표현을 출력 표현으로 결합합니다.
그런 다음 ELMo 표현은 기존 모델에서 토큰의 ELMo 표현과 원래 표현(예: GloVe)을 연결하는 식으로
다운스트림 작업의 기존 지도 모델에 추가 특징으로 추가될 것입니다.
한편으로는,
ELMo 표현이 추가된 후 사전 학습된 양방향 LSTM 모델의 모든 가중치는 동결됩니다.
다른 한편으로는,
기존 지도 모델은 주어진 작업에 특화되어 맞춤 제작됩니다.
당시 서로 다른 작업에 서로 다른 최고의 모델을 활용하면서,
ELMo를 추가하는 것은 여섯 개의 자연어 처리 작업에서 최고 성능을 개선했습니다.
감성 분석, 자연어 추론,
의미역 결정, 상호 참조 해결,
개체명 인식, 질의응답입니다.


## 작업 특화에서 작업 비특화로

ELMo는 다양한 자연어 처리 작업의 해결책을 크게 개선했지만,
각 해결책은 여전히 *작업 특화(task-specific)* 아키텍처에 의존합니다.
그러나, 모든 자연어 처리 작업에 대해 특정 아키텍처를 만드는 것은 실제로 쉽지 않습니다.
GPT(Generative Pre-Training) 모델은 문맥 민감 표현을 위한 일반적인 *작업 비특화(task-agnostic)* 모델을 설계하는 데
한 가지 노력을 나타냅니다 :cite:`Radford.Narasimhan.Salimans.ea.2018`.
트랜스포머 디코더에 기반하여,
GPT는 텍스트 시퀀스를 표현하는 데 사용될 언어 모델을 사전 학습합니다.
GPT를 다운스트림 작업에 적용할 때,
언어 모델의 출력은 작업의 레이블을 예측하기 위해 추가된 선형 출력 레이어에
입력될 것입니다.
사전 학습된 모델의 파라미터를 동결하는 ELMo와는 극명한 대조로,
GPT는 다운스트림 작업의 지도 학습 동안
사전 학습된 트랜스포머 디코더의 *모든* 파라미터를 미세 조정합니다.
GPT는 자연어 추론, 질의응답, 문장 유사도, 분류의 12개 작업에서 평가되었으며,
모델 아키텍처에 최소한의 변경으로 그중 9개에서
최고 성능을 개선했습니다.

그러나, 언어 모델의 자기회귀적 특성으로 인해,
GPT는 (왼쪽에서 오른쪽으로) 앞쪽만 봅니다.
"i went to the bank to deposit cash"와 "i went to the bank to sit down"이라는 문맥에서,
"bank"가 그 왼쪽의 문맥에 민감하므로,
GPT는 "bank"에 대해 같은 표현을 반환할 것입니다.
비록 그것이 다른 의미를 가지더라도 말입니다.


## BERT: 두 세계의 장점을 결합하기

저희가 본 것처럼,
ELMo는 문맥을 양방향으로 인코딩하지만 작업 특화 아키텍처를 사용합니다.
반면 GPT는 작업 비특화이지만 문맥을 왼쪽에서 오른쪽으로 인코딩합니다.
두 세계의 장점을 결합하여,
BERT(Bidirectional Encoder Representations from Transformers)는
문맥을 양방향으로 인코딩하고 광범위한 자연어 처리 작업에 대해
최소한의 아키텍처 변경을 요구합니다 :cite:`Devlin.Chang.Lee.ea.2018`.
사전 학습된 트랜스포머 인코더를 사용하여,
BERT는 그것의 양방향 문맥에 기반하여 임의의 토큰을 표현할 수 있습니다.
다운스트림 작업의 지도 학습 동안,
BERT는 두 가지 측면에서 GPT와 유사합니다.
첫째, BERT 표현은
모든 토큰에 대해 예측하는 것 대 전체 시퀀스에 대해 예측하는 것 같이
작업의 성격에 따라 모델 아키텍처에 최소한의 변경을 가하면서
추가된 출력 레이어에 입력될 것입니다.
둘째,
사전 학습된 트랜스포머 인코더의 모든 파라미터가 미세 조정되며,
추가 출력 레이어는 처음부터 학습될 것입니다.
:numref:`fig_elmo-gpt-bert`는 ELMo, GPT, BERT 사이의 차이를 묘사합니다.

![ELMo, GPT, BERT의 비교.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`


BERT는 (i) 단일 텍스트 분류(예: 감성 분석), (ii) 텍스트 쌍 분류(예: 자연어 추론),
(iii) 질의응답, (iv) 텍스트 태깅(예: 개체명 인식)의 광범위한 범주 아래
열한 개의 자연어 처리 작업에서 최고 성능을 더욱 개선했습니다.
모두 2018년에 제안되었으며,
문맥 민감 ELMo에서 작업 비특화 GPT 및 BERT까지,
개념적으로 단순하지만 경험적으로 강력한 자연어에 대한 깊은 표현의 사전 학습은
다양한 자연어 처리 작업의 해결책에 혁명을 일으켰습니다.

이 장의 나머지 부분에서는,
저희는 BERT의 사전 학습에 깊이 들어갈 것입니다.
:numref:`chap_nlp_app`에서 자연어 처리 응용이 설명될 때,
저희는 다운스트림 응용을 위한 BERT의 미세 조정을 설명할 것입니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**입력 표현**]
:label:`subsec_bert_input_rep`

자연어 처리에서,
어떤 작업(예: 감성 분석)은 단일 텍스트를 입력으로 받는 반면,
다른 어떤 작업(예: 자연어 추론)에서는,
입력이 한 쌍의 텍스트 시퀀스입니다.
BERT 입력 시퀀스는 단일 텍스트와 텍스트 쌍 모두를 명확히 표현합니다.
전자의 경우,
BERT 입력 시퀀스는
특수 분류 토큰 "&lt;cls&gt;",
텍스트 시퀀스의 토큰들,
특수 분리 토큰 "&lt;sep&gt;"의
연결입니다.
후자의 경우,
BERT 입력 시퀀스는
"&lt;cls&gt;", 첫 번째 텍스트 시퀀스의 토큰들,
"&lt;sep&gt;", 두 번째 텍스트 시퀀스의 토큰들, "&lt;sep&gt;"의
연결입니다.
저희는 "BERT 입력 시퀀스"라는 용어를
다른 종류의 "시퀀스"와 일관되게 구별할 것입니다.
예를 들어, 하나의 *BERT 입력 시퀀스* 는 하나의 *텍스트 시퀀스* 또는 두 개의 *텍스트 시퀀스* 중 어느 것이든 포함할 수 있습니다.

텍스트 쌍을 구별하기 위해,
학습된 세그먼트 임베딩 $\mathbf{e}_A$와 $\mathbf{e}_B$가
각각 첫 번째 시퀀스와 두 번째 시퀀스의 토큰 임베딩에 추가됩니다.
단일 텍스트 입력의 경우, $\mathbf{e}_A$만 사용됩니다.

다음의 `get_tokens_and_segments`는 한 문장 또는 두 문장을 입력으로 받아,
BERT 입력 시퀀스의 토큰과
그에 해당하는 세그먼트 ID를 반환합니다.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT는 그것의 양방향 아키텍처로 트랜스포머 인코더를 선택합니다.
트랜스포머 인코더에서 일반적인 것처럼,
위치 임베딩이 BERT 입력 시퀀스의 모든 위치에 추가됩니다.
그러나, 원래의 트랜스포머 인코더와는 다르게,
BERT는 *학습 가능한(learnable)* 위치 임베딩을 사용합니다.
요약하면, :numref:`fig_bert-input`은
BERT 입력 시퀀스의 임베딩이
토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합임을 보여줍니다.

![BERT 입력 시퀀스의 임베딩은
토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합입니다.](../img/bert-input.svg)
:label:`fig_bert-input`

다음 [**`BERTEncoder` 클래스**]는
:numref:`sec_transformer`에서 구현된 `TransformerEncoder` 클래스와 유사합니다.
`TransformerEncoder`와 다르게, `BERTEncoder`는
세그먼트 임베딩과 학습 가능한 위치 임베딩을 사용합니다.

```{.python .input}
#@tab mxnet
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

어휘 크기가 10000이라고 가정합시다.
[**`BERTEncoder`의 순방향 추론**]을 시연하기 위해,
그것의 인스턴스를 만들고 파라미터를 초기화합시다.

```{.python .input}
#@tab mxnet
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_blks, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
ffn_num_input, num_blks, dropout = 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_blks, dropout)
```

저희는 `tokens`을 길이 8의 BERT 입력 시퀀스 2개로 정의합니다.
여기서 각 토큰은 어휘의 인덱스입니다.
입력 `tokens`으로 `BERTEncoder`의 순방향 추론은
인코딩된 결과를 반환하며, 여기서 각 토큰은
하이퍼파라미터 `num_hiddens`에 의해 미리 정의된 길이의 벡터로 표현됩니다.
이 하이퍼파라미터는 일반적으로 트랜스포머 인코더의 *은닉 크기(hidden size)*
(은닉 유닛의 수)로 지칭됩니다.

```{.python .input}
#@tab mxnet
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## 사전 학습 작업
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder`의 순방향 추론은 입력 텍스트의 각 토큰과
삽입된 특수 토큰 "&lt;cls&gt;" 및 "&lt;seq&gt;"의 BERT 표현을 제공합니다.
다음으로, 저희는 BERT를 사전 학습하기 위한 손실 함수를 계산하기 위해
이러한 표현을 사용할 것입니다.
사전 학습은 다음 두 가지 작업으로 구성됩니다.
마스킹된 언어 모델링과 다음 문장 예측입니다.

### [**마스킹된 언어 모델링**]
:label:`subsec_mlm`

:numref:`sec_language-model`에 설명된 것처럼,
언어 모델은 왼쪽의 문맥을 사용해 토큰을 예측합니다.
각 토큰을 표현하기 위해 문맥을 양방향으로 인코딩하기 위해,
BERT는 토큰을 무작위로 마스킹하고 양방향 문맥의 토큰을 사용해
마스킹된 토큰을 자기 지도 방식으로 예측합니다.
이 작업은 *마스킹된 언어 모델(masked language model)* 로 지칭됩니다.

이 사전 학습 작업에서,
15%의 토큰이 예측을 위한 마스킹된 토큰으로 무작위로 선택될 것입니다.
레이블을 사용해 부정 행위 없이 마스킹된 토큰을 예측하기 위해,
한 가지 직관적인 접근법은 BERT 입력 시퀀스에서 그것을 항상 특수 "&lt;mask&gt;" 토큰으로 대체하는 것입니다.
그러나, 인공적인 특수 토큰 "&lt;mask&gt;"는 미세 조정에서는
결코 등장하지 않습니다.
사전 학습과 미세 조정 사이의 이러한 불일치를 피하기 위해,
토큰이 예측을 위해 마스킹되면(예: "this movie is great"에서 "great"가 마스킹되고 예측되도록 선택됨),
입력에서 그것은 다음과 같이 대체될 것입니다.

* 80%의 경우 특수 "&lt;mask&gt;" 토큰(예: "this movie is great"는 "this movie is &lt;mask&gt;"가 됨);
* 10%의 경우 무작위 토큰(예: "this movie is great"는 "this movie is drink"가 됨);
* 10%의 경우 변경되지 않은 레이블 토큰(예: "this movie is great"는 "this movie is great"가 됨).

15%의 시간 중 10%의 시간 동안 무작위 토큰이 삽입됨에 유의하십시오.
이 가끔의 노이즈는 BERT가 그것의 양방향 문맥 인코딩에서 마스킹된 토큰(특히 레이블 토큰이 변경되지 않은 경우)으로 덜 편향되도록 장려합니다.

저희는 BERT 사전 학습의 마스킹된 언어 모델 작업에서 마스킹된 토큰을 예측하기 위해
다음의 `MaskLM` 클래스를 구현합니다.
예측은 하나의 은닉 레이어 MLP(`self.mlp`)를 사용합니다.
순방향 추론에서, 이는 두 개의 입력을 받습니다.
`BERTEncoder`의 인코딩된 결과와 예측을 위한 토큰 위치입니다.
출력은 이러한 위치에서의 예측 결과입니다.

```{.python .input}
#@tab mxnet
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

[**`MaskLM`의 순방향 추론**]을 시연하기 위해,
저희는 그것의 인스턴스 `mlm`을 만들고 이를 초기화합니다.
`BERTEncoder`의 순방향 추론으로부터의 `encoded_X`가
2개의 BERT 입력 시퀀스를 표현함을 떠올리십시오.
저희는 `mlm_positions`를 `encoded_X`의 어느 BERT 입력 시퀀스에서든 예측할 3개의 인덱스로 정의합니다.
`mlm`의 순방향 추론은 `encoded_X`의 모든 마스킹된 위치 `mlm_positions`에서
예측 결과 `mlm_Y_hat`을 반환합니다.
각 예측에 대해, 결과의 크기는 어휘 크기와 같습니다.

```{.python .input}
#@tab mxnet
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

마스크 아래 예측된 토큰 `mlm_Y_hat`의 정답 레이블 `mlm_Y`로,
저희는 BERT 사전 학습의 마스킹된 언어 모델 작업의 교차 엔트로피 손실을 계산할 수 있습니다.

```{.python .input}
#@tab mxnet
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### [**다음 문장 예측**]
:label:`subsec_nsp`

마스킹된 언어 모델링이 단어를 표현하기 위해 양방향 문맥을 인코딩할 수 있지만,
텍스트 쌍 사이의 논리적 관계를 명시적으로 모델링하지는 않습니다.
두 텍스트 시퀀스 사이의 관계를 이해하는 데 도움이 되도록,
BERT는 그것의 사전 학습에서 이진 분류 작업, *다음 문장 예측(next sentence prediction)* 을 고려합니다.
사전 학습을 위한 문장 쌍을 생성할 때,
절반의 시간 동안 그들은 실제로 레이블 "True"인 연속된 문장이고,
나머지 절반의 시간 동안 두 번째 문장은 말뭉치에서 레이블 "False"로 무작위로 샘플링됩니다.

다음의 `NextSentencePred` 클래스는 BERT 입력 시퀀스에서
두 번째 문장이 첫 번째 문장의 다음 문장인지를 예측하기 위해
하나의 은닉 레이어 MLP를 사용합니다.
트랜스포머 인코더의 셀프 어텐션 덕분에,
특수 토큰 "&lt;cls&gt;"의 BERT 표현은
입력으로부터 두 문장을 모두 인코딩합니다.
따라서, MLP 분류기의 출력 레이어(`self.output`)는 `X`를 입력으로 받으며,
여기서 `X`는 인코딩된 "&lt;cls&gt;" 토큰을 입력으로 하는 MLP 은닉 레이어의 출력입니다.

```{.python .input}
#@tab mxnet
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

저희는 [**`NextSentencePred` 인스턴스의 순방향 추론**]이
각 BERT 입력 시퀀스에 대해 이진 예측을 반환함을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default will not flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

2개의 이진 분류의 교차 엔트로피 손실도 계산할 수 있습니다.

```{.python .input}
#@tab mxnet
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

앞서 언급한 두 사전 학습 작업의 모든 레이블이
수작업 레이블링 노력 없이 사전 학습 말뭉치에서 손쉽게 얻을 수 있다는 점이 주목할 만합니다.
원래의 BERT는 BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015`와
영어 위키피디아의 연결로 사전 학습되었습니다.
이러한 두 텍스트 말뭉치는 거대합니다.
그들은 각각 8억 개와 25억 개의 단어를 가집니다.


## [**모두 합치기**]

BERT를 사전 학습할 때, 최종 손실 함수는
마스킹된 언어 모델링과 다음 문장 예측에 대한
두 손실 함수의 선형 결합입니다.
이제 저희는 세 클래스 `BERTEncoder`, `MaskLM`, `NextSentencePred`를 인스턴스화하여
`BERTModel` 클래스를 정의할 수 있습니다.
순방향 추론은 인코딩된 BERT 표현 `encoded_X`,
마스킹된 언어 모델링의 예측 `mlm_Y_hat`,
다음 문장 예측 `nsp_Y_hat`을 반환합니다.

```{.python .input}
#@tab mxnet
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, 
                 num_heads, num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len=max_len)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## 요약

* word2vec과 GloVe 같은 단어 임베딩 모델은 문맥 독립적입니다. 그들은 단어의 문맥(있을 경우)과 관계없이 같은 단어에 같은 사전 학습된 벡터를 할당합니다. 자연어에서 다의어나 복잡한 의미를 잘 처리하기 어렵습니다.
* ELMo와 GPT 같은 문맥 민감 단어 표현의 경우, 단어의 표현은 그 문맥에 의존합니다.
* ELMo는 문맥을 양방향으로 인코딩하지만 작업 특화 아키텍처를 사용합니다(그러나 모든 자연어 처리 작업에 대해 특정 아키텍처를 만드는 것은 실제로 쉽지 않습니다). 반면 GPT는 작업 비특화이지만 문맥을 왼쪽에서 오른쪽으로 인코딩합니다.
* BERT는 두 세계의 장점을 결합합니다. 문맥을 양방향으로 인코딩하고 광범위한 자연어 처리 작업에 대해 최소한의 아키텍처 변경을 요구합니다.
* BERT 입력 시퀀스의 임베딩은 토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합입니다.
* BERT의 사전 학습은 마스킹된 언어 모델링과 다음 문장 예측의 두 가지 작업으로 구성됩니다. 전자는 단어를 표현하기 위해 양방향 문맥을 인코딩할 수 있는 반면, 후자는 텍스트 쌍 사이의 논리적 관계를 명시적으로 모델링합니다.


## 연습문제

1. 다른 모든 조건이 같다면, 마스킹된 언어 모델이 왼쪽-오른쪽 언어 모델보다 수렴하기 위해 더 많은 또는 더 적은 사전 학습 단계를 필요로 할까요? 왜 그럴까요?
1. BERT의 원래 구현에서, `BERTEncoder`의 위치별 피드포워드 네트워크(`d2l.TransformerEncoderBlock`을 통한)와 `MaskLM`의 완전 연결 레이어 모두 가우시안 오차 선형 유닛(GELU) :cite:`Hendrycks.Gimpel.2016`을 활성화 함수로 사용합니다. GELU와 ReLU의 차이를 조사하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:
