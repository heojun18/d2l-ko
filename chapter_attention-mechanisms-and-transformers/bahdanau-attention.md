```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Bahdanau 어텐션 메커니즘
:label:`sec_seq2seq_attention`

:numref:`sec_seq2seq` 에서 기계 번역을 마주했을 때, 저희는 두 개의 RNN :cite:`Sutskever.Vinyals.Le.2014` 에 기반한 시퀀스 투 시퀀스 학습을 위한 인코더-디코더 아키텍처를 설계했습니다.
구체적으로 RNN 인코더는 가변 길이 시퀀스를 *고정 형태(fixed-shape)* 컨텍스트 변수로 변환합니다.
그런 다음 RNN 디코더는 생성된 토큰과 컨텍스트 변수에 기반하여 토큰별로 출력(타깃) 시퀀스를 생성합니다.

추가적인 세부 사항과 함께 (:numref:`fig_s2s_attention_state` 으로) 다시 반복하는 :numref:`fig_seq2seq_details` 을 떠올려 보십시오. 전통적으로 RNN에서는 소스 시퀀스에 대한 모든 관련 정보가 인코더에 의해 어떤 내부 *고정 차원(fixed-dimensional)* 상태 표현으로 번역됩니다. 디코더가 번역된 시퀀스를 생성하기 위한 완전하고 배타적인 정보 원천으로 사용하는 것이 바로 이 상태입니다. 다시 말해, 시퀀스 투 시퀀스 메커니즘은 중간 상태를 입력으로 작용했을 어떤 문자열이든 그것의 충분 통계량으로 취급합니다.

![시퀀스 투 시퀀스 모델. 인코더에 의해 생성된 상태는 인코더와 디코더 간에 공유되는 유일한 정보입니다.](../img/seq2seq-state.svg)
:label:`fig_s2s_attention_state`

이것은 짧은 시퀀스에 대해서는 상당히 합리적이지만, 책의 장이나 매우 긴 문장 같은 긴 시퀀스에 대해서는 실현 불가능한 것이 분명합니다. 결국 머지않아 중간 표현에 소스 시퀀스에서 중요한 모든 것을 저장할 충분한 "공간"이 단순히 없게 될 것입니다. 결과적으로 디코더는 길고 복잡한 문장을 번역하는 데 실패할 것입니다. 이를 가장 먼저 마주친 사람 중 하나는 손글씨 텍스트를 생성하기 위한 RNN을 설계하려 했던 :citet:`Graves.2013` 입니다. 소스 텍스트가 임의의 길이를 가지므로 그들은 텍스트 문자를 훨씬 더 긴 펜 자취와 정렬하기 위한 미분 가능한 어텐션 모델을 설계했는데, 여기서 정렬은 한 방향으로만 이동합니다. 이는 다시 음성 인식의 디코딩 알고리즘, 예를 들어 은닉 마르코프 모델 :cite:`rabiner1993fundamentals` 에 의존합니다.

정렬 학습의 아이디어에서 영감을 받아, :citet:`Bahdanau.Cho.Bengio.2014` 는 단방향 정렬 제한이 *없는* 미분 가능한 어텐션 모델을 제안했습니다.
토큰을 예측할 때, 모든 입력 토큰이 관련이 있는 것이 아니라면, 모델은 현재 예측과 관련이 있다고 여겨지는 입력 시퀀스의 부분들에만 정렬(또는 주의)합니다. 그런 다음 이것이 다음 토큰을 생성하기 전에 현재 상태를 업데이트하는 데 사용됩니다. 그 설명에서는 상당히 무해해 보이지만, 이 *Bahdanau 어텐션 메커니즘* 은 어떤 의미에서 지난 10년간 딥러닝에서 가장 영향력 있는 아이디어 중 하나가 되었으며, 트랜스포머 :cite:`Vaswani.Shazeer.Parmar.ea.2017` 와 많은 관련된 새로운 아키텍처를 낳았습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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
```

## 모델

저희는 :numref:`sec_seq2seq` 의 시퀀스 투 시퀀스 아키텍처, 특히 :eqref:`eq_seq2seq_s_t` 에 의해 소개된 표기법을 따릅니다.
핵심 아이디어는 소스 문장을 요약하는 상태, 즉 컨텍스트 변수 $\mathbf{c}$ 를 고정된 채로 유지하는 대신, 원본 텍스트(인코더 은닉 상태 $\mathbf{h}_{t}$)와 이미 생성된 텍스트(디코더 은닉 상태 $\mathbf{s}_{t'-1}$) 모두의 함수로서 동적으로 업데이트한다는 것입니다. 이는 어떤 디코딩 시간 단계 $t'$ 이후에 업데이트되는 $\mathbf{c}_{t'}$ 를 산출합니다. 입력 시퀀스가 길이 $T$ 라고 가정합시다. 이 경우 컨텍스트 변수는 어텐션 풀링의 출력입니다.

$$\mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}.$$

저희는 $\mathbf{s}_{t' - 1}$ 을 쿼리로, $\mathbf{h}_{t}$ 를 키와 값 모두로 사용했습니다. $\mathbf{c}_{t'}$ 가 그런 다음 상태 $\mathbf{s}_{t'}$ 와 새로운 토큰을 생성하는 데 사용된다는 점에 유의하십시오: :eqref:`eq_seq2seq_s_t` 를 보십시오. 특히 어텐션 가중치 $\alpha$ 는 :eqref:`eq_additive-attn` 에 의해 정의된 가산 어텐션 스코어링 함수를 사용해 :eqref:`eq_attn-scoring-alpha` 에서와 같이 계산됩니다.
어텐션을 사용한 이 RNN 인코더-디코더 아키텍처는 :numref:`fig_s2s_attention_details` 에 묘사되어 있습니다. 나중에 이 모델은 이미 생성된 토큰을 디코더에 추가 컨텍스트로 포함하도록 수정되었다는 점에 유의하십시오(즉, 어텐션 합은 $T$ 에서 멈추지 않고 오히려 $t'-1$ 까지 진행됩니다). 예를 들어 음성 인식에 적용된 이 전략에 대한 설명은 :citet:`chan2015listen` 을 참조하십시오.

![Bahdanau 어텐션 메커니즘을 사용한 RNN 인코더-디코더 모델의 계층들.](../img/seq2seq-details-attention.svg)
:label:`fig_s2s_attention_details`

## 어텐션을 사용한 디코더 정의

어텐션을 사용한 RNN 인코더-디코더를 구현하기 위해, 저희는 디코더만 다시 정의하면 됩니다(어텐션 함수에서 생성된 기호를 생략하면 설계가 단순화됩니다). 상당히 놀랍지 않게 이름 지어진 `AttentionDecoder` 클래스를 정의함으로써 [**어텐션을 사용한 디코더에 대한 기본 인터페이스**]로 시작합시다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class AttentionDecoder(d2l.Decoder):  #@save
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

저희는 `Seq2SeqAttentionDecoder` 클래스에서 [**RNN 디코더를 구현**]해야 합니다.
디코더의 상태는 다음으로 초기화됩니다.
(i) 모든 시간 단계에서의 인코더 마지막 층의 은닉 상태로, 어텐션의 키와 값으로 사용됩니다;
(ii) 마지막 시간 단계에서의 모든 층에서의 인코더 은닉 상태로, 디코더의 은닉 상태를 초기화하는 역할을 합니다;
(iii) 인코더의 유효 길이로, 어텐션 풀링에서 패딩 토큰을 제외합니다.
각 디코딩 시간 단계에서, 이전 시간 단계에서 얻은 디코더의 마지막 층의 은닉 상태가 어텐션 메커니즘의 쿼리로 사용됩니다.
어텐션 메커니즘의 출력과 입력 임베딩 모두 연결되어(concatenated) RNN 디코더의 입력으로 사용됩니다.

```{.python .input}
%%tab mxnet
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = np.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            hidden_state = hidden_state[0]
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Concatenate on the feature dimension
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (batch_size, num_steps, num_hiddens).
        # Length of list hidden_state is num_layers, where the shape of its
        # element is (batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (tf.transpose(outputs, (1, 0, 2)), hidden_state,
                enc_valid_lens)

    def call(self, X, state, **kwargs):
        # Shape of output enc_outputs: # (batch_size, num_steps, num_hiddens)
        # Length of list hidden_state is num_layers, where the shape of its
        # element is (batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X)  # Input X has shape: (batch_size, num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Concatenate on the feature dimension
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (batch_size, num_steps, vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class Seq2SeqAttentionDecoder(nn.Module):
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.attention = d2l.AdditiveAttention(self.num_hiddens, self.dropout)
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.dense = nn.Dense(self.vocab_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout=self.dropout)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        # Attention Weights are returned as part of state; init with None
        return (outputs.transpose(1, 0, 2), hidden_state, enc_valid_lens)

    @nn.compact
    def __call__(self, X, state, training=False):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        # Ignore Attention value in state
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).transpose(1, 0, 2)
        outputs, attention_weights = [], []
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            query = jnp.expand_dims(hidden_state[-1], axis=1)
            # Shape of context: (batch_size, 1, num_hiddens)
            context, attention_w = self.attention(query, enc_outputs,
                                                  enc_outputs, enc_valid_lens,
                                                  training=training)
            # Concatenate on the feature dimension
            x = jnp.concatenate((context, jnp.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.transpose(1, 0, 2), hidden_state,
                                         training=training)
            outputs.append(out)
            attention_weights.append(attention_w)

        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'dec_attention_weights', attention_weights)

        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(jnp.concatenate(outputs, axis=0))
        return outputs.transpose(1, 0, 2), [enc_outputs, hidden_state,
                                            enc_valid_lens]
```

다음에서 저희는 각각 7개의 시간 단계 길이인 네 시퀀스의 미니배치를 사용하여 어텐션을 가진 [**구현된 디코더를 테스트**]합니다.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
if tab.selected('mxnet'):
    X = d2l.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('pytorch'):
    X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('tensorflow'):
    X = tf.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X, training=False), None)
    output, state = decoder(X, state, training=False)
if tab.selected('jax'):
    X = jnp.zeros((batch_size, num_steps), dtype=jnp.int32)
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(),
                                                        X, training=False)[0],
                               None)
    (output, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                  state, training=False)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

## [**학습**]

이제 새로운 디코더를 지정했으므로 저희는 :numref:`sec_seq2seq_training` 와 유사하게 진행할 수 있습니다: 초매개변수를 지정하고, 일반 인코더와 어텐션을 가진 디코더를 인스턴스화하고, 기계 번역을 위해 이 모델을 학습시킵니다.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = d2l.Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = d2l.Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

모델이 학습된 후, 저희는 그것을 사용해 [**몇 개의 영어 문장을 프랑스어로 번역**]하고 그 BLEU 점수를 계산합니다.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(
        trainer.state.params, data.build(engs, fras), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

마지막 영어 문장을 번역할 때 [**어텐션 가중치를 시각화**]해 봅시다.
저희는 각 쿼리가 키-값 쌍에 대해 균등하지 않은 가중치를 할당하는 것을 봅니다.
이는 각 디코딩 단계에서, 입력 시퀀스의 서로 다른 부분이 어텐션 풀링에서 선택적으로 집계됨을 보여줍니다.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
if tab.selected('jax'):
    _, (dec_attention_weights, _) = model.predict_step(
        trainer.state.params, data.build([engs[-1]], [fras[-1]]),
        data.num_steps, True)
attention_weights = d2l.concat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = d2l.reshape(attention_weights, (1, 1, -1, data.num_steps))
```

```{.python .input}
%%tab mxnet
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab pytorch
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab tensorflow
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

```{.python .input}
%%tab jax
# Plus one to include the end-of-sequence token
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key positions', ylabel='Query positions')
```

## 요약

토큰을 예측할 때 모든 입력 토큰이 관련 있는 것이 아니라면, Bahdanau 어텐션 메커니즘을 가진 RNN 인코더-디코더는 입력 시퀀스의 서로 다른 부분들을 선택적으로 집계합니다. 이는 상태(컨텍스트 변수)를 가산 어텐션 풀링의 출력으로 취급함으로써 달성됩니다.
RNN 인코더-디코더에서 Bahdanau 어텐션 메커니즘은 이전 시간 단계의 디코더 은닉 상태를 쿼리로 취급하고, 모든 시간 단계의 인코더 은닉 상태를 키와 값 모두로 취급합니다.


## 연습문제

1. 실험에서 GRU를 LSTM으로 교체하십시오.
1. 가산 어텐션 스코어링 함수를 스케일드 내적으로 교체하도록 실험을 수정하십시오. 이것이 학습 효율성에 어떻게 영향을 미칩니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3868)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18028)
:end_tab:

