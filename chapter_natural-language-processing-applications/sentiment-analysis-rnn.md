# 감성 분석: 순환 신경망 사용하기
:label:`sec_sentiment_rnn` 


단어 유사도 및 유추 작업과 마찬가지로,
저희는 사전 학습된 단어 벡터를
감성 분석에도 적용할 수 있습니다.
:numref:`sec_sentiment`의
IMDb 리뷰 데이터셋은
그다지 크지 않으므로,
대규모 말뭉치에서 사전 학습된
텍스트 표현을 사용하면
모델의 과적합을 줄일 수 있습니다.
:numref:`fig_nlp-map-sa-rnn`에
나타난 구체적인 예로,
저희는 각 토큰을
사전 학습된 GloVe 모델을 사용하여 표현하고,
이 토큰 표현들을
다층 양방향 RNN에 입력하여
텍스트 시퀀스 표현을 얻을 것입니다.
이 표현은
감성 분석 출력으로
변환됩니다 :cite:`Maas.Daly.Pham.ea.2011`.
같은 다운스트림 응용에 대해,
저희는 나중에 다른 아키텍처
선택을 고려할 것입니다.

![이 절에서는 감성 분석을 위해 사전 학습된 GloVe를 RNN 기반 아키텍처에 입력합니다.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## RNN을 사용한 단일 텍스트 표현

감성 분석과 같은
텍스트 분류 작업에서는,
가변 길이 텍스트 시퀀스가
고정 길이 범주로 변환됩니다.
다음 `BiRNN` 클래스에서는,
텍스트 시퀀스의 각 토큰이
임베딩 레이어(`self.embedding`)를 통해
개별적인
사전 학습된 GloVe
표현을 얻는 반면,
전체 시퀀스는
양방향 RNN(`self.encoder`)에 의해 인코딩됩니다.
좀 더 구체적으로,
초기 시간 스텝과 최종 시간 스텝 모두에서의
양방향 LSTM의
(마지막 레이어의) 은닉 상태가
텍스트 시퀀스의 표현으로
연결됩니다.
이 단일 텍스트 표현은
두 개의 출력("긍정" 및 "부정")을 가지는
완전 연결 레이어(`self.decoder`)에 의해
출력 범주로 변환됩니다.

```{.python .input}
#@tab mxnet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) 
        outs = self.decoder(encoding)
        return outs
```

감성 분석을 위해 단일 텍스트를 표현하는 두 개의 은닉 레이어를 가진 양방향 RNN을 구성해 봅시다.

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
#@tab mxnet
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## 사전 학습된 단어 벡터 로딩하기

아래에서 어휘에 있는 토큰들에 대해 사전 학습된 100차원(`embed_size`와 일치해야 함) GloVe 임베딩을 로드합니다.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

어휘의 모든 토큰에 대한
벡터의 형상을 출력합니다.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

저희는 리뷰의 토큰을 표현하기 위해
이러한 사전 학습된 단어 벡터를 사용하며,
학습 동안 이 벡터들을
업데이트하지 않을 것입니다.

```{.python .input}
#@tab mxnet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## 모델 학습 및 평가

이제 감성 분석을 위해 양방향 RNN을 학습할 수 있습니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

학습된 모델 `net`을 사용하여 텍스트 시퀀스의 감정을 예측하는 다음 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

마지막으로, 학습된 모델을 사용하여 두 개의 간단한 문장에 대한 감정을 예측해 봅시다.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## 요약

* 사전 학습된 단어 벡터는 텍스트 시퀀스의 개별 토큰을 표현할 수 있습니다.
* 양방향 RNN은 초기 및 최종 시간 스텝에서의 은닉 상태를 연결하는 방식 등으로 텍스트 시퀀스를 표현할 수 있습니다. 이 단일 텍스트 표현은 완전 연결 레이어를 사용하여 범주로 변환될 수 있습니다.



## 연습문제

1. 에포크 수를 늘려 보세요. 학습 및 테스트 정확도를 향상시킬 수 있습니까? 다른 하이퍼파라미터를 조정하는 것은 어떻습니까?
1. 300차원 GloVe 임베딩과 같이 더 큰 사전 학습된 단어 벡터를 사용해 보세요. 분류 정확도가 향상됩니까?
1. spaCy 토큰화를 사용하여 분류 정확도를 향상시킬 수 있을까요? spaCy를 설치(`pip install spacy`)하고 영어 패키지를 설치(`python -m spacy download en`)해야 합니다. 코드에서 먼저 spaCy를 import(`import spacy`)합니다. 그런 다음 spaCy 영어 패키지를 로드(`spacy_en = spacy.load('en')`)합니다. 마지막으로, `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` 함수를 정의하고 원래의 `tokenizer` 함수를 대체합니다. GloVe와 spaCy에서 구문 토큰의 서로 다른 형태에 유의하세요. 예를 들어, 구문 토큰 "new york"는 GloVe에서는 "new-york" 형태이고, spaCy 토큰화 이후에는 "new york" 형태입니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1424)
:end_tab:
