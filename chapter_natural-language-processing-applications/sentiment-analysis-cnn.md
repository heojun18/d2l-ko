# 감성 분석: 합성곱 신경망 사용하기
:label:`sec_sentiment_cnn` 


:numref:`chap_cnn`에서,
저희는 인접한 픽셀과 같은
국소적 특징에 적용되는
2차원 CNN을 사용하여
2차원 이미지 데이터를
처리하는 메커니즘을
살펴보았습니다.
원래는
컴퓨터 비전을 위해 설계되었지만,
CNN은 자연어 처리에도
폭넓게 사용됩니다.
간단히 말해,
어떠한 텍스트 시퀀스든
1차원 이미지로 생각하면 됩니다.
이런 방식으로,
1차원 CNN은
텍스트의 $n$-그램과 같은
국소적 특징을 처리할 수 있습니다.

이 절에서는,
*textCNN* 모델을 사용하여
단일 텍스트를 표현하기 위한 CNN 아키텍처를
설계하는 방법을
보여드릴 것입니다 :cite:`Kim.2014`.
감성 분석을 위해
GloVe 사전 학습과 함께 RNN 아키텍처를 사용하는
:numref:`fig_nlp-map-sa-rnn`과 비교했을 때,
:numref:`fig_nlp-map-sa-cnn`의 유일한 차이점은
아키텍처의 선택에 있습니다.


![이 절에서는 감성 분석을 위해 사전 학습된 GloVe를 CNN 기반 아키텍처에 입력합니다.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
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

## 1차원 합성곱

모델을 소개하기 전에,
1차원 합성곱이 어떻게 작동하는지 살펴보겠습니다.
이것은 교차 상관 연산에 기반한
2차원 합성곱의 특수한 경우일 뿐임을
명심하세요.

![1차원 교차 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소이며 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

:numref:`fig_conv1d`에 나타난 것처럼,
1차원의 경우,
합성곱 윈도우는
입력 텐서를 가로질러
왼쪽에서 오른쪽으로 슬라이딩합니다.
슬라이딩 동안,
특정 위치에서 합성곱 윈도우에 포함된
입력 부분 텐서(예: :numref:`fig_conv1d`의 $0$과 $1$)와
커널 텐서(예: :numref:`fig_conv1d`의 $1$과 $2$)가
원소별로 곱해집니다.
이러한 곱셈의 합이
출력 텐서의 해당 위치에서
단일 스칼라 값(예: :numref:`fig_conv1d`의 $0\times1+1\times2=2$)을
제공합니다.

다음 `corr1d` 함수에서 1차원 교차 상관을 구현합니다.
입력 텐서 `X`와
커널 텐서 `K`가 주어지면,
출력 텐서 `Y`를 반환합니다.

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

위의 1차원 교차 상관 구현의 출력을 검증하기 위해 :numref:`fig_conv1d`에서 입력 텐서 `X`와 커널 텐서 `K`를 구성할 수 있습니다.

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

여러 채널을 가진
어떤 1차원 입력에 대해서도,
합성곱 커널은
같은 수의 입력 채널을 가져야 합니다.
그런 다음 각 채널에 대해,
입력의 1차원 텐서와 합성곱 커널의 1차원 텐서에 대해 교차 상관 연산을 수행하고,
모든 채널에 대해 결과를 합산하여
1차원 출력 텐서를 생성합니다.
:numref:`fig_conv1d_channel`은 3개의 입력 채널을 가진 1차원 교차 상관 연산을 보여줍니다.

![3개의 입력 채널을 가진 1차원 교차 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소이며 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`


여러 입력 채널에 대한 1차원 교차 상관 연산을 구현하고
:numref:`fig_conv1d_channel`의 결과를 검증할 수 있습니다.

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

다중 입력 채널 1차원 교차 상관은
단일 입력 채널
2차원 교차 상관과
동등하다는 점에
유의하세요.
설명하자면,
:numref:`fig_conv1d_channel`의
다중 입력 채널 1차원 교차 상관의
동등한 형태는
:numref:`fig_conv1d_2d`의
단일 입력 채널
2차원 교차 상관이며,
여기서 합성곱 커널의 높이는
입력 텐서의 높이와 같아야 합니다.


![단일 입력 채널을 가진 2차원 교차 상관 연산. 음영 처리된 부분은 첫 번째 출력 요소이며 출력 계산에 사용된 입력 및 커널 텐서 요소입니다: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

:numref:`fig_conv1d`와 :numref:`fig_conv1d_channel`의 출력은 모두 단 하나의 채널만을 가집니다.
:numref:`subsec_multi-output-channels`에 설명된 다중 출력 채널을 가진 2차원 합성곱과 마찬가지로,
1차원 합성곱에 대해서도
다중 출력 채널을 지정할 수 있습니다.

## 시간 차원 최대 풀링(Max-Over-Time Pooling)

유사하게, 저희는 풀링을 사용하여
시퀀스 표현에서 가장 높은 값을
시간 스텝에 걸친 가장 중요한 특징으로
추출할 수 있습니다.
textCNN에서 사용되는 *시간 차원 최대 풀링(max-over-time pooling)*은
1차원 전역 최대 풀링과
같은 방식으로 작동합니다
:cite:`Collobert.Weston.Bottou.ea.2011`.
각 채널이 서로 다른 시간 스텝에서의 값을 저장하는
다중 채널 입력에 대해,
각 채널에서의 출력은
해당 채널에 대한
최댓값입니다.
시간 차원 최대 풀링은
서로 다른 채널에서
서로 다른 수의 시간 스텝을
허용한다는 점에 유의하세요.

## textCNN 모델

1차원 합성곱과
시간 차원 최대 풀링을 사용하여,
textCNN 모델은
개별적인 사전 학습된 토큰 표현을
입력으로 받아,
다운스트림 응용을 위한
시퀀스 표현을 얻고 변환합니다.

$d$차원 벡터로 표현된
$n$개의 토큰을 가진
단일 텍스트 시퀀스에 대해,
입력 텐서의
너비, 높이, 채널 수는
각각 $n$, $1$, $d$입니다.
textCNN 모델은 입력을
다음과 같이 출력으로 변환합니다:

1. 여러 개의 1차원 합성곱 커널을 정의하고 입력에 대해 합성곱 연산을 별도로 수행합니다. 서로 다른 너비의 합성곱 커널은 서로 다른 수의 인접한 토큰들 간의 국소적 특징을 포착할 수 있습니다.
1. 모든 출력 채널에 대해 시간 차원 최대 풀링을 수행한 다음, 모든 스칼라 풀링 출력을 벡터로 연결합니다.
1. 연결된 벡터를 완전 연결 레이어를 사용하여 출력 범주로 변환합니다. 과적합을 줄이기 위해 드롭아웃을 사용할 수 있습니다.

![textCNN의 모델 아키텍처.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn`은
구체적인 예와 함께 textCNN의 모델 아키텍처를
보여줍니다.
입력은 11개의 토큰을 가진 문장이며,
각 토큰은 6차원 벡터로 표현됩니다.
따라서 너비 11의 6채널 입력을 가집니다.
너비 2와 4의
두 개의 1차원 합성곱 커널을 정의하며,
각각 4개와 5개의 출력 채널을 가집니다.
이들은
너비 $11-2+1=10$의 4개의 출력 채널과
너비 $11-4+1=8$의 5개의 출력 채널을
생성합니다.
이 9개의 채널이 서로 다른 너비를 가짐에도 불구하고,
시간 차원 최대 풀링은
연결된 9차원 벡터를 제공하며,
이는 마침내 이진 감정 예측을 위한
2차원 출력 벡터로
변환됩니다.



### 모델 정의하기

다음 클래스에서 textCNN 모델을 구현합니다.
:numref:`sec_sentiment_rnn`의
양방향 RNN 모델과 비교했을 때,
순환 레이어를 합성곱 레이어로 대체한 것 외에도,
저희는 두 개의 임베딩 레이어를 사용합니다:
하나는 학습 가능한 가중치를 가지며 다른 하나는
고정된 가중치를 가집니다.

```{.python .input}
#@tab mxnet
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

textCNN 인스턴스를 생성해 봅시다.
이는 커널 너비가 3, 4, 5인 3개의 합성곱 레이어를 가지며, 모두 100개의 출력 채널을 가집니다.

```{.python .input}
#@tab mxnet
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)

net.apply(init_weights);
```

### 사전 학습된 단어 벡터 로딩하기

:numref:`sec_sentiment_rnn`과 마찬가지로,
저희는 초기화된 토큰 표현으로
사전 학습된 100차원 GloVe 임베딩을 로드합니다.
이러한 토큰 표현(임베딩 가중치)은
`embedding`에서 학습되고
`constant_embedding`에서 고정됩니다.

```{.python .input}
#@tab mxnet
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### 모델 학습 및 평가

이제 감성 분석을 위해 textCNN 모델을 학습할 수 있습니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

아래에서 저희는 학습된 모델을 사용하여 두 개의 간단한 문장에 대한 감정을 예측합니다.

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## 요약

* 1차원 CNN은 텍스트의 $n$-그램과 같은 국소적 특징을 처리할 수 있습니다.
* 다중 입력 채널 1차원 교차 상관은 단일 입력 채널 2차원 교차 상관과 동등합니다.
* 시간 차원 최대 풀링은 서로 다른 채널에서 서로 다른 수의 시간 스텝을 허용합니다.
* textCNN 모델은 1차원 합성곱 레이어와 시간 차원 최대 풀링 레이어를 사용하여 개별 토큰 표현을 다운스트림 응용 출력으로 변환합니다.


## 연습문제

1. 하이퍼파라미터를 조정하고 :numref:`sec_sentiment_rnn`과 이 절의 감성 분석을 위한 두 아키텍처를 분류 정확도와 계산 효율성 등의 측면에서 비교해 보세요.
1. :numref:`sec_sentiment_rnn`의 연습문제에서 소개된 방법을 사용하여 모델의 분류 정확도를 더욱 향상시킬 수 있겠습니까?
1. 입력 표현에 위치 인코딩을 추가하세요. 이것이 분류 정확도를 향상시키나요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1425)
:end_tab:
