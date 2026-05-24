# 순환 신경망의 간결한 구현
:label:`sec_rnn-concise`

저희의 대부분의 처음부터 구현과 마찬가지로,
:numref:`sec_rnn-scratch`는 각 구성요소가 어떻게 작동하는지에 대한
통찰을 제공하기 위해 설계되었습니다.
그러나 RNN을 매일 사용하거나
프로덕션 코드를 작성할 때는,
구현 시간을 줄이고(흔한 모델과 함수를 위한 라이브러리 코드를 제공함으로써)
계산 시간도 줄이는(이러한 라이브러리 구현을 극도로 최적화함으로써)
라이브러리에 더 의존하고 싶을 것입니다.
이 절에서는 여러분의 딥러닝 프레임워크가 제공하는
고수준 API를 사용하여
같은 언어 모델을 더 효율적으로 구현하는 방법을
보여드릴 것입니다.
이전과 마찬가지로, 저희는
*The Time Machine* 데이터셋을 로드하는 것으로 시작합니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
```

## [**모델 정의하기**]

저희는 고수준 API에 의해 구현된 RNN을 사용하여
다음 클래스를 정의합니다.

:begin_tab:`mxnet`
구체적으로, 은닉 상태를 초기화하기 위해,
저희는 멤버 메서드 `begin_state`를 호출합니다.
이는 미니배치의 각 예시에 대한 초기 은닉 상태를
포함하는 리스트를 반환하며,
그 모양은
(은닉 층의 수, 배치 크기, 은닉 유닛의 수)입니다.
이후에 소개될 일부 모델
(예: 장단기 메모리(long short-term memory))의 경우,
이 리스트에는 다른 정보도 포함될 것입니다.
:end_tab:

:begin_tab:`jax`
Flax는 오늘 기준으로 Vanilla RNN의 간결한 구현을 위한 RNNCell을
제공하지 않습니다. Flax `linen` API에서 사용 가능한 LSTM과 GRU 같은
RNN의 더 고급 변형들이 있습니다.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """The RNN model implemented with high-level APIs."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

:numref:`sec_rnn-scratch`의 `RNNLMScratch` 클래스로부터 상속받아,
다음 `RNNLM` 클래스는 완전한 RNN 기반 언어 모델을 정의합니다.
저희가 별도의 완전 연결 출력 층을 만들어야 한다는 점에 유의하세요.

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """The RNN-based language model implemented with high-level APIs."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## 학습과 예측

모델을 학습시키기 전에, [**무작위 가중치로 초기화된 모델로
예측을 해 봅시다.**]
저희가 신경망을 학습시키지 않았으므로,
그것은 무의미한 예측을 생성할 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

다음으로, 저희는 [**고수준 API를 활용하여, 모델을 학습시킵니다**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

:numref:`sec_rnn-scratch`와 비교하면,
이 모델은 비슷한 펄플렉서티를 달성하지만,
최적화된 구현으로 인해 더 빠르게 실행됩니다.
이전과 마찬가지로, 저희는 지정된 접두사 문자열을 따라
예측된 토큰을 생성할 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## 요약

딥러닝 프레임워크의 고수준 API는 표준 RNN의 구현을 제공합니다.
이러한 라이브러리는 표준 모델을 다시 구현하는 데 시간을 낭비하지 않게 도와줍니다.
더욱이,
프레임워크 구현은 종종 고도로 최적화되어 있어,
처음부터 구현한 것과 비교했을 때
상당한 (계산적) 성능 향상으로 이어집니다.

## 연습문제

1. 고수준 API를 사용하여 RNN 모델을 과대적합시킬 수 있나요?
1. RNN을 사용하여 :numref:`sec_sequence`의 자기회귀 모델을 구현하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18015)
:end_tab:
