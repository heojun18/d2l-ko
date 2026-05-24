# 심층 순환 신경망 (Deep Recurrent Neural Networks)

:label:`sec_deep_rnn`

지금까지 저희는 시퀀스 입력, 단일 은닉 RNN 레이어, 그리고 출력 레이어로 구성된 신경망을 정의하는 데 초점을 맞춰왔습니다. 어떤 시간 단계에서의 입력과 해당 출력 사이에 은닉층이 단 하나만 있음에도 불구하고, 이러한 신경망이 깊다고 볼 수 있는 측면이 있습니다. 첫 번째 시간 단계의 입력은 최종 시간 단계 $T$ (종종 수백 또는 수천 단계 이후)에서의 출력에 영향을 미칠 수 있습니다. 이러한 입력은 최종 출력에 도달하기 전에 순환 레이어의 $T$번 적용을 거쳐 전달됩니다. 그러나, 저희는 주어진 시간 단계의 입력과 같은 시간 단계의 출력 사이의 복잡한 관계를 표현하는 능력도 종종 유지하기를 원합니다. 따라서, 저희는 종종 시간 방향뿐만 아니라 입력에서 출력으로의 방향으로도 깊은 RNN을 구성합니다. 이는 정확히 저희가 MLP와 심층 CNN을 발전시키면서 이미 마주했던 깊이의 개념입니다.


이런 종류의 심층 RNN을 구축하기 위한 표준 방법은 놀랍도록 간단합니다. RNN을 서로 위에 쌓는 것입니다. 길이 $T$의 시퀀스가 주어지면, 첫 번째 RNN은 길이 $T$의 출력 시퀀스를 생성합니다. 이는 차례로 다음 RNN 레이어의 입력을 구성합니다. 이 짧은 절에서는 이 설계 패턴을 보여주고, 그러한 누적된(stacked) RNN을 어떻게 코드로 작성하는지에 대한 간단한 예시를 제시합니다. 아래의 :numref:`fig_deep_rnn`에서는 $L$개의 은닉층을 가진 심층 RNN을 보여줍니다. 각 은닉 상태는 순차적 입력에 대해 작동하고 순차적 출력을 생성합니다. 또한, 각 시간 단계에서 어떤 RNN 셀(:numref:`fig_deep_rnn`의 흰색 상자)이든 이전 시간 단계에서의 같은 레이어의 값과 같은 시간 단계에서의 이전 레이어의 값 모두에 의존합니다.

![심층 RNN의 아키텍처.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

형식적으로, 시간 단계 $t$에서 미니배치 입력 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (예제의 수 $=n$; 각 예제의 입력 수 $=d$)를 가지고 있다고 가정합시다. 같은 시간 단계에서, $l^\textrm{번째}$ 은닉층($l=1,\ldots,L$)의 은닉 상태를 $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$ (은닉 유닛의 수 $=h$)로, 그리고 출력 레이어 변수를 $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (출력의 수: $q$)로 둡니다. $\mathbf{H}_t^{(0)} = \mathbf{X}_t$로 설정하면, 활성화 함수 $\phi_l$을 사용하는 $l^\textrm{번째}$ 은닉층의 은닉 상태는 다음과 같이 계산됩니다.

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

여기서 가중치 $\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$와 $\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$, 그리고 편향 $\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$는 $l^\textrm{번째}$ 은닉층의 모델 매개변수입니다.

마지막으로, 출력 레이어의 계산은 최종 $L^\textrm{번째}$ 은닉층의 은닉 상태에만 기반합니다.

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

여기서 가중치 $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$와 편향 $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$는 출력 레이어의 모델 매개변수입니다.

MLP에서처럼, 은닉층의 수 $L$과 은닉 유닛의 수 $h$는 저희가 조정할 수 있는 하이퍼파라미터입니다. 일반적인 RNN 레이어 폭($h$)은 $(64, 2056)$ 범위에 있고, 일반적인 깊이($L$)는 $(1, 8)$ 범위에 있습니다. 또한, :eqref:`eq_deep_rnn_H`의 은닉 상태 계산을 LSTM이나 GRU의 것으로 대체함으로써 쉽게 심층 게이트가 있는 RNN을 얻을 수 있습니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
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
import jax
from jax import numpy as jnp
```

## 처음부터 구현하기 (Implementation from Scratch)

다층 RNN을 처음부터 구현하기 위해, 각 레이어를 자체적인 학습 가능한 매개변수를 가진 `RNNScratch` 인스턴스로 다룰 수 있습니다.

```{.python .input}
%%tab mxnet, tensorflow
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = [d2l.RNNScratch(num_inputs if i==0 else num_hiddens,
                                    num_hiddens, sigma)
                     for i in range(num_layers)]
```

```{.python .input}
%%tab pytorch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

```{.python .input}
%%tab jax
class StackedRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    num_layers: int
    sigma: float = 0.01

    def setup(self):
        self.rnns = [d2l.RNNScratch(self.num_inputs if i==0 else self.num_hiddens,
                                    self.num_hiddens, self.sigma)
                     for i in range(self.num_layers)]
```

다층 순방향 계산은 단순히 레이어별로 순방향 계산을 수행합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    return outputs, Hs
```

예시로, 저희는 *The Time Machine* 데이터셋(:numref:`sec_rnn-scratch`와 같은)에 대해 심층 GRU 모델을 훈련합니다. 단순하게 유지하기 위해 레이어 수를 2로 설정합니다.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
    model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
        model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## 간결한 구현 (Concise Implementation)

:begin_tab:`pytorch, mxnet, tensorflow`
다행스럽게도 RNN의 여러 레이어를 구현하는 데 필요한 많은 실무적 세부 사항들이 고수준 API에서 쉽게 사용 가능합니다. 저희의 간결한 구현은 이러한 내장된 기능을 사용할 것입니다. 이 코드는 :numref:`sec_gru`에서 이전에 사용했던 것을 일반화하여, 단 한 개의 레이어라는 기본값을 선택하는 대신 레이어의 수를 명시적으로 지정할 수 있게 합니다.
:end_tab:

:begin_tab:`jax`
Flax는 RNN을 구현하는 데 미니멀리스트적 접근 방식을 취합니다. RNN의 레이어 수를 정의하거나 그것을 드롭아웃과 결합하는 것은 기본적으로 사용 가능하지 않습니다. 저희의 간결한 구현은 모든 내장된 기능을 사용하고 그 위에 `num_layers`와 `dropout` 기능을 추가할 것입니다. 이 코드는 :numref:`sec_gru`에서 이전에 사용했던 것을 일반화하여, 단일 레이어라는 기본값을 선택하는 대신 레이어의 수를 명시적으로 지정할 수 있게 합니다.
:end_tab:

```{.python .input}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True, time_major=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(X, state)
        return outputs, state
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                    (batch_size,), self.num_hiddens)] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # Introduce a dropout layer after every GRU layer except last
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU()(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # Final GRU layer without dropout
        out_state, X = GRU()(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)
```

하이퍼파라미터 선택과 같은 아키텍처적 결정은 :numref:`sec_gru`의 것과 매우 유사합니다. 저희는 구별되는 토큰의 수, 즉 `vocab_size`만큼의 동일한 수의 입력과 출력을 선택합니다. 은닉 유닛의 수는 여전히 32입니다. 유일한 차이점은 이제 (**`num_layers`의 값을 지정함으로써 자명하지 않은 수의 은닉층을 선택한다**)는 것입니다.

```{.python .input}
%%tab mxnet
gru = GRU(num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# Running takes > 1h (pending fix from MXNet)
# trainer.fit(model, data)
# model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab pytorch, tensorflow, jax
if tab.selected('tensorflow', 'jax'):
    gru = GRU(num_hiddens=32, num_layers=2)
if tab.selected('pytorch'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
if tab.selected('pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

```{.python .input}
%%tab pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

## 요약

심층 RNN에서, 은닉 상태 정보는 현재 레이어의 다음 시간 단계와 다음 레이어의 현재 시간 단계로 전달됩니다. LSTM, GRU, 또는 기본 RNN 등 심층 RNN의 다양한 종류가 존재합니다. 편리하게도, 이러한 모델들은 모두 딥러닝 프레임워크의 고수준 API의 일부로 사용 가능합니다. 모델의 초기화는 주의를 요합니다. 전반적으로, 심층 RNN은 적절한 수렴을 보장하기 위해 (학습률 및 클리핑과 같은) 상당한 양의 작업을 요구합니다.

## 연습문제

1. GRU를 LSTM으로 교체하고 정확도와 훈련 속도를 비교하십시오.
1. 여러 책을 포함하도록 훈련 데이터를 늘리십시오. 퍼플렉시티 척도에서 얼마나 낮출 수 있습니까?
1. 텍스트를 모델링할 때 서로 다른 저자의 출처를 결합하기를 원하시겠습니까? 왜 이것이 좋은 아이디어입니까? 무엇이 잘못될 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1058)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3862)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18018)
:end_tab:
