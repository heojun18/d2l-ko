# 게이트 순환 유닛 (Gated Recurrent Units, GRU)
:label:`sec_gru`


2010년대 동안 RNN, 특히 LSTM 아키텍처(:numref:`sec_lstm`)가 급속도로 인기를 얻으면서, 많은 연구자들이 내부 상태와 곱셈적 게이팅 메커니즘을 통합한다는 핵심 아이디어는 유지하되 계산 속도를 높이는 것을 목표로 단순화된 아키텍처를 실험하기 시작했습니다. 게이트 순환 유닛(GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`은 LSTM 메모리 셀의 간소화된 버전을 제공했으며, 이는 종종 비슷한 성능을 달성하지만 계산이 더 빠르다는 장점을 가집니다 :cite:`Chung.Gulcehre.Cho.ea.2014`.

```{.python .input  n=5}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=6}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=8}
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

## 리셋 게이트와 업데이트 게이트 (Reset Gate and Update Gate)

여기서, LSTM의 세 게이트는 두 개, 즉 *리셋 게이트(reset gate)*와 *업데이트 게이트(update gate)*로 대체됩니다. LSTM과 마찬가지로, 이 게이트들에는 시그모이드 활성화가 주어져, 그 값이 $(0, 1)$ 구간에 놓이도록 강제됩니다. 직관적으로, 리셋 게이트는 저희가 여전히 기억하고 싶은 이전 상태의 양을 제어합니다. 마찬가지로, 업데이트 게이트는 새 상태의 얼마만큼이 단순히 이전 상태의 복사본인지를 제어할 수 있게 해줍니다. :numref:`fig_gru_1`은 현재 시간 단계의 입력과 이전 시간 단계의 은닉 상태가 주어졌을 때, GRU에서 리셋 게이트와 업데이트 게이트 양쪽 모두에 대한 입력을 보여줍니다. 게이트의 출력은 시그모이드 활성화 함수를 가진 두 개의 완전 연결 레이어에 의해 주어집니다.

![GRU 모델에서의 리셋 게이트와 업데이트 게이트 계산.](../img/gru-1.svg)
:label:`fig_gru_1`

수학적으로, 주어진 시간 단계 $t$에 대해, 입력이 미니배치 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (예제의 수 $=n$; 입력의 수 $=d$)이고 이전 시간 단계의 은닉 상태가 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (은닉 유닛의 수 $=h$)라고 가정합시다. 그러면 리셋 게이트 $\mathbf{R}_t \in \mathbb{R}^{n \times h}$와 업데이트 게이트 $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$는 다음과 같이 계산됩니다.

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z}),
\end{aligned}
$$

여기서 $\mathbf{W}_{\textrm{xr}}, \mathbf{W}_{\textrm{xz}} \in \mathbb{R}^{d \times h}$와 $\mathbf{W}_{\textrm{hr}}, \mathbf{W}_{\textrm{hz}} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고 $\mathbf{b}_\textrm{r}, \mathbf{b}_\textrm{z} \in \mathbb{R}^{1 \times h}$는 편향 매개변수입니다.


## 후보 은닉 상태 (Candidate Hidden State)

다음으로, 저희는 리셋 게이트 $\mathbf{R}_t$를 :eqref:`rnn_h_with_state`에 있는 일반적인 업데이트 메커니즘과 통합하여, 시간 단계 $t$에서 다음과 같은 *후보 은닉 상태(candidate hidden state)* $\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$로 이어집니다.

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$
:eqlabel:`gru_tilde_H`

여기서 $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$와 $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고, $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$는 편향이며, 기호 $\odot$는 아다마르(원소별) 곱셈 연산자입니다. 여기서는 tanh 활성화 함수를 사용합니다.

업데이트 게이트의 작용을 여전히 통합해야 하기 때문에, 그 결과는 *후보(candidate)*입니다. :eqref:`rnn_h_with_state`와 비교하면, 이전 상태들의 영향력은 이제 :eqref:`gru_tilde_H`에서 $\mathbf{R}_t$와 $\mathbf{H}_{t-1}$의 원소별 곱셈으로 감소될 수 있습니다. 리셋 게이트 $\mathbf{R}_t$의 항목들이 1에 가까울 때마다, 저희는 :eqref:`rnn_h_with_state`와 같은 기본 RNN을 복원합니다. 리셋 게이트 $\mathbf{R}_t$의 모든 항목이 0에 가까울 때는, 후보 은닉 상태가 $\mathbf{X}_t$를 입력으로 하는 MLP의 결과입니다. 따라서 기존의 모든 은닉 상태는 기본값으로 *리셋*됩니다.

:numref:`fig_gru_2`는 리셋 게이트를 적용한 후의 계산 흐름을 보여줍니다.

![GRU 모델에서의 후보 은닉 상태 계산.](../img/gru-2.svg)
:label:`fig_gru_2`


## 은닉 상태 (Hidden State)

마지막으로, 업데이트 게이트 $\mathbf{Z}_t$의 효과를 통합할 필요가 있습니다. 이는 새 은닉 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$가 새 후보 상태 $\tilde{\mathbf{H}}_t$를 닮은 정도와 비교하여, 이전 상태 $\mathbf{H}_{t-1}$과 얼마나 일치하는지의 정도를 결정합니다. 업데이트 게이트 $\mathbf{Z}_t$는 단순히 $\mathbf{H}_{t-1}$과 $\tilde{\mathbf{H}}_t$의 원소별 볼록 결합을 취함으로써, 이 목적에 사용될 수 있습니다. 이는 GRU의 최종 업데이트 방정식으로 이어집니다.

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$


업데이트 게이트 $\mathbf{Z}_t$가 1에 가까울 때마다, 저희는 단순히 이전 상태를 유지합니다. 이 경우 $\mathbf{X}_t$로부터의 정보는 무시되며, 의존성 사슬에서 시간 단계 $t$를 효과적으로 건너뜁니다. 반대로, $\mathbf{Z}_t$가 0에 가까울 때마다, 새로운 잠재 상태 $\mathbf{H}_t$는 후보 잠재 상태 $\tilde{\mathbf{H}}_t$에 접근합니다. :numref:`fig_gru_3`은 업데이트 게이트가 작동한 후의 계산 흐름을 보여줍니다.

![GRU 모델에서의 은닉 상태 계산.](../img/gru-3.svg)
:label:`fig_gru_3`


요약하면, GRU는 다음과 같은 두 가지 구별되는 특징을 가집니다.

* 리셋 게이트는 시퀀스에서 단기 의존성을 포착하는 데 도움이 됩니다.
* 업데이트 게이트는 시퀀스에서 장기 의존성을 포착하는 데 도움이 됩니다.

## 처음부터 구현하기 (Implementation from Scratch)

GRU 모델을 더 잘 이해하기 위해, 처음부터 구현해 봅시다.

### (**모델 매개변수 초기화**)

첫 번째 단계는 모델 매개변수를 초기화하는 것입니다. 표준편차가 `sigma`인 가우시안 분포로부터 가중치를 추출하고 편향은 0으로 설정합니다. 하이퍼파라미터 `num_hiddens`는 은닉 유닛의 수를 정의합니다. 업데이트 게이트, 리셋 게이트, 후보 은닉 상태와 관련된 모든 가중치와 편향을 인스턴스화합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        if tab.selected('mxnet'):
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))            
        if tab.selected('pytorch'):
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        if tab.selected('tensorflow'):
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))            
            
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state        
```

```{.python .input}
%%tab jax
class GRUScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        init_weight = lambda name, shape: self.param(name,
                                                     nn.initializers.normal(self.sigma),
                                                     shape)
        triple = lambda name : (
            init_weight(f'W_x{name}', (self.num_inputs, self.num_hiddens)),
            init_weight(f'W_h{name}', (self.num_hiddens, self.num_hiddens)),
            self.param(f'b_{name}', nn.initializers.zeros, (self.num_hiddens)))

        self.W_xz, self.W_hz, self.b_z = triple('z')  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple('r')  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple('h')  # Candidate hidden state
```

### 모델 정의하기

이제 [**GRU 순방향 계산을 정의**]할 준비가 되었습니다. 그 구조는 기본 RNN 셀의 구조와 동일하지만, 업데이트 방정식이 더 복잡합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
    outputs = []
    for X in inputs:
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) +
                        d2l.matmul(H, self.W_hz) + self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) + 
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) + 
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H
```

```{.python .input}
%%tab jax
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    # Use lax.scan primitive instead of looping over the
    # inputs, since scan saves time in jit compilation
    def scan_fn(H, X):
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) + d2l.matmul(H, self.W_hz) +
                        self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) +
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) +
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        return H, H  # return carry, y

    if H is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H

    # scan takes the scan_fn, initial carry state, xs with leading axis to be scanned
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### 훈련

*The Time Machine* 데이터셋에 대해 언어 모델을 [**훈련**]하는 것은 :numref:`sec_rnn-scratch`에서와 정확히 같은 방식으로 동작합니다.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**간결한 구현**]

고수준 API에서는 GRU 모델을 직접 인스턴스화할 수 있습니다. 이는 저희가 위에서 명시했던 모든 구성 세부 사항을 캡슐화합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens)
        if tab.selected('tensorflow'):
            self.rnn = tf.keras.layers.GRU(num_hiddens, return_sequences=True, 
                                           return_state=True)
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None, training=False):
        if H is None:
            batch_size = inputs.shape[1]
            H = nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                                            (batch_size,), self.num_hiddens)

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        H, outputs = GRU()(H, inputs)
        return outputs, H
```

이 코드는 Python 대신 컴파일된 연산자를 사용하므로 훈련에서 상당히 더 빠릅니다.

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('jax'):
    gru = GRU(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

훈련 후에, 저희는 훈련 세트에 대한 퍼플렉시티와 제공된 접두사 다음에 이어지는 예측된 시퀀스를 출력합니다.

```{.python .input}
%%tab mxnet, pytorch
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

LSTM과 비교하여, GRU는 유사한 성능을 달성하지만 계산적으로 더 가벼운 경향이 있습니다. 일반적으로 단순 RNN과 비교했을 때, LSTM과 GRU 같은 게이트가 있는 RNN은 큰 시간 단계 거리를 가진 시퀀스에 대한 의존성을 더 잘 포착할 수 있습니다. GRU는 리셋 게이트가 켜져 있을 때마다 기본 RNN을 극단적인 경우로 포함합니다. 또한 업데이트 게이트를 켜서 부분 시퀀스를 건너뛸 수도 있습니다.


## 연습문제

1. 시간 단계 $t' $에서의 입력만을 사용하여 시간 단계 $t > t'$에서의 출력을 예측하고 싶다고 가정해 봅시다. 각 시간 단계에 대한 리셋 게이트와 업데이트 게이트의 최적 값은 무엇입니까?
1. 하이퍼파라미터를 조정하고 실행 시간, 퍼플렉시티, 출력 시퀀스에 미치는 영향을 분석해 보십시오.
1. `rnn.RNN`과 `rnn.GRU` 구현 사이의 실행 시간, 퍼플렉시티, 출력 문자열을 서로 비교해 보십시오.
1. GRU의 일부분, 예를 들어 리셋 게이트만 또는 업데이트 게이트만을 구현한다면 어떻게 됩니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1056)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3860)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18017)
:end_tab:
