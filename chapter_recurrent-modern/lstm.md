# 장단기 메모리 (Long Short-Term Memory, LSTM)
:label:`sec_lstm`


최초의 Elman 스타일 RNN이 역전파를 사용하여 훈련된 직후 :cite:`elman1990finding`, (기울기 소실 및 기울기 폭주로 인한) 장기 의존성을 학습하는 문제가 두드러지게 되었으며, Bengio와 Hochreiter가 이 문제를 논의했습니다 :cite:`bengio1994learning,Hochreiter.Bengio.Frasconi.ea.2001`. Hochreiter는 1991년 그의 석사 논문에서 이미 이 문제를 명확하게 설명했지만, 그 논문이 독일어로 쓰였기 때문에 그 결과는 널리 알려지지 않았습니다. 기울기 클리핑이 기울기 폭주에는 도움이 되지만, 기울기 소실을 다루는 것은 더 정교한 해결책이 필요해 보입니다. 기울기 소실을 해결하기 위한 최초이자 가장 성공적인 기법 중 하나는 :citet:`Hochreiter.Schmidhuber.1997`에 의한 장단기 메모리(LSTM) 모델의 형태로 등장했습니다. LSTM은 표준 순환 신경망과 비슷하지만, 여기서는 각각의 일반적인 순환 노드가 *메모리 셀(memory cell)*로 대체됩니다. 각 메모리 셀은 *내부 상태(internal state)*, 즉 가중치가 1로 고정된 자기 연결 순환 에지를 가진 노드를 포함하여, 기울기가 소실되거나 폭주하지 않고 많은 시간 단계에 걸쳐 전달될 수 있도록 보장합니다.

"장단기 메모리"라는 용어는 다음과 같은 직관에서 비롯됩니다. 단순한 순환 신경망은 가중치의 형태로 *장기 메모리(long-term memory)*를 가집니다. 가중치는 훈련 중에 천천히 변하며, 데이터에 대한 일반적인 지식을 인코딩합니다. 또한 일시적인 활성화의 형태로 *단기 메모리(short-term memory)*도 가지는데, 이는 각 노드에서 다음 노드로 전달됩니다. LSTM 모델은 메모리 셀을 통해 중간 형태의 저장 장치를 도입합니다. 메모리 셀은 복합 단위로서, 특정한 연결 패턴 안에서 더 단순한 노드들로부터 구축되며, 곱셈 노드들이 새롭게 포함되어 있습니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
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

## 게이트가 있는 메모리 셀 (Gated Memory Cell)

각 메모리 셀은 *내부 상태*와 다음과 같은 여러 곱셈 게이트들을 갖추고 있습니다. (i) 주어진 입력이 내부 상태에 영향을 미쳐야 하는지(*입력 게이트(input gate)*), (ii) 내부 상태가 $0$으로 비워져야 하는지(*망각 게이트(forget gate)*), (iii) 주어진 뉴런의 내부 상태가 셀의 출력에 영향을 미치도록 허용되어야 하는지(*출력 게이트(output gate)*)를 결정합니다.


### 게이트가 있는 은닉 상태 (Gated Hidden State)

기본 RNN과 LSTM의 주요 차이점은 후자가 은닉 상태의 게이팅을 지원한다는 점입니다. 이는 저희가 은닉 상태를 언제 *업데이트*해야 하는지, 그리고 언제 *리셋*해야 하는지에 대한 전용 메커니즘을 가지고 있다는 것을 의미합니다. 이러한 메커니즘은 학습되며, 위에 나열된 문제들을 해결합니다. 예를 들어, 첫 번째 토큰이 매우 중요하다면 저희는 첫 번째 관측 이후 은닉 상태를 업데이트하지 않도록 학습할 것입니다. 마찬가지로, 무관한 일시적 관측은 건너뛰도록 학습할 것입니다. 마지막으로, 필요할 때마다 잠재 상태를 리셋하도록 학습할 것입니다. 저희는 이를 아래에서 자세히 논의하겠습니다.

### 입력 게이트, 망각 게이트, 출력 게이트 (Input Gate, Forget Gate, and Output Gate)

LSTM 게이트로 들어가는 데이터는 :numref:`fig_lstm_0`에서 보여지는 바와 같이, 현재 시간 단계에서의 입력과 이전 시간 단계의 은닉 상태입니다. 시그모이드 활성화 함수를 가진 세 개의 완전 연결 레이어가 입력 게이트, 망각 게이트, 출력 게이트의 값을 계산합니다. 시그모이드 활성화의 결과로, 세 게이트의 모든 값은 $(0, 1)$ 범위 내에 있습니다. 또한, 저희는 *입력 노드(input node)*를 필요로 하며, 이는 일반적으로 *tanh* 활성화 함수로 계산됩니다. 직관적으로, *입력 게이트*는 입력 노드의 값 중 얼마만큼이 현재 메모리 셀 내부 상태에 더해져야 하는지를 결정합니다. *망각 게이트*는 메모리의 현재 값을 유지할지 비울지를 결정합니다. 그리고 *출력 게이트*는 메모리 셀이 현재 시간 단계에서 출력에 영향을 미쳐야 하는지를 결정합니다.


![LSTM 모델에서의 입력 게이트, 망각 게이트, 출력 게이트 계산.](../img/lstm-0.svg)
:label:`fig_lstm_0`

수학적으로, $h$개의 은닉 유닛이 있고, 배치 크기가 $n$이며, 입력의 수가 $d$라고 가정해 봅시다. 따라서 입력은 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$이고 이전 시간 단계의 은닉 상태는 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$입니다. 이에 대응하여, 시간 단계 $t$에서의 게이트들은 다음과 같이 정의됩니다. 입력 게이트는 $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, 망각 게이트는 $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, 출력 게이트는 $\mathbf{O}_t \in \mathbb{R}^{n \times h}$입니다. 이들은 다음과 같이 계산됩니다.

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$

여기서 $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$와 $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고 $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$는 편향 매개변수입니다. 합산 중에 브로드캐스팅(:numref:`subsec_broadcasting` 참조)이 발생함에 주의하십시오. 저희는 입력 값을 $(0, 1)$ 구간으로 매핑하기 위해 시그모이드 함수(:numref:`sec_mlp`에서 소개됨)를 사용합니다.


### 입력 노드 (Input Node)

다음으로 메모리 셀을 설계합니다. 다양한 게이트의 작용을 아직 명시하지 않았으므로, 먼저 *입력 노드* $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$를 소개합니다. 그 계산은 위에서 설명한 세 게이트와 유사하지만, $(-1, 1)$의 값 범위를 가진 $\tanh$ 함수를 활성화 함수로 사용합니다. 이는 시간 단계 $t$에서의 다음 방정식으로 이어집니다.

$$\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}),$$

여기서 $\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$와 $\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$는 가중치 매개변수이고 $\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$는 편향 매개변수입니다.

입력 노드에 대한 간단한 그림이 :numref:`fig_lstm_1`에 나타나 있습니다.

![LSTM 모델에서의 입력 노드 계산.](../img/lstm-1.svg)
:label:`fig_lstm_1`


### 메모리 셀 내부 상태 (Memory Cell Internal State)

LSTM에서, 입력 게이트 $\mathbf{I}_t$는 $\tilde{\mathbf{C}}_t$를 통해 새로운 데이터를 얼마만큼 고려할지를 제어하고, 망각 게이트 $\mathbf{F}_t$는 이전 셀 내부 상태 $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ 중 얼마만큼을 유지할지를 다룹니다. 아다마르(원소별) 곱셈 연산자 $\odot$를 사용하면, 다음과 같은 업데이트 방정식에 도달합니다.

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

만약 망각 게이트가 항상 1이고 입력 게이트가 항상 0이라면, 메모리 셀 내부 상태 $\mathbf{C}_{t-1}$는 영원히 변하지 않고 유지되어, 변하지 않은 채 각 이후 시간 단계로 전달될 것입니다. 그러나, 입력 게이트와 망각 게이트는 모델에 이 값을 변하지 않게 유지할지, 또는 이후 입력에 응답하여 그것을 변경할지를 학습할 수 있는 유연성을 부여합니다. 실제로, 이러한 설계는 기울기 소실 문제를 완화하여, 특히 긴 시퀀스 길이를 가진 데이터셋을 다룰 때 훈련하기에 훨씬 더 쉬운 모델을 만듭니다.

따라서 저희는 :numref:`fig_lstm_2`의 흐름도에 도달합니다.

![LSTM 모델에서의 메모리 셀 내부 상태 계산.](../img/lstm-2.svg)

:label:`fig_lstm_2`


### 은닉 상태 (Hidden State)

마지막으로, 메모리 셀의 출력, 즉 다른 레이어들에서 보이는 은닉 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$를 어떻게 계산할지를 정의해야 합니다. 여기서 출력 게이트가 역할을 합니다. LSTM에서는 먼저 메모리 셀 내부 상태에 $\tanh$를 적용하고, 그런 다음 이번에는 출력 게이트와 또 다른 점별 곱셈을 적용합니다. 이는 $\mathbf{H}_t$의 값이 항상 $(-1, 1)$ 구간 내에 있도록 보장합니다.

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$


출력 게이트가 1에 가까울 때마다, 메모리 셀 내부 상태가 이후 레이어들에 제약 없이 영향을 미치도록 허용되며, 반면 출력 게이트 값이 0에 가까울 때는 현재 메모리가 현재 시간 단계에서 신경망의 다른 레이어들에 영향을 미치는 것을 막습니다. 메모리 셀이 (출력 게이트가 0에 가까운 값을 가지는 한) 신경망의 나머지에 영향을 미치지 않고 많은 시간 단계에 걸쳐 정보를 축적할 수 있으며, 그런 다음 출력 게이트가 0에 가까운 값에서 1에 가까운 값으로 바뀌자마자 이후 시간 단계에서 신경망에 갑자기 영향을 미칠 수 있다는 점에 주목하십시오. :numref:`fig_lstm_3`은 데이터 흐름의 그래픽 표현을 보여줍니다.

![LSTM 모델에서의 은닉 상태 계산.](../img/lstm-3.svg)
:label:`fig_lstm_3`



## 처음부터 구현하기 (Implementation from Scratch)

이제 LSTM을 처음부터 구현해 보겠습니다. :numref:`sec_rnn-scratch`의 실험과 마찬가지로, 먼저 *The Time Machine* 데이터셋을 로드합니다.

### [**모델 매개변수 초기화**]

다음으로, 모델 매개변수를 정의하고 초기화해야 합니다. 이전과 같이, 하이퍼파라미터 `num_hiddens`는 은닉 유닛의 수를 결정합니다. 저희는 표준편차가 0.01인 가우시안 분포를 따라 가중치를 초기화하고, 편향은 0으로 설정합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple()  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple()  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple()  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple()  # Input node
```

```{.python .input}
%%tab jax
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple('i')  # Input gate
        self.W_xf, self.W_hf, self.b_f = triple('f')  # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple('o')  # Output gate
        self.W_xc, self.W_hc, self.b_c = triple('c')  # Input node
```

:begin_tab:`pytorch, mxnet, tensorflow`
[**실제 모델**]은 위에서 설명한 것처럼 정의되며, 세 개의 게이트와 입력 노드로 구성됩니다. 은닉 상태만이 출력 레이어로 전달된다는 점에 주목하십시오.
:end_tab:

:begin_tab:`jax`
[**실제 모델**]은 위에서 설명한 것처럼 정의되며, 세 개의 게이트와 입력 노드로 구성됩니다. 은닉 상태만이 출력 레이어로 전달된다는 점에 주목하십시오.
`forward` 메서드에서의 긴 for 루프는 첫 실행 시 매우 긴 JIT 컴파일 시간을 초래할 것입니다. 이에 대한 해결책으로, for 루프를 사용하여 매 시간 단계마다 상태를 업데이트하는 대신, JAX는 동일한 동작을 달성하기 위한 `jax.lax.scan` 유틸리티 변환을 가지고 있습니다. 이는 `carry`라고 불리는 초기 상태와 선행 축을 따라 스캔되는 `inputs` 배열을 입력으로 받습니다. `scan` 변환은 궁극적으로 최종 상태와 스택된 출력을 예상대로 반환합니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
            C = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    # Use lax.scan primitive instead of looping over the
    # inputs, since scan saves time in jit compilation.
    def scan_fn(carry, X):
        H, C = carry
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) + (
            d2l.matmul(H, self.W_hi)) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        return (H, C), H  # return carry, y

    if H_C is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens)), \
                jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H_C

    # scan takes the scan_fn, initial carry state, xs with leading axis to be scanned
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### [**훈련**] 및 예측

:numref:`sec_rnn-scratch`의 `RNNLMScratch` 클래스를 인스턴스화하여 LSTM 모델을 훈련해 봅시다.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## [**간결한 구현**]

고수준 API를 사용하면, LSTM 모델을 직접 인스턴스화할 수 있습니다. 이는 저희가 위에서 명시했던 모든 구성 세부 사항을 캡슐화합니다. 이 코드는 저희가 이전에 자세히 설명했던 많은 세부 사항에 대해 Python 대신 컴파일된 연산자를 사용하므로 상당히 더 빠릅니다.

```{.python .input}
%%tab mxnet
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.LSTM(num_hiddens)

    def forward(self, inputs, H_C=None):
        if H_C is None: H_C = self.rnn.begin_state(
            inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab pytorch
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab tensorflow
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.LSTM(
                num_hiddens, return_sequences=True,
                return_state=True, time_major=True)

    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(inputs, H_C)
        return outputs, H_C
```

```{.python .input}
%%tab jax
class LSTM(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H_C=None, training=False):
        if H_C is None:
            batch_size = inputs.shape[1]
            H_C = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0),
                                                        (batch_size,),
                                                        self.num_hiddens)

        LSTM = nn.scan(nn.OptimizedLSTMCell, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})

        H_C, outputs = LSTM()(H_C, inputs)
        return outputs, H_C
```

```{.python .input}
%%tab all
if tab.selected('pytorch'):
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'tensorflow', 'jax'):
    lstm = LSTM(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

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

LSTM은 자명하지 않은 상태 제어를 갖춘 잠재 변수 자기회귀 모델의 원형입니다. 다층, 잔차 연결, 다양한 종류의 정규화 등 여러 가지 변형이 수년에 걸쳐 제안되어 왔습니다. 그러나, LSTM과 다른 시퀀스 모델(예: GRU)을 훈련하는 것은 시퀀스의 장거리 의존성 때문에 상당히 비용이 많이 듭니다. 이후 저희는 일부 경우에 사용할 수 있는 Transformer와 같은 대안 모델을 만나게 될 것입니다.


## 요약

LSTM은 1997년에 발표되었지만, 2000년대 중반의 예측 대회에서의 몇 차례 우승과 함께 큰 명성을 얻었으며, 2011년부터 2017년에 시작된 Transformer 모델의 부상 전까지 시퀀스 학습에 대한 지배적인 모델이 되었습니다. Transformer조차도 그 핵심 아이디어 중 일부는 LSTM에 의해 도입된 아키텍처 설계 혁신에 빚지고 있습니다.


LSTM은 정보의 흐름을 제어하는 세 가지 유형의 게이트, 즉 입력 게이트, 망각 게이트, 출력 게이트를 가집니다. LSTM의 은닉층 출력에는 은닉 상태와 메모리 셀 내부 상태가 포함됩니다. 은닉 상태만이 출력 레이어로 전달되고, 메모리 셀 내부 상태는 완전히 내부에 남아 있습니다. LSTM은 기울기 소실과 기울기 폭주를 완화할 수 있습니다.



## 연습문제

1. 하이퍼파라미터를 조정하고 실행 시간, 퍼플렉시티(perplexity), 출력 시퀀스에 미치는 영향을 분석해 보십시오.
1. 단지 문자의 시퀀스가 아닌 적절한 단어를 생성하기 위해 모델을 어떻게 변경해야 합니까?
1. 주어진 은닉 차원에 대해 GRU, LSTM, 일반 RNN의 계산 비용을 비교해 보십시오. 훈련 및 추론 비용에 특히 주의를 기울이십시오.
1. 후보 메모리 셀이 $\tanh$ 함수를 사용하여 값 범위가 $-1$과 $1$ 사이임을 보장하는데, 왜 은닉 상태가 출력 값 범위가 $-1$과 $1$ 사이임을 보장하기 위해 $\tanh$ 함수를 다시 사용해야 합니까?
1. 문자 시퀀스 예측이 아닌 시계열 예측을 위한 LSTM 모델을 구현해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1057)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3861)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18016)
:end_tab:
