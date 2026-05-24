# 양방향 순환 신경망 (Bidirectional Recurrent Neural Networks)
:label:`sec_bi_rnn`

지금까지 시퀀스 학습 과제의 실용적인 예시로 저희가 다룬 것은 언어 모델링이었으며, 여기서는 시퀀스 내 이전 토큰들이 모두 주어진 경우 다음 토큰을 예측하는 것을 목표로 합니다. 이 시나리오에서, 저희는 왼쪽 문맥에만 조건을 부여하기를 원하며, 따라서 표준 RNN의 단방향 체이닝이 적절해 보입니다. 그러나 매 시간 단계에서의 예측에 왼쪽과 오른쪽 문맥 양쪽 모두를 조건으로 부여해도 전혀 문제가 없는, 다른 많은 시퀀스 학습 과제 맥락들이 있습니다. 예를 들어, 품사 탐지를 고려해 보십시오. 주어진 단어와 연관된 품사를 평가할 때 양방향의 문맥을 고려하지 않을 이유가 무엇이 있겠습니까?

또 다른 흔한 과제(관심 있는 실제 과제에 모델을 미세 조정하기 전의 사전 학습 연습으로 종종 유용한)는 텍스트 문서에서 무작위 토큰을 마스킹하고 누락된 토큰의 값을 예측하도록 시퀀스 모델을 훈련하는 것입니다. 빈칸 뒤에 무엇이 오는지에 따라, 누락된 토큰의 가능한 값이 극적으로 변한다는 점에 유의하십시오.

* I am `___`.
* I am `___` hungry.
* I am `___` hungry, and I can eat half a pig.

첫 번째 문장에서는 "happy"가 가능성 있는 후보로 보입니다. 두 번째 문장에서는 "not"과 "very"라는 단어가 그럴듯해 보이지만, "not"은 세 번째 문장과 양립할 수 없어 보입니다.


다행스럽게도, 단순한 기법 하나가 어떤 단방향 RNN이든 양방향 RNN으로 변환합니다 :cite:`Schuster.Paliwal.1997`. 저희는 단순히 반대 방향으로 체인화되어 동일한 입력에 작용하는 두 개의 단방향 RNN 레이어를 구현합니다(:numref:`fig_birnn`). 첫 번째 RNN 레이어의 경우, 첫 번째 입력은 $\mathbf{x}_1$이고 마지막 입력은 $\mathbf{x}_T$이지만, 두 번째 RNN 레이어의 경우 첫 번째 입력은 $\mathbf{x}_T$이고 마지막 입력은 $\mathbf{x}_1$입니다. 이 양방향 RNN 레이어의 출력을 생성하기 위해, 저희는 단순히 두 개의 기저 단방향 RNN 레이어의 해당하는 출력을 연결(concatenate)합니다.


![양방향 RNN의 아키텍처.](../img/birnn.svg)
:label:`fig_birnn`


형식적으로 어떤 시간 단계 $t$에 대해서든, 저희는 미니배치 입력 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (예제의 수 $=n$; 각 예제의 입력 수 $=d$)를 고려하며, 은닉층 활성화 함수를 $\phi$로 둡니다. 양방향 아키텍처에서, 이 시간 단계에 대한 순방향 및 역방향 은닉 상태는 각각 $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$와 $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$이며, 여기서 $h$는 은닉 유닛의 수입니다. 순방향 및 역방향 은닉 상태 업데이트는 다음과 같습니다.


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}
$$

여기서 가중치 $\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ and } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$, 그리고 편향 $\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$와 $\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$는 모두 모델 매개변수입니다.

다음으로, 저희는 순방향 및 역방향 은닉 상태 $\overrightarrow{\mathbf{H}}_t$와 $\overleftarrow{\mathbf{H}}_t$를 연결하여 출력 레이어에 공급할 은닉 상태 $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$를 얻습니다. 다중 은닉층을 가진 심층 양방향 RNN에서, 이러한 정보는 다음 양방향 레이어로 *입력*으로 전달됩니다. 마지막으로, 출력 레이어는 출력 $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (출력의 수 $=q$)를 계산합니다.

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

여기서, 가중치 행렬 $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$와 편향 $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$는 출력 레이어의 모델 매개변수입니다. 기술적으로는 두 방향이 다른 수의 은닉 유닛을 가질 수 있지만, 이러한 설계 선택은 실제로 거의 이루어지지 않습니다. 이제 양방향 RNN의 간단한 구현을 시연하겠습니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import npx, np
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
from jax import numpy as jnp
```

## 처음부터 구현하기 (Implementation from Scratch)

양방향 RNN을 처음부터 구현하고 싶다면, 별도의 학습 가능한 매개변수를 가진 두 개의 단방향 `RNNScratch` 인스턴스를 포함시킬 수 있습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

```{.python .input}
%%tab jax
class BiRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

순방향 및 역방향 RNN의 상태는 별도로 업데이트되는 반면, 이 두 RNN의 출력은 연결됩니다.

```{.python .input}
%%tab all
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)
```

## 간결한 구현 (Concise Implementation)

:begin_tab:`pytorch, mxnet, tensorflow`
고수준 API를 사용하면, 양방향 RNN을 더 간결하게 구현할 수 있습니다. 여기서는 GRU 모델을 예시로 들겠습니다.
:end_tab:

:begin_tab:`jax`
Flax API는 RNN 레이어를 제공하지 않으므로 `bidirectional` 인수의 개념도 없습니다. 양방향 레이어가 필요한 경우, 처음부터 구현한 것에서 보여진 것처럼 수동으로 입력을 뒤집어야 합니다.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

## 요약

양방향 RNN에서, 각 시간 단계에 대한 은닉 상태는 현재 시간 단계의 이전과 이후의 데이터에 의해 동시에 결정됩니다. 양방향 RNN은 주로 시퀀스 인코딩과 양방향 문맥이 주어진 관측치 추정에 유용합니다. 양방향 RNN은 긴 기울기 사슬 때문에 훈련 비용이 매우 많이 듭니다.

## 연습문제

1. 다른 방향이 다른 수의 은닉 유닛을 사용한다면, $\mathbf{H}_t$의 형태는 어떻게 변하겠습니까?
1. 다중 은닉층을 가진 양방향 RNN을 설계해 보십시오.
1. 다의어는 자연어에서 흔합니다. 예를 들어, "bank"라는 단어는 "i went to the bank to deposit cash"와 "i went to the bank to sit down"이라는 문맥에서 서로 다른 의미를 가집니다. 문맥 시퀀스와 단어가 주어졌을 때 올바른 문맥에서의 단어 벡터 표현이 반환되도록 신경망 모델을 어떻게 설계할 수 있겠습니까? 다의어를 다루기 위해 어떤 종류의 신경망 아키텍처가 선호됩니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1059)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18019)
:end_tab:
