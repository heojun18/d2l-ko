# 순환 신경망 (Recurrent Neural Networks)
:label:`sec_rnn`


:numref:`sec_language-model`에서 저희는 언어 모델링을 위한 마르코프 모델과 $n$-그램을 설명했는데, 여기서 타임스텝 $t$의 토큰 $x_t$의 조건부 확률은 오직 이전 $n-1$개의 토큰에만 의존합니다.
타임스텝 $t-(n-1)$보다 더 이른 토큰이 $x_t$에 미치는 잠재적인 영향을 포함하고자 한다면,
저희는 $n$을 늘려야 합니다.
그러나 모델 파라미터의 수도 그에 따라 지수적으로 증가할 것입니다. 왜냐하면 어휘 집합 $\mathcal{V}$에 대해 $|\mathcal{V}|^n$개의 숫자를 저장해야 하기 때문입니다.
따라서, $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$을 모델링하기보다는 잠재 변수 모델을 사용하는 것이 선호됩니다.

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

여기서 $h_{t-1}$은 타임스텝 $t-1$까지의 시퀀스 정보를 저장하는 *은닉 상태(hidden state)* 입니다.
일반적으로,
어떤 타임스텝 $t$에서의 은닉 상태는 현재 입력 $x_{t}$와 이전 은닉 상태 $h_{t-1}$ 모두를 바탕으로 계산될 수 있습니다.

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

:eqref:`eq_ht_xt`에서 충분히 강력한 함수 $f$에 대해, 잠재 변수 모델은 근사가 아닙니다. 결국, $h_t$는 그것이 지금까지 관측한 모든 데이터를 단순히 저장할 수도 있습니다.
그러나, 이는 계산과 저장 모두를 잠재적으로 비싸게 만들 수 있습니다.

:numref:`chap_perceptrons`에서 저희가 은닉 유닛을 가진 은닉 층을 논의했다는 점을 떠올려 보세요.
은닉 층과 은닉 상태가 매우 다른 두 개념을 가리킨다는 점은
주목할 만합니다.
은닉 층은, 설명한 바와 같이, 입력에서 출력으로의 경로에서 시야에 가려진 층입니다.
은닉 상태는 엄밀히 말해 주어진 스텝에서 저희가 하는 어떤 작업에든 *입력*이며,
오직 이전 타임스텝의 데이터를 봄으로써만 계산될 수 있습니다.

*순환 신경망(Recurrent neural networks, RNN)* 은 은닉 상태를 가진 신경망입니다. RNN 모델을 소개하기 전에, 저희는 먼저 :numref:`sec_mlp`에서 소개된 MLP 모델을 다시 살펴봅니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

## 은닉 상태가 없는 신경망

은닉 층이 하나인 MLP를 살펴봅시다.
은닉 층의 활성화 함수를 $\phi$라고 합시다.
배치 크기 $n$과 $d$개의 입력을 가진 예시의 미니배치 $\mathbf{X} \in \mathbb{R}^{n \times d}$가 주어졌을 때, 은닉 층 출력 $\mathbf{H} \in \mathbb{R}^{n \times h}$는 다음과 같이 계산됩니다.

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_without_state`

:eqref:`rnn_h_without_state`에서, 저희는 은닉 층에 대한 가중치 파라미터 $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$, 편향 파라미터 $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$, 그리고 은닉 유닛의 수 $h$를 가지고 있습니다.
이렇게 갖추고, 저희는 합산 동안 브로드캐스팅을 적용합니다 (:numref:`subsec_broadcasting` 참고).
다음으로, 은닉 층 출력 $\mathbf{H}$는 출력 층의 입력으로 사용되며, 이는 다음과 같이 주어집니다.

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

여기서 $\mathbf{O} \in \mathbb{R}^{n \times q}$는 출력 변수, $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$는 가중치 파라미터, $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$는 출력 층의 편향 파라미터입니다. 만약 분류 문제라면, 저희는 $\mathrm{softmax}(\mathbf{O})$를 사용하여 출력 범주의 확률 분포를 계산할 수 있습니다.

이는 :numref:`sec_sequence`에서 저희가 이전에 풀었던 회귀 문제와 완전히 유사하므로, 자세한 내용은 생략합니다.
저희가 무작위로 특징-레이블 쌍을 고르고 자동 미분과 확률적 경사 하강법을 통해 신경망의 파라미터를 학습할 수 있다고 말하는 것으로 충분합니다.

## 은닉 상태를 가진 순환 신경망
:label:`subsec_rnn_w_hidden_states`

은닉 상태를 가지면 상황이 완전히 달라집니다. 구조를 좀 더 자세히 살펴봅시다.

타임스텝 $t$에 입력의 미니배치
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$가
있다고 가정합시다.
다시 말해,
$n$개의 시퀀스 예시의 미니배치에 대해,
$\mathbf{X}_t$의 각 행은 시퀀스의 타임스텝 $t$에서의 한 예시에 해당합니다.
다음으로,
타임스텝 $t$의 은닉 층 출력을 $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$로 표기합시다.
MLP와는 달리, 여기서 저희는 이전 타임스텝의 은닉 층 출력 $\mathbf{H}_{t-1}$을 저장하고 이전 타임스텝의 은닉 층 출력을 현재 타임스텝에서 어떻게 사용할지 기술하기 위해 새로운 가중치 파라미터 $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$를 도입합니다. 구체적으로, 현재 타임스텝의 은닉 층 출력의 계산은 현재 타임스텝의 입력과 함께 이전 타임스텝의 은닉 층 출력에 의해 결정됩니다.

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_with_state`

:eqref:`rnn_h_without_state`와 비교하면, :eqref:`rnn_h_with_state`는 항 $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$를 하나 더 추가하고 따라서
:eqref:`eq_ht_xt`를 인스턴스화합니다.
인접한 타임스텝의 은닉 층 출력 $\mathbf{H}_t$와 $\mathbf{H}_{t-1}$ 사이의 관계로부터,
저희는 이러한 변수가 마치 신경망의 현재 타임스텝의 상태 또는 메모리처럼, 현재 타임스텝까지의 시퀀스의 과거 정보를 포착하고 유지했다는 것을 알 수 있습니다. 따라서, 이러한 은닉 층 출력을 *은닉 상태(hidden state)* 라고 합니다.
은닉 상태가 현재 타임스텝에서 이전 타임스텝의 동일한 정의를 사용하기 때문에, :eqref:`rnn_h_with_state`의 계산은 *순환적(recurrent)* 입니다. 따라서, 저희가 말했듯이,
순환 계산을 기반으로 한 은닉 상태를 가진 신경망을
*순환 신경망(recurrent neural networks)* 이라고 부릅니다.
RNN에서
:eqref:`rnn_h_with_state`의 계산을 수행하는 층을
*순환 층(recurrent layers)* 이라고 합니다.


RNN을 구성하는 데에는 많은 다른 방법이 있습니다.
:eqref:`rnn_h_with_state`로 정의된 은닉 상태를 가진 것들이 매우 흔합니다.
타임스텝 $t$에 대해,
출력 층의 출력은 MLP에서의 계산과 비슷합니다.

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

RNN의 파라미터에는
은닉 층의 가중치 $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$,
편향 $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$와 함께
출력 층의 가중치 $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$,
편향 $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$가 포함됩니다.
다른 타임스텝에서도,
RNN은 항상 이러한 모델 파라미터를 사용한다는 점을
언급할 가치가 있습니다.
따라서, RNN의 파라미터화 비용은
타임스텝의 수가 증가함에 따라 커지지 않습니다.

:numref:`fig_rnn`은 인접한 세 타임스텝에서 RNN의 계산 논리를 보여줍니다.
어떤 타임스텝 $t$에서든,
은닉 상태의 계산은 다음과 같이 다룰 수 있습니다.
(i) 현재 타임스텝 $t$의 입력 $\mathbf{X}_t$와 이전 타임스텝 $t-1$의 은닉 상태 $\mathbf{H}_{t-1}$을 연결한 다음,
(ii) 그 연결 결과를 활성화 함수 $\phi$를 가진 완전 연결 층에 공급합니다.
그러한 완전 연결 층의 출력이 현재 타임스텝 $t$의 은닉 상태 $\mathbf{H}_t$입니다.
이 경우에,
모델 파라미터는 모두 :eqref:`rnn_h_with_state`에서 온, $\mathbf{W}_{\textrm{xh}}$와 $\mathbf{W}_{\textrm{hh}}$의 연결, 그리고 편향 $\mathbf{b}_\textrm{h}$입니다.
현재 타임스텝 $t$의 은닉 상태 $\mathbf{H}_t$는 다음 타임스텝 $t+1$의 은닉 상태 $\mathbf{H}_{t+1}$을 계산하는 데 참여할 것입니다.
더욱이, $\mathbf{H}_t$는 현재 타임스텝 $t$의 출력
$\mathbf{O}_t$를 계산하기 위해
완전 연결 출력 층에도
공급될 것입니다.

![은닉 상태를 가진 RNN.](../img/rnn.svg)
:label:`fig_rnn`

저희는 은닉 상태에 대한 $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$의 계산이
$\mathbf{X}_t$와 $\mathbf{H}_{t-1}$의 연결과
$\mathbf{W}_{\textrm{xh}}$와 $\mathbf{W}_{\textrm{hh}}$의 연결의
행렬 곱셈과 동등하다고
방금 언급했습니다.
이것은 수학적으로 증명될 수 있지만,
다음에서는 시연으로 간단한 코드 스니펫만 사용합니다.
우선,
저희는 각각 모양이 (3, 1), (1, 4), (3, 4), (4, 4)인 행렬 `X`, `W_xh`, `H`, `W_hh`를 정의합니다.
`X`에 `W_xh`를, `H`에 `W_hh`를 곱하고, 그 다음 이 두 곱을 더하면,
모양 (3, 4)인 행렬을 얻습니다.

```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab jax
X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
                                                        d2l.get_key(), (1, 4))
H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
                                                        d2l.get_key(), (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

이제 저희는 행렬 `X`와 `H`를
열(축 1)을 따라 연결하고,
행렬 `W_xh`와 `W_hh`를 행(축 0)을 따라 연결합니다.
이 두 연결은
각각 모양 (3, 5)와 모양 (5, 4)인
행렬을 만듭니다.
이 두 연결된 행렬을 곱하면,
저희는 위와 동일한 모양 (3, 4)의
출력 행렬을 얻습니다.

```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## RNN 기반 문자 수준 언어 모델

:numref:`sec_language-model`에서의 언어 모델링에 대해,
저희가 현재와 과거의 토큰을 바탕으로
다음 토큰을 예측하는 것을 목표로 하며,
따라서 원래 시퀀스를 한 토큰씩 이동시켜
타깃(레이블)으로 삼는다는 점을 떠올려 보세요.
:citet:`Bengio.Ducharme.Vincent.ea.2003`이 언어 모델링에 신경망을 사용하는 것을
처음으로 제안했습니다.
다음에서 저희는 RNN이 어떻게 언어 모델을 구축하는 데 사용될 수 있는지를 보여줍니다.
미니배치 크기를 1로 하고, 텍스트의 시퀀스를 "machine"이라고 합시다.
후속 절에서 학습을 단순화하기 위해,
저희는 텍스트를 단어가 아니라 문자로 토큰화하고
*문자 수준 언어 모델(character-level language model)* 을 고려합니다.
:numref:`fig_rnn_train`은 문자 수준 언어 모델링을 위해 RNN을 통해 현재와 이전 문자를 바탕으로 다음 문자를 예측하는 방법을 보여줍니다.

![RNN 기반의 문자 수준 언어 모델. 입력과 타깃 시퀀스는 각각 "machin"과 "achine"입니다.](../img/rnn-train.svg)
:label:`fig_rnn_train`

학습 과정 동안,
저희는 각 타임스텝에 대해 출력 층의 출력에 소프트맥스 연산을 실행한 다음, 모델 출력과 타깃 사이의 오차를 계산하기 위해 교차 엔트로피 손실을 사용합니다.
은닉 층에서의 은닉 상태의 순환 계산 때문에, :numref:`fig_rnn_train`에서 타임스텝 3의 출력 $\mathbf{O}_3$은 텍스트 시퀀스 "m", "a", "c"에 의해 결정됩니다. 학습 데이터에서 시퀀스의 다음 문자가 "h"이므로, 타임스텝 3의 손실은 특징 시퀀스 "m", "a", "c"와 이 타임스텝의 타깃 "h"를 바탕으로 생성된 다음 문자의 확률 분포에 의존할 것입니다.

실제로는, 각 토큰이 $d$차원 벡터로 표현되고, 저희는 배치 크기 $n>1$을 사용합니다. 따라서, 타임스텝 $t$의 입력 $\mathbf X_t$는 $n\times d$ 행렬일 것이며, 이는 :numref:`subsec_rnn_w_hidden_states`에서 저희가 논의한 것과 동일합니다.

다음 절들에서, 저희는 문자 수준 언어 모델을 위한 RNN을
구현할 것입니다.


## 요약

은닉 상태에 순환 계산을 사용하는 신경망을 순환 신경망(RNN)이라고 합니다.
RNN의 은닉 상태는 현재 타임스텝까지의 시퀀스의 과거 정보를 포착할 수 있습니다. 순환 계산으로, RNN 모델 파라미터의 수는 타임스텝의 수가 증가함에 따라 커지지 않습니다. 응용으로는, RNN은 문자 수준 언어 모델을 만드는 데 사용될 수 있습니다.


## 연습문제

1. 텍스트 시퀀스에서 다음 문자를 예측하기 위해 RNN을 사용한다면, 어떤 출력에 대해서든 필요한 차원은 무엇인가요?
1. RNN이 텍스트 시퀀스의 이전 모든 토큰을 바탕으로 어떤 타임스텝에서의 토큰의 조건부 확률을 왜 표현할 수 있나요?
1. 긴 시퀀스를 통해 역전파하면 그래디언트에 무슨 일이 일어나나요?
1. 이 절에서 설명한 언어 모델과 관련된 몇 가지 문제는 무엇인가요?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/180013)
:end_tab:
