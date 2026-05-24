# BPTT(Backpropagation Through Time)
:label:`sec_bptt`

만약 :numref:`sec_rnn-scratch`의 연습문제를 완료해 보셨다면,
가끔씩 발생하는 거대한 그래디언트가 훈련을 불안정하게 만드는 것을 막기 위해
그래디언트 클리핑이 매우 중요하다는 점을 보셨을 것입니다.
저희는 폭주하는 그래디언트가 긴 시퀀스를 가로질러 역전파하는 데서
비롯된다는 점을 암시했었습니다.
일련의 현대적인 RNN 아키텍처를 소개하기에 앞서,
시퀀스 모델에서 *역전파*가 어떻게 동작하는지
수학적으로 자세히 들여다보겠습니다.
이 논의가 *소실*하는 그래디언트와 *폭주*하는 그래디언트의 개념에
어느 정도 정밀성을 부여해 주기를 바랍니다.
:numref:`sec_backprop`에서 MLP를 소개할 때
계산 그래프를 통한 순방향 전파와 역방향 전파에 대한
저희의 논의를 떠올리신다면,
RNN의 순방향 전파는 비교적 직관적일 것입니다.
RNN에 역전파를 적용하는 것을
*backpropagation through time*(BPTT)라고 부릅니다 :cite:`Werbos.1990`.
이 절차는 저희가 RNN의 계산 그래프를 한 번에 한 타임스텝씩
펼치는(또는 풀어내는) 것을 요구합니다.
펼쳐진 RNN은 본질적으로
동일한 파라미터가 펼쳐진 네트워크 전반에 걸쳐 반복되며
각 타임스텝마다 등장한다는 특수한 성질을 가진
피드포워드 신경망입니다.
그런 다음, 임의의 피드포워드 신경망에서와 마찬가지로,
저희는 연쇄 법칙(chain rule)을 적용하여
펼쳐진 네트워크를 통해 그래디언트를 역전파할 수 있습니다.
각 파라미터에 대한 그래디언트는
펼쳐진 네트워크에서 그 파라미터가 등장하는 모든 위치에 걸쳐
합산되어야 합니다.
이러한 가중치 묶기(weight tying)를 처리하는 방식은
합성곱 신경망에 대한 저희의 챕터에서 익숙할 것입니다.


시퀀스가 꽤 길 수 있다는 점 때문에
복잡한 문제가 발생합니다.
1000개가 넘는 토큰으로 이루어진 텍스트 시퀀스를 다루는 것은
드문 일이 아닙니다.
이것이 계산적(메모리가 너무 많이 듦) 측면과
최적화(수치적 불안정성) 측면 모두에서 문제를 일으킨다는 점에
유의하세요.
첫 단계로부터의 입력은 출력에 도달하기 전에
1000번이 넘는 행렬 곱을 거치고,
그래디언트를 계산하는 데에도
또 다른 1000번의 행렬 곱이 필요합니다.
이제 저희는 무엇이 잘못될 수 있는지, 그리고
실제로 이를 어떻게 해결할 수 있는지를 분석합니다.


## RNN에서의 그래디언트 분석
:label:`subsec_bptt_analysis`

저희는 RNN이 어떻게 동작하는지에 대한 단순화된 모델로부터 시작합니다.
이 모델은 은닉 상태의 구체적인 사항과
그것이 어떻게 업데이트되는지에 대한 세부 내용을 무시합니다.
여기서의 수학적 표기법은
스칼라, 벡터, 행렬을 명시적으로 구분하지 않습니다.
저희는 단지 어느 정도의 직관을 형성하려고 할 뿐입니다.
이 단순화된 모델에서, 저희는 타임스텝 $t$에서
$h_t$를 은닉 상태,
$x_t$를 입력, $o_t$를 출력으로 나타냅니다.
:numref:`subsec_rnn_w_hidden_states`에서 저희가 논의했듯이,
입력과 은닉 상태는 은닉 층에서 하나의 가중치 변수와 곱해지기 전에
연결될 수 있음을 떠올려 봅시다.
따라서 저희는 은닉 층과 출력 층의 가중치를 나타내기 위해
각각 $w_\textrm{h}$와 $w_\textrm{o}$를 사용합니다.
그 결과, 각 타임스텝에서의 은닉 상태와 출력은 다음과 같습니다.

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

여기서 $f$와 $g$는 각각 은닉 층과 출력 층의 변환입니다.
따라서 저희는 순환 계산을 통해 서로 의존하는
값들의 사슬 $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$을 가집니다.
순방향 전파는 상당히 직관적입니다.
저희가 해야 할 일은 한 번에 한 타임스텝씩 $(x_t, h_t, o_t)$ 삼중항을 반복 순회하는 것뿐입니다.
그러면 출력 $o_t$와 원하는 목표값 $y_t$ 사이의 불일치는
모든 $T$ 타임스텝에 걸쳐 목적 함수에 의해 다음과 같이 평가됩니다.

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



역전파의 경우, 특히 목적 함수 $L$의 파라미터 $w_\textrm{h}$에 대해
그래디언트를 계산할 때는 문제가 좀 더 까다롭습니다.
구체적으로 말하면, 연쇄 법칙에 의해

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh`의 곱셈에서
첫 번째 인자와 두 번째 인자는
계산하기 쉽습니다.
세 번째 인자 $\partial h_t/\partial w_\textrm{h}$가 까다로워지는 부분인데,
저희는 파라미터 $w_\textrm{h}$가 $h_t$에 미치는 영향을 순환적으로 계산해야 하기 때문입니다.
:eqref:`eq_bptt_ht_ot`의 순환 계산에 따르면,
$h_t$는 $h_{t-1}$과 $w_\textrm{h}$ 모두에 의존하며,
여기서 $h_{t-1}$의 계산 또한
$w_\textrm{h}$에 의존합니다.
따라서 연쇄 법칙을 사용하여
$h_t$의 $w_\textrm{h}$에 대한 전미분(total derivative)을 평가하면 다음을 얻습니다.

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


위의 그래디언트를 유도하기 위해, 저희가
$a_{0}=0$과 $t=1, 2,\ldots$에 대해 $a_{t}=b_{t}+c_{t}a_{t-1}$을 만족하는
세 수열 $\{a_{t}\},\{b_{t}\},\{c_{t}\}$을 가지고 있다고 가정합시다.
그러면 $t\geq 1$에 대해, 다음을 쉽게 보일 수 있습니다.

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

$a_t$, $b_t$, $c_t$를 다음과 같이 치환하면

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},\end{aligned}$$

:eqref:`eq_bptt_partial_ht_wh_recur`의 그래디언트 계산은
$a_{t}=b_{t}+c_{t}a_{t-1}$을 만족합니다.
따라서 :eqref:`eq_bptt_at`에 따라,
저희는 :eqref:`eq_bptt_partial_ht_wh_recur`의 순환 계산을 다음과 같이
제거할 수 있습니다.

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

저희가 $\partial h_t/\partial w_\textrm{h}$를 재귀적으로 계산하기 위해 연쇄 법칙을 사용할 수는 있지만,
이 사슬은 $t$가 클 때마다 매우 길어질 수 있습니다.
이 문제를 다루기 위한 여러 전략에 대해 논의해 봅시다.

### 전체 계산 ### 

한 가지 아이디어는 :eqref:`eq_bptt_partial_ht_wh_gen`의 전체 합을 계산하는 것입니다.
그러나 이것은 매우 느리고 그래디언트가 폭주할 수 있는데,
초기 조건의 미묘한 변화가
결과에 잠재적으로 크게 영향을 미칠 수 있기 때문입니다.
즉, 저희는 초기 조건의 아주 작은 변화가
결과에 불균형적인 변화를 가져오는 나비효과와 유사한 현상을
볼 수 있습니다.
이것은 일반적으로 바람직하지 않습니다.
결국 저희가 찾고 있는 것은 잘 일반화되는 견고한 추정기이기 때문입니다.
따라서 이 전략은 실제로는 거의 사용되지 않습니다.

### 타임스텝 절단하기 ###

대안으로,
저희는 :eqref:`eq_bptt_partial_ht_wh_gen`의 합을
$\tau$ 단계 이후에 절단할 수 있습니다.
이것이 저희가 지금까지 논의해 온 내용입니다.
이는 단순히 합을 $\partial h_{t-\tau}/\partial w_\textrm{h}$에서 끝냄으로써
실제 그래디언트의 *근사*로 이어집니다.
실제로는 이것이 꽤 잘 동작합니다.
이는 일반적으로 truncated backpropagation through time이라고 불리는 것입니다 :cite:`Jaeger.2002`.
이로 인한 결과 중 하나는 모델이
장기적 결과보다는 단기적 영향에
주로 집중한다는 것입니다.
이는 추정을 더 단순하고 더 안정적인 모델 쪽으로 편향시키므로
사실 *바람직합니다*.


### 무작위 절단 ### 

마지막으로, 저희는 $\partial h_t/\partial w_\textrm{h}$를
기댓값으로는 올바르지만 시퀀스를 절단하는 확률 변수로
대체할 수 있습니다.
이는 미리 정의된 $0 \leq \pi_t \leq 1$에 대해
$P(\xi_t = 0) = 1-\pi_t$이고
$P(\xi_t = \pi_t^{-1}) = \pi_t$이므로 $E[\xi_t] = 1$인
$\xi_t$의 수열을 사용함으로써 달성됩니다.
저희는 이를 :eqref:`eq_bptt_partial_ht_wh_recur`의 그래디언트
$\partial h_t/\partial w_\textrm{h}$를 다음과 같이 대체하는 데
사용합니다.

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$


$\xi_t$의 정의에 따라
$E[z_t] = \partial h_t/\partial w_\textrm{h}$임을 알 수 있습니다.
$\xi_t = 0$일 때마다 순환 계산은
해당 타임스텝 $t$에서 종료됩니다.
이는 길이가 다양한 시퀀스들의 가중합으로 이어지는데,
긴 시퀀스는 드물지만 적절하게 가중치가 부여됩니다.
이 아이디어는
:citet:`Tallec.Ollivier.2017`에 의해 제안되었습니다.

### 전략 비교

![RNN에서 그래디언트를 계산하기 위한 전략 비교. 위에서 아래로: 무작위 절단, 정규 절단, 전체 계산.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`


:numref:`fig_truncated_bptt`는 RNN에 대한 BPTT를 사용해
*The Time Machine*의 처음 몇 글자를 분석할 때의
세 가지 전략을 보여 줍니다.

* 첫 번째 행은 텍스트를 가변 길이의 세그먼트로 분할하는 무작위 절단입니다.
* 두 번째 행은 텍스트를 동일한 길이의 부분 시퀀스로 나누는 정규 절단입니다. 이것이 저희가 RNN 실험에서 해 온 작업입니다.
* 세 번째 행은 계산적으로 다루기 어려운 식으로 이어지는 전체 BPTT입니다.


불행히도, 이론적으로는 매력적이지만,
무작위 절단은 정규 절단보다 그다지 잘 동작하지 않는데,
이는 여러 요인 때문일 가능성이 가장 큽니다.
첫째, 과거로 여러 단계의 역전파를 거친 뒤
관측의 효과는 실제로 의존성을 포착하기에
이미 충분합니다.
둘째, 증가된 분산은 더 많은 단계를 거칠수록 그래디언트가
더 정확해진다는 사실을 상쇄합니다.
셋째, 저희는 사실 짧은 범위의 상호작용만을 가지는
모델을 *원합니다*.
따라서 정규적으로 절단된 BPTT는
바람직할 수 있는 약간의 정규화 효과를 가집니다.

## BPTT 상세히 보기

일반적인 원리를 논의했으니,
이제 BPTT를 자세히 논의해 봅시다.
:numref:`subsec_bptt_analysis`의 분석과 달리,
다음에서는 분해된 모델 파라미터 모두에 대해
목적 함수의 그래디언트를 어떻게 계산하는지
보여 드리겠습니다.
간단하게 유지하기 위해, 저희는 편향 파라미터가 없는
RNN을 고려하며, 그 은닉 층의 활성화 함수는
항등 사상($\phi(x)=x$)을 사용합니다.
타임스텝 $t$에 대해, 단일 예제 입력과
목표값을 각각 $\mathbf{x}_t \in \mathbb{R}^d$와 $y_t$라고 합시다.
은닉 상태 $\mathbf{h}_t \in \mathbb{R}^h$와
출력 $\mathbf{o}_t \in \mathbb{R}^q$은
다음과 같이 계산됩니다.

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$

여기서 $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$,
$\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$은
가중치 파라미터입니다.
타임스텝 $t$에서의 손실을 $l(\mathbf{o}_t, y_t)$로
표기합니다.
따라서 저희의 목적 함수, 즉 시퀀스의 시작부터
$T$ 타임스텝에 걸친 손실은 다음과 같습니다.

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


RNN의 계산 동안 모델 변수와 파라미터 사이의
의존성을 시각화하기 위해,
저희는 :numref:`fig_rnn_bptt`에서 보이는 것처럼
모델에 대한 계산 그래프를 그릴 수 있습니다.
예를 들어, 타임스텝 3의 은닉 상태
$\mathbf{h}_3$의 계산은 모델 파라미터
$\mathbf{W}_\textrm{hx}$와 $\mathbf{W}_\textrm{hh}$,
이전 타임스텝의 은닉 상태 $\mathbf{h}_2$,
그리고 현재 타임스텝의 입력 $\mathbf{x}_3$에 의존합니다.

![세 타임스텝을 가지는 RNN 모델의 의존성을 보여 주는 계산 그래프. 상자는 변수(음영 없음) 또는 파라미터(음영 있음)를 나타내고 원은 연산자를 나타냅니다.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

방금 언급했듯이, :numref:`fig_rnn_bptt`의 모델 파라미터는
$\mathbf{W}_\textrm{hx}$, $\mathbf{W}_\textrm{hh}$, $\mathbf{W}_\textrm{qh}$입니다.
일반적으로 이 모델을 훈련하려면 이러한 파라미터에 대한
$\partial L/\partial \mathbf{W}_\textrm{hx}$, $\partial L/\partial \mathbf{W}_\textrm{hh}$, $\partial L/\partial \mathbf{W}_\textrm{qh}$의
그래디언트 계산이 필요합니다.
:numref:`fig_rnn_bptt`의 의존성에 따라,
저희는 화살표의 반대 방향으로 순회하면서
그래디언트를 차례로 계산하고 저장할 수 있습니다.
연쇄 법칙에서 다양한 모양의 행렬, 벡터, 스칼라의 곱셈을
유연하게 표현하기 위해,
저희는 :numref:`sec_backprop`에 설명된 대로
$\textrm{prod}$ 연산자를 계속해서 사용합니다.


먼저, 임의의 타임스텝 $t$에서 모델 출력에 대해
목적 함수를 미분하는 것은
상당히 직관적입니다.

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

이제 출력 층의 파라미터 $\mathbf{W}_\textrm{qh}$에 대한
목적 함수의 그래디언트를 계산할 수 있습니다.
$\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$.
:numref:`fig_rnn_bptt`에 따르면,
목적 함수 $L$은 $\mathbf{o}_1, \ldots, \mathbf{o}_T$를 통해
$\mathbf{W}_\textrm{qh}$에 의존합니다.
연쇄 법칙을 사용하면 다음을 얻습니다.

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

여기서 $\partial L/\partial \mathbf{o}_t$는
:eqref:`eq_bptt_partial_L_ot`에 의해 주어집니다.

다음으로, :numref:`fig_rnn_bptt`에서 보이는 것처럼,
마지막 타임스텝 $T$에서
목적 함수 $L$은 $\mathbf{o}_T$를 통해서만
은닉 상태 $\mathbf{h}_T$에 의존합니다.
따라서 연쇄 법칙을 사용하여 그래디언트
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$을 쉽게 구할 수 있습니다.

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

임의의 타임스텝 $t < T$에 대해서는 더 까다로워지는데,
여기서 목적 함수 $L$은
$\mathbf{h}_{t+1}$과 $\mathbf{o}_t$를 통해 $\mathbf{h}_t$에 의존합니다.
연쇄 법칙에 따라,
임의의 타임스텝 $t < T$에서 은닉 상태의 그래디언트
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$은
다음과 같이 순환적으로 계산될 수 있습니다.


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

분석을 위해, 임의의 타임스텝 $1 \leq t \leq T$에 대해
순환 계산을 전개하면 다음을 얻습니다.

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

:eqref:`eq_bptt_partial_L_ht`로부터 저희는
이 단순한 선형 예제조차도 이미 긴 시퀀스 모델의 핵심 문제 몇 가지를
드러낸다는 것을 알 수 있습니다.
이는 $\mathbf{W}_\textrm{hh}^\top$의 잠재적으로 매우 큰 거듭제곱을 포함합니다.
그 안에서 1보다 작은 고윳값은 소멸하고
1보다 큰 고윳값은 발산합니다.
이는 수치적으로 불안정하며,
그래디언트 소실과 폭주의 형태로 드러납니다.
이를 다루는 한 가지 방법은 :numref:`subsec_bptt_analysis`에서 논의된 대로
계산적으로 편리한 크기에서 타임스텝을 절단하는 것입니다.
실제로, 이 절단은 주어진 타임스텝 수 이후에
그래디언트를 분리(detach)함으로써도 영향을 받을 수 있습니다.
이후 저희는 장단기 메모리(LSTM)와 같이 더 정교한 시퀀스 모델이
이를 어떻게 더 완화할 수 있는지 보게 될 것입니다.

마지막으로, :numref:`fig_rnn_bptt`은
목적 함수 $L$이
은닉 상태 $\mathbf{h}_1, \ldots, \mathbf{h}_T$을 통해
은닉 층의 모델 파라미터 $\mathbf{W}_\textrm{hx}$와 $\mathbf{W}_\textrm{hh}$에
의존한다는 것을 보여 줍니다.
이러한 파라미터에 대한 그래디언트
$\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$과 $\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$을 계산하기 위해,
저희는 연쇄 법칙을 적용하면 다음을 얻습니다.

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

여기서 :eqref:`eq_bptt_partial_L_hT_final_step`과
:eqref:`eq_bptt_partial_L_ht_recur`에 의해
순환적으로 계산되는 $\partial L/\partial \mathbf{h}_t$는
수치적 안정성에 영향을 미치는 핵심 양입니다.



:numref:`sec_backprop`에서 설명한 것처럼,
BPTT는 RNN에서의 역전파의 적용이므로,
RNN 훈련은 순방향 전파와 BPTT를 번갈아 가며 진행합니다.
또한, BPTT는
위의 그래디언트를 차례로 계산하고 저장합니다.
구체적으로, 저장된 중간값은
중복 계산을 피하기 위해 재사용되며,
예를 들어 $\partial L / \partial \mathbf{W}_\textrm{hx}$와
$\partial L / \partial \mathbf{W}_\textrm{hh}$의 계산 모두에 사용되도록
$\partial L/\partial \mathbf{h}_t$를 저장합니다.


## 요약

BPTT는 은닉 상태를 가진 시퀀스 모델에 대한 역전파의 적용일 뿐입니다.
계산상의 편의와 수치적 안정성을 위해 정규 또는 무작위와 같은 절단이 필요합니다.
행렬의 높은 거듭제곱은 발산하거나 소멸하는 고윳값으로 이어질 수 있습니다. 이는 폭주 또는 소실 그래디언트의 형태로 드러납니다.
효율적인 계산을 위해, BPTT 동안 중간값이 캐시됩니다.



## 연습문제

1. 저희가 고윳값 $\lambda_i$와 그에 대응하는 고유벡터가 $\mathbf{v}_i$($i = 1, \ldots, n$)인 대칭 행렬 $\mathbf{M} \in \mathbb{R}^{n \times n}$을 가지고 있다고 가정합시다. 일반성을 잃지 않고, 그것들이 $|\lambda_i| \geq |\lambda_{i+1}|$의 순서로 정렬되어 있다고 가정합시다.
   1. $\mathbf{M}^k$이 고윳값 $\lambda_i^k$을 가짐을 보이세요.
   1. 무작위 벡터 $\mathbf{x} \in \mathbb{R}^n$에 대해, $\mathbf{M}^k \mathbf{x}$이 $\mathbf{M}$의 고유벡터 $\mathbf{v}_1$
과 매우 잘 정렬될 것임을 높은 확률로 증명하세요. 이 진술을 형식화하세요.
   1. 위 결과는 RNN의 그래디언트에 대해 무엇을 의미하나요?
1. 그래디언트 클리핑 외에, 순환 신경망에서 그래디언트 폭주를 다루기 위한 다른 방법을 떠올릴 수 있나요?

[Discussions](https://discuss.d2l.ai/t/334)
