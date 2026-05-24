# 근사 학습
:label:`sec_approx_train`

:numref:`sec_word2vec`의 논의를 떠올려 보십시오.
스킵그램 모델의 주된 아이디어는
:eqref:`eq_skip-gram-softmax`에서
주어진 중심 단어 $w_c$에 기반하여
문맥 단어 $w_o$를 생성할
조건부 확률을
소프트맥스 연산을 사용해 계산하는 것이며,
이에 대응하는 로그 손실은
:eqref:`eq_skip-gram-log`의 반대로 주어집니다.



소프트맥스 연산의 성질로 인해,
문맥 단어가
사전 $\mathcal{V}$의 어떤 단어든 될 수 있으므로,
:eqref:`eq_skip-gram-log`의 반대는
어휘 전체 크기만큼의
항에 대한 합을 포함합니다.
따라서,
:eqref:`eq_skip-gram-grad`의 스킵그램 모델과
:eqref:`eq_cbow-gradient`의 연속 단어 묶음 모델에 대한
경사 계산은
모두
이 합을 포함합니다.
유감스럽게도,
큰 사전(흔히 수십만 또는 수백만 개의 단어)에 대해
합산을 수행하는 이러한 경사의
계산 비용은
엄청납니다!

위에서 언급한 계산 복잡도를 줄이기 위해, 이 절에서는 두 가지 근사 학습 방법을 소개합니다.
*네거티브 샘플링(negative sampling)* 과 *계층적 소프트맥스(hierarchical softmax)* 입니다.
스킵그램 모델과
연속 단어 묶음 모델의 유사성으로 인해,
저희는 단지 스킵그램 모델을 예로 들어
이 두 가지 근사 학습 방법을 설명하겠습니다.

## 네거티브 샘플링
:label:`subsec_negative-sampling`


네거티브 샘플링은 원래의 목적 함수를 수정합니다.
중심 단어 $w_c$의 문맥 윈도우가 주어졌을 때,
임의의 (문맥) 단어 $w_o$가
이 문맥 윈도우에서 나온다는 사실은
다음과 같이 확률이 모델링되는
사건으로 간주됩니다.


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

여기서 $\sigma$는 시그모이드 활성화 함수의 정의를 사용합니다.

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

단어 임베딩을 학습하기 위해
텍스트 시퀀스에서
이러한 모든 사건의 결합 확률을
최대화하는 것부터 시작해 봅시다.
구체적으로,
길이 $T$의 텍스트 시퀀스가 주어졌을 때,
시간 단계 $t$의 단어를 $w^{(t)}$로 표기하고
문맥 윈도우 크기를 $m$이라고 하겠습니다. 그리고 다음의 결합 확률을 최대화하는 것을 고려하십시오.


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`


그러나
:eqref:`eq-negative-sample-pos`은
양성 예제를 포함하는 사건만을
고려합니다.
결과적으로,
:eqref:`eq-negative-sample-pos`의 결합 확률은
모든 단어 벡터가 무한대와 같을 때에만
1로 최대화됩니다.
물론,
이러한 결과는 무의미합니다.
목적 함수를
더 의미 있게 만들기 위해,
*네거티브 샘플링*
은 미리 정의된 분포에서 샘플링된
음성 예제를 추가합니다.

문맥 단어 $w_o$가
중심 단어 $w_c$의 문맥 윈도우에서 나온다는
사건을 $S$로 표시합시다.
$w_o$를 포함하는 이 사건에 대해,
미리 정의된 분포 $P(w)$에서
이 문맥 윈도우에서 나오지 않은
$K$개의 *노이즈 단어(noise word)* 를 샘플링합니다.
노이즈 단어 $w_k$ ($k=1, \ldots, K$)가
$w_c$의 문맥 윈도우에서 나오지 않은
사건을 $N_k$로 표시합시다.
양성 예제와 음성 예제를 모두 포함하는
이러한 사건들 $S, N_1, \ldots, N_K$가
서로 독립이라고 가정합시다.
네거티브 샘플링은
:eqref:`eq-negative-sample-pos`의 (양성 예제만 포함하는)
결합 확률을 다음과 같이 다시 씁니다.

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

여기서 조건부 확률은
사건 $S, N_1, \ldots, N_K$를 통해 근사됩니다.

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

텍스트 시퀀스의 시간 단계 $t$에서의
단어 $w^{(t)}$와
노이즈 단어 $w_k$의 인덱스를
각각 $i_t$와 $h_k$로 표시합시다.
:eqref:`eq-negative-sample-conditional-prob`의 조건부 확률에 대한 로그 손실은 다음과 같습니다.

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$


이제 각 학습 단계에서의 경사 계산 비용이
사전 크기와 무관하며,
$K$에 선형적으로 의존한다는 것을
알 수 있습니다.
하이퍼파라미터 $K$를
더 작은 값으로 설정하면,
네거티브 샘플링을 사용한 각 학습 단계에서의
경사 계산 비용이
더 작아집니다.




## 계층적 소프트맥스

대안적인 근사 학습 방법으로,
*계층적 소프트맥스* 는
:numref:`fig_hi_softmax`에 묘사된
데이터 구조인 이진 트리를 사용하며,
트리의 각 잎 노드는
사전 $\mathcal{V}$의 한 단어를 표현합니다.

![근사 학습을 위한 계층적 소프트맥스. 트리의 각 잎 노드는 사전의 한 단어를 표현합니다.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

루트 노드에서 이진 트리의 단어 $w$를 표현하는 잎 노드까지의
경로상의 노드 수(양 끝 포함)를
$L(w)$로 표시합시다.
$n(w,j)$를 이 경로상의 $j^\textrm{번째}$ 노드라 하고,
그 문맥 단어 벡터를
$\mathbf{u}_{n(w, j)}$로 표시합니다.
예를 들어,
:numref:`fig_hi_softmax`에서 $L(w_3) = 4$입니다.
계층적 소프트맥스는 :eqref:`eq_skip-gram-softmax`의 조건부 확률을 다음과 같이 근사합니다.


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

여기서 함수 $\sigma$
는 :eqref:`eq_sigma-f`에서 정의되며,
$\textrm{leftChild}(n)$은 노드 $n$의 왼쪽 자식 노드입니다. $x$가 참이면 $[\![x]\!] = 1$이고, 그렇지 않으면 $[\![x]\!] = -1$입니다.

이를 설명하기 위해,
:numref:`fig_hi_softmax`에서 단어 $w_c$가 주어졌을 때
단어 $w_3$를 생성할
조건부 확률을 계산해 봅시다.
이는 $w_c$의 단어 벡터 $\mathbf{v}_c$와
루트에서 $w_3$까지의 경로(:numref:`fig_hi_softmax`에서 굵게 표시된 경로)상의
잎이 아닌 노드 벡터들 사이의 내적이
필요하며, 이 경로는 왼쪽, 오른쪽, 그다음 왼쪽으로 진행합니다.


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

$\sigma(x)+\sigma(-x) = 1$이므로,
임의의 단어 $w_c$를 기반으로
사전 $\mathcal{V}$의 모든 단어를 생성할
조건부 확률은
합쳐서 1이 됨이 성립합니다.

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

다행히, 이진 트리 구조로 인해 $L(w_o)-1$이 $\mathcal{O}(\textrm{log}_2|\mathcal{V}|)$ 차수이므로,
사전 크기 $\mathcal{V}$가 매우 클 때
계층적 소프트맥스를 사용한 각 학습 단계의 계산 비용은
근사 학습을 사용하지 않은 경우와 비교해
크게 감소합니다.

## 요약

* 네거티브 샘플링은 양성 예제와 음성 예제를 모두 포함하는 서로 독립인 사건들을 고려하여 손실 함수를 구성합니다. 학습 계산 비용은 각 단계에서 노이즈 단어의 수에 선형적으로 의존합니다.
* 계층적 소프트맥스는 이진 트리에서 루트 노드부터 잎 노드까지의 경로를 사용하여 손실 함수를 구성합니다. 학습 계산 비용은 각 단계에서 사전 크기의 로그에 의존합니다.

## 연습문제

1. 네거티브 샘플링에서 노이즈 단어를 어떻게 샘플링할 수 있습니까?
1. :eqref:`eq_hi-softmax-sum-one`이 성립함을 검증하십시오.
1. 네거티브 샘플링과 계층적 소프트맥스로 연속 단어 묶음 모델을 각각 어떻게 학습합니까?

[Discussions](https://discuss.d2l.ai/t/382)
