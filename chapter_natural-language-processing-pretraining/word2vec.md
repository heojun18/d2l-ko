# 단어 임베딩 (word2vec)
:label:`sec_word2vec`


자연어는 의미를 표현하기 위해 사용되는 복잡한 시스템입니다.
이 시스템에서 단어는 의미의 기본 단위입니다.
이름에서 알 수 있듯이,
*단어 벡터(word vector)* 는 단어를 표현하기 위해 사용되는 벡터이며,
단어의 특징 벡터나 표현으로도 간주될 수 있습니다.
단어를 실수 벡터로 매핑하는 기법을
*단어 임베딩(word embedding)* 이라고 합니다.
최근 몇 년간,
단어 임베딩은 점차
자연어 처리의 기본 지식이 되었습니다.


## 원-핫 벡터는 좋지 않은 선택입니다

:numref:`sec_rnn-scratch`에서 저희는 원-핫 벡터를 사용해 단어(문자도 단어임)를 표현했습니다.
사전에 있는 서로 다른 단어의 개수(사전 크기)가 $N$이고,
각 단어가 $0$부터 $N-1$까지의 서로 다른 정수(인덱스)에
대응한다고 가정합시다.
인덱스 $i$를 가진 임의의 단어에 대한
원-핫 벡터 표현을 얻기 위해,
모든 원소가 0인 길이 $N$짜리 벡터를 만들고
위치 $i$의 원소를 1로 설정합니다.
이러한 방식으로 각 단어는 길이 $N$짜리 벡터로 표현되며,
신경망에서 직접 사용될 수 있습니다.


원-핫 단어 벡터는 만들기 쉬우나,
일반적으로 좋은 선택은 아닙니다.
주된 이유는 원-핫 단어 벡터가 서로 다른 단어들 간의 유사도(예: 저희가 자주 사용하는 *코사인 유사도*)를 정확히 표현할 수 없다는 것입니다.
벡터 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$에 대해, 이들의 코사인 유사도는 두 벡터 사이 각의 코사인 값입니다.


$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$


서로 다른 두 단어의 원-핫 벡터 사이의 코사인 유사도는 0이므로,
원-핫 벡터는 단어 간의 유사도를 인코딩할 수 없습니다.


## 자기 지도 word2vec

[word2vec](https://code.google.com/archive/p/word2vec/) 도구는 위의 문제를 해결하기 위해 제안되었습니다.
이는 각 단어를 고정 길이의 벡터로 매핑하며, 이러한 벡터는 서로 다른 단어들 간의 유사도 및 유추 관계를 더 잘 표현할 수 있습니다.
word2vec 도구는 두 가지 모델을 포함합니다. 즉 *스킵그램(skip-gram)* :cite:`Mikolov.Sutskever.Chen.ea.2013`과 *연속 단어 묶음(continuous bag of words, CBOW)* :cite:`Mikolov.Chen.Corrado.ea.2013`입니다.
의미적으로 유의미한 표현을 위해,
이들의 학습은 어떤 단어를 말뭉치 내에서 주변 단어 일부를 사용해 예측하는 것으로 볼 수 있는
조건부 확률에
의존합니다.
지도가 레이블 없는 데이터에서 오기 때문에,
스킵그램과 연속 단어 묶음은
모두 자기 지도 모델입니다.

다음에서는 이 두 모델과 그 학습 방법을 소개하겠습니다.


## 스킵그램 모델
:label:`subsec_skip-gram`

*스킵그램* 모델은 한 단어가 텍스트 시퀀스에서 주변 단어를 생성하는 데 사용될 수 있다고 가정합니다.
예시로 텍스트 시퀀스 "the", "man", "loves", "his", "son"을 살펴봅시다.
"loves"를 *중심 단어(center word)* 로 선택하고 문맥 윈도우 크기를 2로 설정합시다.
:numref:`fig_skip_gram`에서 보듯이,
중심 단어 "loves"가 주어졌을 때,
스킵그램 모델은
*문맥 단어(context word)*("the", "man", "his", "son")를 생성할 조건부 확률을 고려합니다.
이들은 중심 단어로부터 2 단어 이내에 위치합니다.

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

중심 단어가 주어졌을 때 문맥 단어가 독립적으로 생성된다고 가정합시다(즉, 조건부 독립).
이 경우, 위의 조건부 확률은
다음과 같이 다시 쓸 수 있습니다.

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![스킵그램 모델은 중심 단어가 주어졌을 때 주변 문맥 단어를 생성할 조건부 확률을 고려합니다.](../img/skip-gram.svg)
:label:`fig_skip_gram`

스킵그램 모델에서 각 단어는
조건부 확률을 계산하기 위한
두 개의 $d$차원 벡터 표현을 가집니다.
더 구체적으로,
사전에 인덱스 $i$를 가진 임의의 단어에 대해,
*중심* 단어로 사용될 때와 *문맥* 단어로 사용될 때의
두 벡터를 각각 $\mathbf{v}_i\in\mathbb{R}^d$,
$\mathbf{u}_i\in\mathbb{R}^d$로 나타냅니다.
중심 단어 $w_c$(사전에서 인덱스 $c$)가 주어졌을 때 임의의 문맥 단어 $w_o$(사전에서 인덱스 $o$)가 생성될 조건부 확률은
벡터 내적에 대한 소프트맥스 연산으로 모델링할 수 있습니다.


$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

여기서 어휘 인덱스 집합은 $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$입니다.
길이 $T$의 텍스트 시퀀스가 주어졌을 때, 시간 단계 $t$의 단어를 $w^{(t)}$로 표기합니다.
임의의 중심 단어가 주어졌을 때
문맥 단어가 독립적으로 생성된다고 가정합시다.
문맥 윈도우 크기가 $m$일 때,
스킵그램 모델의 가능도 함수는
임의의 중심 단어가 주어졌을 때
모든 문맥 단어가 생성될 확률입니다.


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

여기서 $1$보다 작거나 $T$보다 큰 시간 단계는 생략될 수 있습니다.

### 학습

스킵그램 모델의 파라미터는 어휘의 각 단어에 대한 중심 단어 벡터와 문맥 단어 벡터입니다.
학습에서는 가능도 함수를 최대화함으로써(즉, 최대 가능도 추정) 모델 파라미터를 학습합니다. 이는 다음 손실 함수를 최소화하는 것과 동등합니다.

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \textrm{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

확률적 경사 하강법을 사용해 손실을 최소화할 때,
각 반복에서 모델 파라미터를 업데이트하기 위해
더 짧은 부분 시퀀스를 무작위로 샘플링하여
이 부분 시퀀스에 대한 (확률적) 경사를 계산할 수 있습니다.
이 (확률적) 경사를 계산하려면,
중심 단어 벡터와 문맥 단어 벡터에 대한
로그 조건부 확률의 경사를
구해야 합니다.
일반적으로 :eqref:`eq_skip-gram-softmax`에 따르면,
중심 단어 $w_c$와
문맥 단어 $w_o$의 어떤 쌍을 포함하는
로그 조건부 확률은


$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

미분을 통해 중심 단어 벡터 $\mathbf{v}_c$에 대한 그 경사를
다음과 같이 구할 수 있습니다.

$$\begin{aligned}\frac{\partial \textrm{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`


:eqref:`eq_skip-gram-grad`의 계산은 $w_c$를 중심 단어로 하는 사전 내 모든 단어의 조건부 확률을 필요로 함에 유의하십시오.
나머지 단어 벡터에 대한 경사도 같은 방식으로 구할 수 있습니다.


학습 후, 사전에 인덱스 $i$를 가진 임의의 단어에 대해 저희는 두 단어 벡터
$\mathbf{v}_i$ (중심 단어로서)와 $\mathbf{u}_i$ (문맥 단어로서)를 모두 얻습니다.
자연어 처리 응용에서는 일반적으로 스킵그램 모델의 중심 단어 벡터가
단어 표현으로 사용됩니다.


## 연속 단어 묶음(CBOW) 모델


*연속 단어 묶음(CBOW)* 모델은 스킵그램 모델과 유사합니다.
스킵그램 모델과의 주된 차이는
연속 단어 묶음 모델이
텍스트 시퀀스에서 주변 문맥 단어를 바탕으로
중심 단어가 생성된다고 가정하는 것입니다.
예를 들어,
같은 텍스트 시퀀스 "the", "man", "loves", "his", "son"에서, "loves"를 중심 단어로 하고 문맥 윈도우 크기가 2일 때,
연속 단어 묶음 모델은
문맥 단어 "the", "man", "his", "son"을 바탕으로 중심 단어 "loves"를 생성할 조건부 확률을 고려합니다(:numref:`fig_cbow` 참조).

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![연속 단어 묶음 모델은 주변 문맥 단어가 주어졌을 때 중심 단어를 생성할 조건부 확률을 고려합니다.](../img/cbow.svg)
:label:`fig_cbow`


연속 단어 묶음 모델에는
여러 문맥 단어가 있으므로,
조건부 확률 계산에서
이러한 문맥 단어 벡터들이 평균화됩니다.
구체적으로,
사전에서 인덱스 $i$를 가진 임의의 단어에 대해,
*문맥* 단어와 *중심* 단어로 사용될 때의
두 벡터를 각각 $\mathbf{v}_i\in\mathbb{R}^d$,
$\mathbf{u}_i\in\mathbb{R}^d$로 표시합니다
(스킵그램 모델에서 의미가 뒤바뀝니다).
주변 문맥 단어 $w_{o_1}, \ldots, w_{o_{2m}}$(사전에서 인덱스 $o_1, \ldots, o_{2m}$)이 주어졌을 때 임의의 중심 단어 $w_c$(사전에서 인덱스 $c$)가 생성될 조건부 확률은 다음과 같이 모델링할 수 있습니다.



$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\exp\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \exp\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`


간결성을 위해, $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$과 $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}} \right)/(2m)$로 두겠습니다. 그러면 :eqref:`fig_cbow-full`은 다음과 같이 간단해질 수 있습니다.

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

길이 $T$의 텍스트 시퀀스가 주어졌을 때, 시간 단계 $t$의 단어를 $w^{(t)}$로 표시합니다.
문맥 윈도우 크기가 $m$일 때,
연속 단어 묶음 모델의 가능도 함수는
문맥 단어가 주어졌을 때
모든 중심 단어가 생성될 확률입니다.


$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### 학습

연속 단어 묶음 모델의 학습은
스킵그램 모델의 학습과
거의 동일합니다.
연속 단어 묶음 모델의 최대 가능도 추정은 다음 손실 함수를 최소화하는 것과 동등합니다.



$$  -\sum_{t=1}^T  \textrm{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

다음을 살펴봅시다.

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

미분을 통해, 임의의 문맥 단어 벡터 $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$)에 대한 그 경사를
다음과 같이 구할 수 있습니다.


$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`


나머지 단어 벡터에 대한 경사도 같은 방식으로 구할 수 있습니다.
스킵그램 모델과 달리,
연속 단어 묶음 모델은
일반적으로
문맥 단어 벡터를 단어 표현으로 사용합니다.




## 요약

* 단어 벡터는 단어를 표현하기 위해 사용되는 벡터이며, 단어의 특징 벡터나 표현으로도 간주될 수 있습니다. 단어를 실수 벡터로 매핑하는 기법을 단어 임베딩이라고 합니다.
* word2vec 도구는 스킵그램 모델과 연속 단어 묶음 모델을 모두 포함합니다.
* 스킵그램 모델은 한 단어가 텍스트 시퀀스에서 주변 단어를 생성하는 데 사용될 수 있다고 가정합니다. 반면 연속 단어 묶음 모델은 주변 문맥 단어를 바탕으로 중심 단어가 생성된다고 가정합니다.



## 연습문제

1. 각 경사를 계산하기 위한 계산 복잡도는 얼마입니까? 사전 크기가 매우 크다면 어떤 문제가 발생할 수 있을까요?
1. 영어의 일부 고정된 구절은 "new york" 같이 여러 단어로 구성됩니다. 이들의 단어 벡터는 어떻게 학습합니까? 힌트: word2vec 논문 :cite:`Mikolov.Sutskever.Chen.ea.2013`의 4절을 참조하십시오.
1. 스킵그램 모델을 예로 들어 word2vec 설계를 되짚어 봅시다. 스킵그램 모델에서 두 단어 벡터의 내적과 코사인 유사도의 관계는 무엇입니까? 의미가 유사한 한 쌍의 단어에 대해, (스킵그램 모델로 학습한) 그들의 단어 벡터의 코사인 유사도가 높을 수 있는 이유는 무엇입니까?

[Discussions](https://discuss.d2l.ai/t/381)
