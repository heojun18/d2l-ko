# 전역 벡터를 이용한 단어 임베딩 (GloVe)
:label:`sec_glove`


문맥 윈도우 내에서의
단어-단어 동시 출현은
풍부한 의미 정보를 담고 있을 수 있습니다.
예를 들어,
큰 말뭉치에서
"solid"라는 단어는
"steam"보다 "ice"와
동시 출현할 가능성이 더 높지만,
"gas"라는 단어는
아마도 "ice"보다 "steam"과
더 자주 동시 출현할 것입니다.
또한,
이러한 동시 출현에 대한
전역 말뭉치 통계는
미리 계산될 수 있으며, 이는 더 효율적인 학습으로 이어질 수 있습니다.
단어 임베딩을 위해
전체 말뭉치의 통계 정보를 활용하기 위해,
먼저 :numref:`subsec_skip-gram`의
스킵그램 모델을 다시 살펴봅시다.
다만 동시 출현 횟수 같은
전역 말뭉치 통계를 사용하여
이를 해석해 봅시다.

## 전역 말뭉치 통계를 이용한 스킵그램
:label:`subsec_skipgram-global`

스킵그램 모델에서 단어 $w_i$가 주어졌을 때
단어 $w_j$의 조건부 확률
$P(w_j\mid w_i)$를
$q_{ij}$로 표기하면,
다음을 얻습니다.

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

여기서
임의의 인덱스 $i$에 대해
벡터 $\mathbf{v}_i$와 $\mathbf{u}_i$는
단어 $w_i$를 중심 단어와 문맥 단어로 각각 표현하며,
$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$는
어휘의 인덱스 집합입니다.

말뭉치에서 여러 번 등장할 수 있는
단어 $w_i$를 고려합시다.
전체 말뭉치에서,
$w_i$가 그들의 중심 단어로 사용된
모든 문맥 단어들은
*같은 원소의 여러 인스턴스를 허용하는*
단어 인덱스의 *다중집합(multiset)* $\mathcal{C}_i$를
형성합니다.
임의의 원소에 대해,
그 인스턴스의 수를 *중복도(multiplicity)* 라고 합니다.
예를 들어 설명하자면,
단어 $w_i$가 말뭉치에서 두 번 등장하고
두 문맥 윈도우에서 $w_i$를 그들의 중심 단어로 하는
문맥 단어들의 인덱스가
$k, j, m, k$와 $k, l, k, j$라고 가정합시다.
따라서, 다중집합 $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$이며,
여기서 원소 $j, k, l, m$의 중복도는
각각 2, 4, 1, 1입니다.

이제 다중집합 $\mathcal{C}_i$에서 원소 $j$의 중복도를
$x_{ij}$로 표기합시다.
이는 전체 말뭉치의
같은 문맥 윈도우에서
단어 $w_j$(문맥 단어로서)와
단어 $w_i$(중심 단어로서)의
전역 동시 출현 횟수입니다.
이러한 전역 말뭉치 통계를 사용하면,
스킵그램 모델의 손실 함수는
다음과 동등합니다.

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

저희는 또한
$w_i$가 그들의 중심 단어로 등장하는
문맥 윈도우 내 모든 문맥 단어의 수를
$x_i$로 표기하는데,
이는 $|\mathcal{C}_i|$와 동등합니다.
$p_{ij}$를
중심 단어 $w_i$가 주어졌을 때
문맥 단어 $w_j$를 생성할 조건부 확률
$x_{ij}/x_i$라 하면,
:eqref:`eq_skipgram-x_ij`은
다음과 같이 다시 쓸 수 있습니다.

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij`에서, $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$는
전역 말뭉치 통계의
조건부 분포 $p_{ij}$와
모델 예측의
조건부 분포 $q_{ij}$의
교차 엔트로피를
계산합니다.
이 손실은
위에서 설명한 대로 $x_i$로도 가중치가 적용됩니다.
:eqref:`eq_skipgram-p_ij`의 손실 함수를 최소화하는 것은
예측된 조건부 분포가
전역 말뭉치 통계로부터의 조건부 분포에
가까워지게 할 것입니다.


확률 분포 사이의 거리를 측정하는 데
흔히 사용되지만,
교차 엔트로피 손실 함수는 여기서 좋은 선택이 아닐 수 있습니다.
한편으로는, :numref:`sec_approx_train`에서 언급했듯이,
$q_{ij}$를 적절히 정규화하는 비용이
전체 어휘에 대한 합을 초래하며,
이는 계산적으로 비쌀 수 있습니다.
다른 한편으로는,
큰 말뭉치에서 나오는 다수의 드문 사건들이
교차 엔트로피 손실에 의해
과도한 가중치를 할당받도록
모델링되는 경우가 많습니다.

## GloVe 모델

이를 고려하여,
*GloVe* 모델은 제곱 손실에 기반하여 스킵그램 모델에 세 가지 변경을 가합니다 :cite:`Pennington.Socher.Manning.2014`.

1. 확률 분포가 아닌 변수 $p'_{ij}=x_{ij}$와 $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$를 사용하고 둘 다 로그를 취하여, 제곱 손실 항은 $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$이 됩니다.
2. 각 단어 $w_i$에 대해 두 개의 스칼라 모델 파라미터를 추가합니다. 중심 단어 편향 $b_i$와 문맥 단어 편향 $c_i$입니다.
3. 각 손실 항의 가중치를 가중치 함수 $h(x_{ij})$로 대체하며, 여기서 $h(x)$는 $[0, 1]$ 구간에서 증가하는 함수입니다.

이 모든 것을 합쳐 GloVe를 학습하는 것은 다음 손실 함수를 최소화하는 것입니다.

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

가중치 함수에 대해 권장되는 선택은 다음과 같습니다.
$x < c$ (예: $c = 100$)이면 $h(x) = (x/c) ^\alpha$ (예: $\alpha = 0.75$), 그렇지 않으면 $h(x) = 1$입니다.
이 경우,
$h(0)=0$이므로,
$x_{ij}=0$인 임의의 경우에 대한 제곱 손실 항은
계산 효율을 위해 생략할 수 있습니다.
예를 들어,
학습에 미니배치 확률적 경사 하강법을 사용할 때,
각 반복에서
저희는 경사를 계산하고 모델 파라미터를 업데이트하기 위해
*0이 아닌* $x_{ij}$의 미니배치를 무작위로 샘플링합니다.
이러한 0이 아닌 $x_{ij}$는 미리 계산된
전역 말뭉치 통계임에 유의하십시오.
따라서, 모델은 *전역 벡터(Global Vectors)* 라는 의미에서 GloVe로 불립니다.

다음을 강조해야 합니다.
단어 $w_i$가 단어 $w_j$의 문맥 윈도우에 등장하면,
*그 역도 마찬가지*입니다.
따라서, $x_{ij}=x_{ji}$입니다.
비대칭 조건부 확률 $p_{ij}$에 적합한 word2vec과 달리,
GloVe는 대칭 $\log \, x_{ij}$에 적합합니다.
따라서, GloVe 모델에서 임의 단어의 중심 단어 벡터와
문맥 단어 벡터는 수학적으로 동등합니다.
그러나 실제로는, 서로 다른 초기화 값으로 인해,
같은 단어가 학습 후 이 두 벡터에서 여전히 서로 다른 값을 가질 수 있습니다.
GloVe는 이들을 출력 벡터로 합산합니다.



## 동시 출현 확률의 비율로 GloVe 해석하기


저희는 다른 관점에서 GloVe 모델을 해석할 수도 있습니다.
:numref:`subsec_skipgram-global`의 같은 표기법을 사용해,
$p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$를 말뭉치에서 $w_i$를 중심 단어로 했을 때 문맥 단어 $w_j$를 생성할 조건부 확률이라 합시다.
:numref:`tab_glove`는
"ice"와 "steam"이라는 단어가 주어졌을 때
큰 말뭉치의 통계에 기반한
여러 동시 출현 확률과 그 비율을 나열합니다.


:큰 말뭉치로부터의 단어-단어 동시 출현 확률과 그 비율 (:citet:`Pennington.Socher.Manning.2014`의 표 1에서 차용)
:label:`tab_glove`

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|



:numref:`tab_glove`에서 다음을 관찰할 수 있습니다.

* "ice"와 관련 있지만 "steam"과 관련 없는 단어 $w_k$, 예를 들어 $w_k=\textrm{solid}$의 경우, 저희는 8.9와 같은 더 큰 동시 출현 확률의 비율을 기대합니다.
* "steam"과 관련 있지만 "ice"와 관련 없는 단어 $w_k$, 예를 들어 $w_k=\textrm{gas}$의 경우, 저희는 0.085와 같은 더 작은 동시 출현 확률의 비율을 기대합니다.
* "ice"와 "steam" 양쪽 모두와 관련 있는 단어 $w_k$, 예를 들어 $w_k=\textrm{water}$의 경우, 저희는 1.36과 같이 1에 가까운 동시 출현 확률의 비율을 기대합니다.
* "ice"와 "steam" 양쪽 모두와 관련 없는 단어 $w_k$, 예를 들어 $w_k=\textrm{fashion}$의 경우, 저희는 0.96과 같이 1에 가까운 동시 출현 확률의 비율을 기대합니다.




동시 출현 확률의 비율이
단어 간의 관계를
직관적으로 표현할 수 있음을
알 수 있습니다.
따라서 저희는 이 비율에 적합한
세 단어 벡터의 함수를 설계할 수 있습니다.
$w_i$를 중심 단어로 하고
$w_j$와 $w_k$를 문맥 단어로 하는
동시 출현 확률의 비율
${p_{ij}}/{p_{ik}}$에 대해,
저희는 어떤 함수 $f$를 사용해
이 비율에 적합하기를 원합니다.

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

$f$에 대한 많은 가능한 설계 중에서,
저희는 다음에서 합리적인 선택 하나만 고릅니다.
동시 출현 확률의 비율이
스칼라이므로,
저희는 $f$가 스칼라 함수일 것을 요구합니다. 예를 들어
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$입니다.
:eqref:`eq_glove-f`에서
단어 인덱스 $j$와 $k$를 바꾸면,
$f(x)f(-x)=1$이 성립해야 하므로,
한 가지 가능성은 $f(x)=\exp(x)$입니다.
즉,

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

이제
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$를 고릅시다.
여기서 $\alpha$는 상수입니다.
$p_{ij}=x_{ij}/x_i$이므로, 양변에 로그를 취하면 $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$을 얻습니다.
$- \log\, \alpha + \log\, x_i$에 적합하기 위해 추가 편향 항을 사용할 수 있습니다. 예를 들어 중심 단어 편향 $b_i$와 문맥 단어 편향 $c_j$입니다.

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

가중치를 적용한 :eqref:`eq_glove-square`의
제곱 오차를 측정하면,
:eqref:`eq_glove-loss`의 GloVe 손실 함수를 얻습니다.



## 요약

* 스킵그램 모델은 단어-단어 동시 출현 횟수 같은 전역 말뭉치 통계를 사용하여 해석할 수 있습니다.
* 교차 엔트로피 손실은 두 확률 분포의 차이를 측정하는 데 좋은 선택이 아닐 수 있습니다. 특히 큰 말뭉치의 경우에 그렇습니다. GloVe는 미리 계산된 전역 말뭉치 통계에 적합하기 위해 제곱 손실을 사용합니다.
* GloVe에서 임의 단어에 대해 중심 단어 벡터와 문맥 단어 벡터는 수학적으로 동등합니다.
* GloVe는 단어-단어 동시 출현 확률의 비율로부터 해석할 수 있습니다.


## 연습문제

1. 단어 $w_i$와 $w_j$가 같은 문맥 윈도우에서 동시 출현한다면, 텍스트 시퀀스에서 그들의 거리를 사용해 조건부 확률 $p_{ij}$를 계산하는 방법을 어떻게 재설계할 수 있습니까? 힌트: GloVe 논문 :cite:`Pennington.Socher.Manning.2014`의 4.2절을 참조하십시오.
1. 임의의 단어에 대해, GloVe에서 그 중심 단어 편향과 문맥 단어 편향은 수학적으로 동등합니까? 왜 그렇습니까?


[Discussions](https://discuss.d2l.ai/t/385)
