# 정보 이론
:label:`sec_information_theory`

우주는 정보로 흘러넘치고 있습니다. 정보는 학문적 균열을 넘어 공통된 언어를 제공합니다. 셰익스피어의 소네트에서 코넬 ArXiv의 연구자 논문까지, 반 고흐의 인쇄물 별이 빛나는 밤에서 베토벤의 음악 교향곡 5번까지, 첫 번째 프로그래밍 언어 Plankalkül에서 최첨단 머신러닝 알고리즘까지. 모든 것은 형식에 관계없이 정보 이론의 규칙을 따라야 합니다. 정보 이론을 가지고, 저희는 다른 신호에 얼마나 많은 정보가 존재하는지 측정하고 비교할 수 있습니다. 이 절에서, 저희는 정보 이론의 기본 개념과 머신러닝에서의 정보 이론의 응용을 조사할 것입니다.

시작하기 전에, 머신러닝과 정보 이론 사이의 관계를 개략적으로 설명하겠습니다. 머신러닝은 데이터에서 흥미로운 신호를 추출하고 중요한 예측을 하는 것을 목표로 합니다. 반면에, 정보 이론은 정보의 인코딩, 디코딩, 전송, 그리고 조작을 연구합니다. 결과적으로, 정보 이론은 머신러닝 시스템에서의 정보 처리를 논의하기 위한 기본 언어를 제공합니다. 예를 들어, 많은 머신러닝 응용은 :numref:`sec_softmax`에서 설명된 교차 엔트로피 손실을 사용합니다. 이 손실은 정보 이론적 고려사항으로부터 직접 유도될 수 있습니다.


## 정보

정보 이론의 "영혼"인 정보로 시작합시다. *정보*는 하나 이상의 인코딩 형식의 특정 시퀀스를 가진 어떤 것에도 인코딩될 수 있습니다. 저희가 정보의 개념을 정의하려고 한다고 가정해 봅시다. 저희의 시작점이 무엇이 될 수 있을까요?

다음 사고 실험을 고려해 보십시오. 저희에게 카드 한 벌이 있는 친구가 있습니다. 그들은 덱을 섞고, 일부 카드를 뒤집고, 저희에게 카드에 대한 진술을 할 것입니다. 저희는 각 진술의 정보 내용을 평가하려고 노력할 것입니다.

먼저, 그들은 카드를 뒤집고 저희에게 "카드가 보입니다."라고 말합니다. 이는 저희에게 전혀 정보를 제공하지 않습니다. 저희는 이미 이것이 사실이라고 확신했으므로, 정보가 0이어야 하기를 바랍니다.

다음으로, 그들은 카드를 뒤집고 "하트가 보입니다."라고 말합니다. 이는 저희에게 약간의 정보를 제공하지만, 실제로는 가능한 $4$개의 다른 모양만 있고, 각각 동등하게 가능성이 있으므로, 저희는 이 결과에 놀라지 않습니다. 정보의 척도가 무엇이든, 이 사건은 낮은 정보 내용을 가져야 한다고 바랍니다.

다음으로, 그들은 카드를 뒤집고 "이것은 스페이드 $3$입니다."라고 말합니다. 이것은 더 많은 정보입니다. 사실 $52$개의 동등하게 가능성 있는 결과가 있었고, 저희 친구는 그것이 어떤 것인지 알려주었습니다. 이것은 중간 정도의 정보여야 합니다.

이를 논리적인 극단으로 가져갑시다. 마지막으로 그들이 덱의 모든 카드를 뒤집고 섞인 덱의 전체 시퀀스를 읽는다고 가정해 보십시오. 덱에 대한 $52!$개의 다른 순서가 있고, 다시 모두 동등하게 가능성이 있으므로, 저희는 그것이 어떤 것인지 알기 위해 많은 정보가 필요합니다.

저희가 발전시키는 어떤 정보의 개념도 이 직관에 부합해야 합니다. 사실, 다음 절들에서 저희는 이러한 사건이 각각 $0\textrm{ 비트}$, $2\textrm{ 비트}$, $~5.7\textrm{ 비트}$, 그리고 $~225.6\textrm{ 비트}$의 정보를 가지고 있음을 계산하는 방법을 배울 것입니다.

만약 이러한 사고 실험을 통해 읽으면, 저희는 자연스러운 아이디어를 봅니다. 시작점으로, 지식에 신경 쓰는 대신, 저희는 정보가 사건의 놀라움 정도 또는 추상적 가능성을 나타낸다는 아이디어에서 출발할 수 있습니다. 예를 들어, 만약 비정상적인 사건을 설명하고 싶다면, 저희는 많은 정보가 필요합니다. 일반적인 사건의 경우, 많은 정보가 필요하지 않을 수 있습니다.

1948년, 클로드 E. 섀넌은 정보 이론을 수립하는 *통신의 수학적 이론* :cite:`Shannon.1948`을 발표했습니다. 그의 논문에서, 섀넌은 처음으로 정보 엔트로피 개념을 도입했습니다. 저희는 여기서 여정을 시작할 것입니다.


### 자기 정보

정보가 사건의 추상적 가능성을 구체화하므로, 저희는 가능성을 비트 수로 어떻게 매핑할까요? 섀넌은 정보의 단위로 *비트*라는 용어를 도입했는데, 이는 원래 존 튜키에 의해 만들어졌습니다. 그래서 "비트"란 무엇이며 왜 저희는 정보를 측정하기 위해 그것을 사용합니까? 역사적으로, 골동품 송신기는 두 가지 유형의 코드만 보내거나 받을 수 있었습니다. $0$과 $1$. 사실, 이진 인코딩은 여전히 모든 현대 디지털 컴퓨터에서 일반적으로 사용됩니다. 이런 식으로, 어떤 정보든 일련의 $0$과 $1$로 인코딩됩니다. 그리고 따라서, 길이 $n$의 이진 숫자의 시리즈는 $n$ 비트의 정보를 포함합니다.

이제, 어떤 코드 시리즈에 대해서도, 각 $0$ 또는 $1$이 $\frac{1}{2}$의 확률로 발생한다고 가정해 보십시오. 따라서, 길이 $n$의 코드 시리즈를 가진 사건 $X$는 $\frac{1}{2^n}$의 확률로 발생합니다. 동시에, 이전에 언급한 것처럼, 이 시리즈는 $n$ 비트의 정보를 포함합니다. 그래서, 저희는 확률 $p$를 비트 수로 전송할 수 있는 수학 함수로 일반화할 수 있을까요? 섀넌은 *자기 정보*를 정의함으로써 답을 주었습니다.

$$I(X) = - \log_2 (p),$$

이는 이 사건 $X$에 대해 저희가 받은 정보의 *비트*입니다. 이 절에서 저희는 항상 밑 2 로그를 사용할 것이라는 점에 유의하십시오. 단순함을 위해, 이 절의 나머지는 로그 표기법에서 첨자 2를 생략할 것입니다. 즉, $\log(.)$는 항상 $\log_2(.)$를 가리킵니다. 예를 들어, 코드 "0010"은 다음과 같은 자기 정보를 가집니다.

$$I(\textrm{"0010"}) = - \log (p(\textrm{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \textrm{ bits}.$$

저희는 아래와 같이 자기 정보를 계산할 수 있습니다. 그 전에, 먼저 이 절에서 필요한 모든 패키지를 임포트합시다.

```{.python .input}
#@tab mxnet
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # Define nansum, as pytorch does not offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## 엔트로피

자기 정보는 단일 이산 사건의 정보만 측정하므로, 저희는 이산 또는 연속 분포의 어떤 확률 변수에 대해서도 더 일반화된 척도가 필요합니다.


### 엔트로피의 동기 부여

저희가 원하는 것에 대해 구체적으로 알아보려고 합시다. 이것은 *섀넌 엔트로피의 공리*로 알려진 것의 비공식적 진술이 될 것입니다. 다음 상식적 진술의 모음이 저희를 정보의 고유한 정의로 강요한다는 것이 밝혀질 것입니다. 이러한 공리의 형식적 버전과 몇몇 다른 것들은 :citet:`Csiszar.2008`에서 찾을 수 있습니다.

1.  확률 변수를 관찰함으로써 얻는 정보는 저희가 원소를 무엇이라고 부르는지, 또는 확률 0을 가진 추가 원소의 존재에 의존하지 않습니다.
2.  두 확률 변수를 관찰함으로써 얻는 정보는 그것들을 별도로 관찰함으로써 얻는 정보의 합보다 더 많지 않습니다. 만약 그것들이 독립이라면, 정확히 합입니다.
3.  (거의) 확실한 사건을 관찰할 때 얻은 정보는 (거의) 0입니다.

이 사실을 증명하는 것은 저희 텍스트의 범위를 벗어나지만, 이것이 엔트로피가 취해야 하는 형태를 고유하게 결정한다는 것을 아는 것이 중요합니다. 이것들이 허용하는 유일한 모호함은 기본 단위의 선택에 있으며, 이는 단일 공정 동전 던지기에 의해 제공되는 정보가 1비트라는 저희가 이전에 본 선택을 함으로써 가장 자주 정규화됩니다.

### 정의

확률 밀도 함수(p.d.f.) 또는 확률 질량 함수(p.m.f.) $p(x)$를 가진 확률 분포 $P$를 따르는 어떤 확률 변수 $X$에 대해서도, 저희는 *엔트로피*(또는 *섀넌 엔트로피*)를 통해 기대 정보량을 측정합니다.

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`

구체적으로, 만약 $X$가 이산이라면, $$H(X) = - \sum_i p_i \log p_i \textrm{, where } p_i = P(X_i).$$

그렇지 않으면, 만약 $X$가 연속이라면, 저희는 엔트로피를 *미분 엔트로피*라고도 부릅니다.

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

저희는 아래와 같이 엔트로피를 정의할 수 있습니다.

```{.python .input}
#@tab mxnet
def entropy(p):
    entropy = - p * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### 해석

여러분은 궁금할 수 있습니다. 엔트로피 정의 :eqref:`eq_ent_def`에서, 왜 저희는 음의 로그의 기댓값을 사용합니까? 여기 몇 가지 직관이 있습니다.

먼저, 왜 저희는 *로그* 함수 $\log$를 사용합니까? $p(x) = f_1(x) f_2(x) \ldots, f_n(x)$이라고 가정합니다. 여기서 각 성분 함수 $f_i(x)$는 서로 독립입니다. 이는 각 $f_i(x)$가 $p(x)$에서 얻은 총 정보에 독립적으로 기여한다는 것을 의미합니다. 위에서 논의된 것처럼, 저희는 엔트로피 공식이 독립 확률 변수에 대해 가산적이기를 원합니다. 다행히도, $\log$는 확률 분포의 곱을 자연스럽게 개별 항의 합산으로 바꿀 수 있습니다.

다음으로, 왜 저희는 *음의* $\log$를 사용합니까? 직관적으로, 더 자주 발생하는 사건은 덜 일반적인 사건보다 더 적은 정보를 포함해야 합니다. 왜냐하면 저희는 종종 비정상적인 경우에서 일반적인 경우보다 더 많은 정보를 얻기 때문입니다. 그러나, $\log$는 확률과 함께 단조 증가하며, 사실 $[0, 1]$의 모든 값에 대해 음수입니다. 저희는 사건의 확률과 그것들의 엔트로피 사이에 단조 감소 관계를 구성해야 하는데, 이는 이상적으로는 항상 양수여야 합니다(저희가 관찰하는 어떤 것도 저희가 알고 있던 것을 잊게 강요해서는 안 되기 때문입니다). 따라서, 저희는 $\log$ 함수 앞에 음의 부호를 추가합니다.

마지막으로, *기댓값* 함수는 어디에서 옵니까? 확률 변수 $X$를 고려해 보십시오. 저희는 자기 정보($-\log(p)$)를 특정 결과를 볼 때 느끼는 *놀라움*의 양으로 해석할 수 있습니다. 사실, 확률이 0에 접근함에 따라, 놀라움은 무한대가 됩니다. 유사하게, 저희는 엔트로피를 $X$를 관찰함으로부터의 평균 놀라움 양으로 해석할 수 있습니다. 예를 들어, 슬롯 머신 시스템이 각각 확률 ${p_1, \ldots, p_k}$로 통계적으로 독립적인 기호 ${s_1, \ldots, s_k}$를 방출한다고 상상해 보십시오. 그러면 이 시스템의 엔트로피는 각 출력을 관찰하는 것으로부터의 평균 자기 정보와 같습니다. 즉,

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$



### 엔트로피의 속성

위의 예제와 해석에 의해, 저희는 엔트로피 :eqref:`eq_ent_def`의 다음 속성을 유도할 수 있습니다. 여기서, 저희는 $X$를 사건으로, $P$를 $X$의 확률 분포로 가리킵니다.

* 모든 이산 $X$에 대해 $H(X) \geq 0$(연속 $X$에 대해서는 엔트로피가 음수일 수 있습니다).

* 만약 $X \sim P$가 p.d.f. 또는 p.m.f. $p(x)$를 가지고, 저희가 p.d.f. 또는 p.m.f. $q(x)$를 가진 새로운 확률 분포 $Q$로 $P$를 추정하려고 시도한다면, $$H(X) = - E_{x \sim P} [\log p(x)] \leq  - E_{x \sim P} [\log q(x)], \textrm{ with equality if and only if } P = Q.$$  대안적으로, $H(X)$는 $P$에서 추출된 기호를 인코딩하는 데 필요한 평균 비트 수의 하한을 제공합니다.

* 만약 $X \sim P$이면, $x$는 모든 가능한 결과 중에서 균등하게 퍼져 있다면 최대량의 정보를 전달합니다. 구체적으로, 만약 확률 분포 $P$가 $k$ 클래스 $\{p_1, \ldots, p_k \}$로 이산이라면, $$H(X) \leq \log(k), \textrm{ with equality if and only if } p_i = \frac{1}{k}, \forall i.$$ 만약 $P$가 연속 확률 변수라면, 이야기는 훨씬 더 복잡해집니다. 그러나, 만약 $P$가 유한 구간(모든 값이 $0$과 $1$ 사이)에서 지원된다고 추가로 부과하면, $P$는 그 구간에서 균등 분포인 경우에 가장 높은 엔트로피를 가집니다.


## 상호 정보

이전에 저희는 단일 확률 변수 $X$의 엔트로피를 정의했는데, 한 쌍의 확률 변수 $(X, Y)$의 엔트로피는 어떨까요? 저희는 이러한 기법을 다음과 같은 유형의 질문에 답하려고 시도하는 것으로 생각할 수 있습니다. "$X$와 $Y$에 함께 포함된 정보는 각각 별도로 포함된 정보와 비교하여 무엇인가? 중복된 정보가 있는가, 아니면 모두 고유한가?"

다음 논의를 위해, 저희는 항상 p.d.f. 또는 p.m.f. $p_{X, Y}(x, y)$를 가진 결합 확률 분포 $P$를 따르는 한 쌍의 확률 변수로 $(X, Y)$를 사용할 것이며, $X$와 $Y$는 각각 확률 분포 $p_X(x)$와 $p_Y(y)$를 따릅니다.


### 결합 엔트로피

단일 확률 변수의 엔트로피 :eqref:`eq_ent_def`와 유사하게, 저희는 한 쌍의 확률 변수 $(X, Y)$의 *결합 엔트로피* $H(X, Y)$를 다음과 같이 정의합니다.

$$H(X, Y) = -E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

정확히, 한편으로, 만약 $(X, Y)$가 한 쌍의 이산 확률 변수라면

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

다른 한편으로, 만약 $(X, Y)$가 한 쌍의 연속 확률 변수라면, 저희는 *미분 결합 엔트로피*를 다음과 같이 정의합니다.

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

저희는 :eqref:`eq_joint_ent_def`을 한 쌍의 확률 변수에서의 총 무작위성을 알려주는 것으로 생각할 수 있습니다. 한 쌍의 극단으로, 만약 $X = Y$가 두 동일한 확률 변수라면, 쌍의 정보는 정확히 하나의 정보이며 $H(X, Y) = H(X) = H(Y)$를 가집니다. 다른 극단에서는, 만약 $X$와 $Y$가 독립이라면 $H(X, Y) = H(X) + H(Y)$. 사실 저희는 항상 한 쌍의 확률 변수에 포함된 정보가 어느 확률 변수의 엔트로피보다 작지 않고 둘의 합보다 많지 않을 것임을 가질 것입니다.

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

결합 엔트로피를 처음부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

이것은 이전과 같은 *코드*이지만, 이제 저희는 두 확률 변수의 결합 분포에서 작동하는 것으로 다르게 해석합니다.


### 조건부 엔트로피

위에서 정의된 결합 엔트로피는 한 쌍의 확률 변수에 포함된 정보량입니다. 이는 유용하지만, 종종 저희가 신경 쓰는 것이 아닙니다. 머신러닝의 설정을 고려해 보십시오. $X$를 이미지의 픽셀 값을 설명하는 확률 변수(또는 확률 변수의 벡터)로, $Y$를 클래스 레이블인 확률 변수로 취해 봅시다. $X$는 상당한 정보를 포함해야 합니다(자연 이미지는 복잡한 것입니다). 그러나, 이미지가 표시된 후 $Y$에 포함된 정보는 낮아야 합니다. 사실, 숫자의 이미지는 숫자가 읽기 어렵지 않는 한 그것이 어떤 숫자인지에 대한 정보를 이미 포함해야 합니다. 따라서, 정보 이론의 저희 어휘를 계속 확장하기 위해, 저희는 다른 것에 조건부인 확률 변수의 정보 내용에 대해 추론할 수 있어야 합니다.

확률 이론에서, 저희는 변수 사이의 관계를 측정하기 위한 *조건부 확률*의 정의를 보았습니다. 저희는 이제 유사하게 *조건부 엔트로피* $H(Y \mid X)$를 정의하고 싶습니다. 저희는 이를 다음과 같이 쓸 수 있습니다.

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

여기서 $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$는 조건부 확률입니다. 구체적으로, 만약 $(X, Y)$가 한 쌍의 이산 확률 변수라면

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

만약 $(X, Y)$가 한 쌍의 연속 확률 변수라면, *미분 조건부 엔트로피*는 유사하게 정의됩니다.

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$


이제 자연스럽게 묻자면, *조건부 엔트로피* $H(Y \mid X)$가 엔트로피 $H(X)$ 및 결합 엔트로피 $H(X, Y)$와 어떻게 관련되어 있습니까? 위의 정의를 사용하여, 저희는 이를 깔끔하게 표현할 수 있습니다.

$$H(Y \mid X) = H(X, Y) - H(X).$$

이는 직관적인 해석을 가집니다. $X$가 주어진 $Y$의 정보 ($H(Y \mid X)$)는 $X$와 $Y$ 모두에 함께 있는 정보 ($H(X, Y)$)에서 $X$에 이미 포함된 정보를 뺀 것과 같습니다. 이는 저희에게 $X$에 또한 나타내지지 않은 $Y$의 정보를 제공합니다.

이제, 조건부 엔트로피 :eqref:`eq_cond_ent_def`를 처음부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### 상호 정보

확률 변수 $(X, Y)$의 이전 설정이 주어졌을 때, 여러분은 궁금할 수 있습니다. "이제 $X$에는 없지만 $Y$에 얼마나 많은 정보가 포함되어 있는지 알았으므로, 저희는 유사하게 $X$와 $Y$ 사이에 얼마나 많은 정보가 공유되는지 물을 수 있습니까?" 답은 $(X, Y)$의 *상호 정보*가 될 것이며, 저희는 이를 $I(X, Y)$로 쓸 것입니다.

형식적 정의로 곧장 다이빙하기보다, 저희가 이전에 구성한 항을 전적으로 기반으로 한 상호 정보에 대한 식을 먼저 유도하려고 시도함으로써 직관을 연습해 봅시다. 저희는 두 확률 변수 사이에 공유된 정보를 찾고 싶습니다. 저희가 이를 시도할 수 있는 한 가지 방법은 $X$와 $Y$ 모두에 함께 포함된 모든 정보로 시작한 다음, 공유되지 않는 부분을 제거하는 것입니다. $X$와 $Y$ 모두에 함께 포함된 정보는 $H(X, Y)$로 작성됩니다. 저희는 이로부터 $X$에는 있지만 $Y$에는 없는 정보와, $Y$에는 있지만 $X$에는 없는 정보를 빼고 싶습니다. 이전 절에서 본 것처럼, 이는 각각 $H(X \mid Y)$와 $H(Y \mid X)$로 주어집니다. 따라서, 저희는 상호 정보가 다음과 같아야 한다는 것을 가집니다.

$$
I(X, Y) = H(X, Y) - H(Y \mid X) - H(X \mid Y).
$$

사실, 이것은 상호 정보에 대한 유효한 정의입니다. 만약 이러한 항의 정의를 확장하고 결합하면, 약간의 대수가 이것이 다음과 같음을 보여줍니다.

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`


저희는 이러한 모든 관계를 이미지 :numref:`fig_mutual_information`에 요약할 수 있습니다. 다음 진술이 모두 $I(X, Y)$와도 동등한 이유를 보는 것은 직관의 훌륭한 테스트입니다.

* $H(X) - H(X \mid Y)$
* $H(Y) - H(Y \mid X)$
* $H(X) + H(Y) - H(X, Y)$

![결합 엔트로피와 조건부 엔트로피와의 상호 정보의 관계.](../img/mutual-information.svg)
:label:`fig_mutual_information`


여러 면에서 저희는 상호 정보 :eqref:`eq_mut_ent_def`를 :numref:`sec_random_variables`에서 본 상관 계수의 원칙적인 확장으로 생각할 수 있습니다. 이는 저희가 변수 사이의 선형 관계뿐만 아니라, 어떤 종류의 두 확률 변수 사이에 공유된 최대 정보를 묻는 것을 허용합니다.

이제, 상호 정보를 처음부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### 상호 정보의 속성

상호 정보 :eqref:`eq_mut_ent_def`의 정의를 암기하기보다, 저희는 그 주목할 만한 속성만 마음에 새기면 됩니다.

* 상호 정보는 대칭입니다. 즉, $I(X, Y) = I(Y, X)$.
* 상호 정보는 음이 아닙니다. 즉, $I(X, Y) \geq 0$.
* $X$와 $Y$가 독립인 경우에 그리고 오직 그 경우에만 $I(X, Y) = 0$. 예를 들어, 만약 $X$와 $Y$가 독립이라면, $Y$를 아는 것은 $X$에 대한 어떤 정보도 주지 않으며, 그 반대도 마찬가지이므로, 그들의 상호 정보는 0입니다.
* 대안적으로, 만약 $X$가 $Y$의 가역 함수라면, $Y$와 $X$는 모든 정보를 공유하고 $$I(X, Y) = H(Y) = H(X).$$

### 점별 상호 정보

이 장의 시작에서 저희가 엔트로피로 작업할 때, 저희는 $-\log(p_X(x))$를 특정 결과에 저희가 얼마나 *놀랐는지*로 해석을 제공할 수 있었습니다. 저희는 상호 정보의 로그 항에 유사한 해석을 줄 수 있으며, 이는 종종 *점별 상호 정보*라고 합니다.

$$\textrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`

저희는 :eqref:`eq_pmi_def`를 결과 $x$와 $y$의 특정 조합이 독립적인 무작위 결과에 대해 저희가 기대할 것과 비교하여 얼마나 더 또는 덜 가능성이 있는지를 측정하는 것으로 생각할 수 있습니다. 만약 그것이 크고 양수라면, 이 두 특정 결과는 무작위 우연과 비교하여(*참고*: 분모는 두 결과가 독립이었을 확률인 $p_X(x) p_Y(y)$입니다) 훨씬 더 자주 발생하는 반면, 만약 그것이 크고 음수라면 두 결과가 무작위 우연에 의해 저희가 기대할 것보다 훨씬 덜 발생하는 것을 나타냅니다.

이는 저희가 상호 정보 :eqref:`eq_mut_ent_def`를 그것들이 독립이었다면 저희가 기대할 것과 비교하여 두 결과가 함께 발생하는 것을 보고 놀란 평균량으로 해석할 수 있게 합니다.

### 상호 정보의 응용

상호 정보는 그 순수한 정의에서 약간 추상적일 수 있는데, 그것이 머신러닝과 어떻게 관련되어 있습니까? 자연어 처리에서, 가장 어려운 문제 중 하나는 *모호성 해결*, 즉 단어의 의미가 맥락에서 불분명한 문제입니다. 예를 들어, 최근에 뉴스의 한 헤드라인이 "Amazon이 불타고 있다"고 보도했습니다. 여러분은 회사 Amazon이 불에 타는 건물이 있는지, 아니면 Amazon 열대우림이 불타고 있는지 궁금할 수 있습니다.

이 경우, 상호 정보는 저희가 이 모호성을 해결하는 데 도움이 될 수 있습니다. 저희는 먼저 회사 Amazon과 상대적으로 큰 상호 정보를 각각 가진 단어 그룹, 예를 들어 전자상거래, 기술, 그리고 온라인을 찾습니다. 둘째, 저희는 Amazon 열대우림과 상대적으로 큰 상호 정보를 각각 가진 다른 단어 그룹, 예를 들어 비, 숲, 그리고 열대를 찾습니다. "Amazon"의 모호성을 해소해야 할 때, 저희는 어느 그룹이 단어 Amazon의 맥락에서 더 많은 발생을 가지는지 비교할 수 있습니다. 이 경우 기사는 숲을 설명하고 맥락을 분명하게 만들 것입니다.


## 쿨백-라이블러 발산

:numref:`sec_linear-algebra`에서 논의한 것처럼, 저희는 어떤 차원성의 공간에서도 두 점 사이의 거리를 측정하기 위해 노름을 사용할 수 있습니다. 저희는 확률 분포로 유사한 작업을 할 수 있기를 원합니다. 이를 처리하는 많은 방법이 있지만, 정보 이론은 가장 좋은 방법 중 하나를 제공합니다. 저희는 이제 두 분포가 함께 가까운지 아닌지 측정하는 방법을 제공하는 *쿨백-라이블러 (KL) 발산*을 탐색합니다.


### 정의

p.d.f. 또는 p.m.f. $p(x)$를 가진 확률 분포 $P$를 따르는 확률 변수 $X$가 주어졌을 때, 저희는 p.d.f. 또는 p.m.f. $q(x)$를 가진 다른 확률 분포 $Q$로 $P$를 추정합니다. 그러면 $P$와 $Q$ 사이의 *쿨백-라이블러 (KL) 발산*(또는 *상대 엔트로피*)은 다음과 같습니다.

$$D_{\textrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`

점별 상호 정보 :eqref:`eq_pmi_def`와 마찬가지로, 저희는 다시 로그 항에 대한 해석을 제공할 수 있습니다. $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$는 저희가 $Q$에 대해 기대할 것보다 $P$ 하에서 $x$를 훨씬 더 자주 본다면 크고 양수일 것이며, 결과를 기대한 것보다 훨씬 덜 본다면 크고 음수일 것입니다. 이런 식으로, 저희는 이를 저희의 참조 분포로부터 그것을 관찰할 때 얼마나 놀랄지와 비교하여 결과를 관찰하는 데 있어서의 저희의 *상대* 놀라움으로 해석할 수 있습니다.

KL 발산을 처음부터 구현해 봅시다.

```{.python .input}
#@tab mxnet
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### KL 발산 속성

KL 발산 :eqref:`eq_kl_def`의 몇 가지 속성을 살펴봅시다.

* KL 발산은 비대칭입니다. 즉, $$D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P).$$인 $P,Q$가 있습니다.
* KL 발산은 음이 아닙니다. 즉, $$D_{\textrm{KL}}(P\|Q) \geq 0.$$ 등식은 $P = Q$일 때만 성립한다는 점에 유의하십시오.
* 만약 $p(x) > 0$이고 $q(x) = 0$인 $x$가 존재한다면, $D_{\textrm{KL}}(P\|Q) = \infty$.
* KL 발산과 상호 정보 사이에는 밀접한 관계가 있습니다. :numref:`fig_mutual_information`에 표시된 관계 외에도, $I(X, Y)$는 또한 다음 항들과 수치적으로 동등합니다.
    1. $D_{\textrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\textrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\textrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$.

  첫 번째 항의 경우, 저희는 상호 정보를 $P(X, Y)$와 $P(X)$와 $P(Y)$의 곱 사이의 KL 발산으로 해석하며, 따라서 그것들이 독립이었다면 결합 분포가 분포와 얼마나 다른지에 대한 측정값입니다. 두 번째 항의 경우, 상호 정보는 저희에게 $X$의 분포의 값을 학습함으로써 발생하는 $Y$에 대한 불확실성의 평균 감소를 알려줍니다. 세 번째 항과 유사합니다.


### 예제

비대칭성을 명시적으로 보기 위해 장난감 예제를 살펴봅시다.

먼저, 길이 $10,000$의 세 텐서를 생성하고 정렬합시다. 정규 분포 $N(0, 1)$을 따르는 목적 텐서 $p$, 그리고 각각 정규 분포 $N(-1, 1)$과 $N(1, 1)$을 따르는 두 후보 텐서 $q_1$과 $q_2$.

```{.python .input}
#@tab mxnet
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

$q_1$과 $q_2$가 y축(즉, $x=0$)에 대해 대칭이므로, 저희는 $D_{\textrm{KL}}(p\|q_1)$과 $D_{\textrm{KL}}(p\|q_2)$ 사이의 유사한 KL 발산 값을 기대합니다. 아래에서 볼 수 있듯이, $D_{\textrm{KL}}(p\|q_1)$과 $D_{\textrm{KL}}(p\|q_2)$ 사이에는 3% 미만의 차이만 있습니다.

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

대조적으로, 여러분은 $D_{\textrm{KL}}(q_2 \|p)$와 $D_{\textrm{KL}}(p \| q_2)$가 아래에 보여진 것처럼 약 40% 정도 많이 차이가 난다는 것을 발견할 수 있습니다.

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## 교차 엔트로피

만약 딥러닝에서 정보 이론의 응용에 대해 궁금하다면, 여기 빠른 예제가 있습니다. 저희는 확률 분포 $p(x)$를 가진 참 분포 $P$와 확률 분포 $q(x)$를 가진 추정 분포 $Q$를 정의하고, 이 절의 나머지에서 그것들을 사용할 것입니다.

저희가 주어진 $n$개의 데이터 예제 {$x_1, \ldots, x_n$}에 기반하여 이진 분류 문제를 풀어야 한다고 합시다. $1$과 $0$을 각각 양과 음의 클래스 레이블 $y_i$로 인코딩한다고 가정하고, 저희의 신경망이 $\theta$로 매개변수화되어 있다고 가정합니다. 만약 $\hat{y}_i= p_{\theta}(y_i \mid x_i)$가 되도록 가장 좋은 $\theta$를 찾는 것을 목표로 한다면, :numref:`sec_maximum_likelihood`에서 본 것처럼 최대 로그 가능도 접근법을 적용하는 것이 자연스럽습니다. 구체적으로, 참 레이블 $y_i$와 예측 $\hat{y}_i= p_{\theta}(y_i \mid x_i)$에 대해, 양으로 분류될 확률은 $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$입니다. 따라서, 로그 가능도 함수는 다음과 같을 것입니다.

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$

로그 가능도 함수 $l(\theta)$를 최대화하는 것은 $- l(\theta)$를 최소화하는 것과 동일하며, 따라서 저희는 여기서 가장 좋은 $\theta$를 찾을 수 있습니다. 위 손실을 어떤 분포로 일반화하기 위해, 저희는 $-l(\theta)$를 *교차 엔트로피 손실* $\textrm{CE}(y, \hat{y})$라고도 불렀는데, 여기서 $y$는 참 분포 $P$를 따르고 $\hat{y}$는 추정 분포 $Q$를 따릅니다.

이 모든 것은 최대 가능도 관점에서 작업함으로써 유도되었습니다. 그러나, 자세히 살펴보면 $\log(\pi_i)$와 같은 항이 저희의 계산에 들어왔음을 볼 수 있는데, 이는 저희가 정보 이론적 관점에서 식을 이해할 수 있다는 확실한 표시입니다.


### 형식적 정의

KL 발산처럼, 확률 변수 $X$에 대해, 저희는 또한 *교차 엔트로피*를 통해 추정 분포 $Q$와 참 분포 $P$ 사이의 발산을 측정할 수 있습니다.

$$\textrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

위에서 논의된 엔트로피의 속성을 사용함으로써, 저희는 또한 이를 엔트로피 $H(P)$와 $P$와 $Q$ 사이의 KL 발산의 합산으로 해석할 수 있습니다. 즉,

$$\textrm{CE} (P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).$$


저희는 아래와 같이 교차 엔트로피 손실을 구현할 수 있습니다.

```{.python .input}
#@tab mxnet
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    # `tf.gather_nd` is used to select specific indices of a tensor.
    ce = -tf.math.log(tf.gather_nd(y_hat, indices = [[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return tf.reduce_mean(ce).numpy()
```

이제 레이블과 예측에 대한 두 텐서를 정의하고, 그것들의 교차 엔트로피 손실을 계산합시다.

```{.python .input}
#@tab mxnet
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### 속성

이 절의 시작에서 암시한 것처럼, 교차 엔트로피 :eqref:`eq_ce_def`는 최적화 문제에서 손실 함수를 정의하는 데 사용될 수 있습니다. 다음이 동등하다는 것이 밝혀집니다.

1. 분포 $P$에 대한 $Q$의 예측 확률을 최대화하는 것, (즉, $E_{x
\sim P} [\log (q(x))]$);
1. 교차 엔트로피 $\textrm{CE} (P, Q)$를 최소화하는 것;
1. KL 발산 $D_{\textrm{KL}}(P\|Q)$를 최소화하는 것.

교차 엔트로피의 정의는 참 데이터의 엔트로피 $H(P)$가 상수인 한, 목적 2와 목적 3 사이의 동등한 관계를 간접적으로 증명합니다.


### 다중 클래스 분류의 목적 함수로서의 교차 엔트로피

만약 교차 엔트로피 손실 $\textrm{CE}$를 가진 분류 목적 함수에 깊이 들어가면, 저희는 $\textrm{CE}$를 최소화하는 것이 로그 가능도 함수 $L$을 최대화하는 것과 동등함을 발견할 것입니다.

시작하기 위해, $n$개의 예제를 가진 데이터셋이 주어졌고, $k$ 클래스로 분류될 수 있다고 가정해 봅시다. 각 데이터 예제 $i$에 대해, 저희는 어떤 $k$ 클래스 레이블 $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$를 *원핫 인코딩*으로 나타냅니다. 구체적으로, 만약 예제 $i$가 클래스 $j$에 속한다면, 저희는 $j$번째 항목을 $1$로, 다른 모든 성분을 $0$으로 설정합니다. 즉,

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\textrm{otherwise.}\end{cases}$$

예를 들어, 만약 다중 클래스 분류 문제가 세 클래스 $A$, $B$, 그리고 $C$를 포함한다면, 레이블 $\mathbf{y}_i$는 {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$}으로 인코딩될 수 있습니다.


저희의 신경망이 $\theta$로 매개변수화되어 있다고 가정합니다. 참 레이블 벡터 $\mathbf{y}_i$와 예측 $$\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$$에 대해.

따라서, *교차 엔트로피 손실*은 다음과 같을 것입니다.

$$
\textrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

다른 한편으로, 저희는 또한 최대 가능도 추정을 통해 문제에 접근할 수 있습니다. 시작하기 위해, $k$ 클래스 다항분포를 빠르게 소개합시다. 이는 베르누이 분포를 이진 클래스에서 다중 클래스로 확장한 것입니다. 만약 확률 변수 $\mathbf{z} = (z_{1}, \ldots, z_{k})$가 확률 $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$)을 가진 $k$ 클래스 *다항분포*를 따른다면, 즉, $$p(\mathbf{z}) = p(z_1, \ldots, z_k) = \textrm{Multi} (p_1, \ldots, p_k), \textrm{ where } \sum_{i=1}^k p_i = 1,$$ 이면, $\mathbf{z}$의 결합 확률 질량 함수(p.m.f.)는 다음과 같습니다.
$$\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$$


각 데이터 예제 $\mathbf{y}_i$의 레이블이 확률 $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$)을 가진 $k$ 클래스 다항분포를 따르고 있음을 볼 수 있습니다. 따라서, 각 데이터 예제 $\mathbf{y}_i$의 결합 p.m.f.는 $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$
따라서, 로그 가능도 함수는 다음과 같을 것입니다.

$$
\begin{aligned}
l(\theta)
 = \log L(\theta)
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$

최대 가능도 추정에서, 저희는 $\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$를 가짐으로써 목적 함수 $l(\theta)$를 최대화하고 있습니다. 따라서, 어떤 다중 클래스 분류에 대해서도, 위의 로그 가능도 함수 $l(\theta)$를 최대화하는 것은 CE 손실 $\textrm{CE}(y, \hat{y})$를 최소화하는 것과 동등합니다.


위 증명을 테스트하기 위해, 내장된 척도 `NegativeLogLikelihood`를 적용해 봅시다. 이전 예제와 동일한 `labels`와 `preds`를 사용하면, 저희는 이전 예제와 소수점 5자리까지 같은 수치적 손실을 얻을 것입니다.

```{.python .input}
#@tab mxnet
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# Implementation of cross-entropy loss in PyTorch combines `nn.LogSoftmax()`
# and `nn.NLLLoss()`
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # Convert labels to one-hot vectors.
    y = tf.keras.utils.to_categorical(y, num_classes= y_hat.shape[1])
    # We will not calculate negative log-likelihood from the definition.
    # Rather, we will follow a circular argument. Because NLL is same as
    # `cross_entropy`, if we calculate cross_entropy that would give us NLL
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(cross_entropy(y, y_hat)).numpy()

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## 요약

* 정보 이론은 정보의 인코딩, 디코딩, 전송, 그리고 조작에 대한 연구 분야입니다.
* 엔트로피는 다른 신호에 얼마나 많은 정보가 제시되어 있는지 측정하는 단위입니다.
* KL 발산은 또한 두 분포 사이의 발산을 측정할 수 있습니다.
* 교차 엔트로피는 다중 클래스 분류의 목적 함수로 볼 수 있습니다. 교차 엔트로피 손실을 최소화하는 것은 로그 가능도 함수를 최대화하는 것과 동등합니다.


## 연습문제

1. 첫 번째 절의 카드 예제가 실제로 주장된 엔트로피를 가지는지 확인하십시오.
1. KL 발산 $D(p\|q)$가 모든 분포 $p$와 $q$에 대해 음이 아님을 보이십시오. 힌트: 옌센의 부등식을 사용하십시오. 즉, $-\log x$가 볼록 함수라는 사실을 사용하십시오.
1. 몇 가지 데이터 소스로부터 엔트로피를 계산해 봅시다.
    * 타자기에서 원숭이가 생성하는 출력을 보고 있다고 가정해 봅시다. 원숭이는 타자기의 $44$개 키 중 어느 것이든 무작위로 누릅니다(아직 특수 키나 시프트 키를 발견하지 못했다고 가정할 수 있습니다). 문자당 무작위성의 몇 비트를 관찰합니까?
    * 원숭이에 만족하지 않아서, 당신은 그것을 술취한 식자공으로 대체했습니다. 그것은 일관성은 없지만 단어를 생성할 수 있습니다. 대신, 그것은 $2,000$ 단어의 어휘에서 무작위 단어를 선택합니다. 영어에서 단어의 평균 길이가 $4.5$ 글자라고 가정합시다. 이제 문자당 무작위성의 몇 비트를 관찰합니까?
    * 여전히 결과에 만족하지 않아서, 당신은 식자공을 고품질 언어 모델로 대체합니다. 언어 모델은 현재 단어당 $15$ 점으로 낮은 퍼플렉시티를 얻을 수 있습니다. 언어 모델의 문자 *퍼플렉시티*는 확률 집합의 기하 평균의 역으로 정의되며, 각 확률은 단어의 문자에 해당합니다. 구체적으로, 만약 주어진 단어의 길이가 $l$이라면, $\textrm{PPL}(\textrm{word}) = \left[\prod_i p(\textrm{character}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\textrm{character}_i)} \right].$  테스트 단어가 4.5 글자라고 가정하면, 이제 문자당 무작위성의 몇 비트를 관찰합니까?
1. $I(X, Y) = H(X) - H(X \mid Y)$인 이유를 직관적으로 설명하십시오. 그런 다음, 결합 분포에 대한 기댓값으로 양변을 표현함으로써 이것이 참임을 보이십시오.
1. 두 가우시안 분포 $\mathcal{N}(\mu_1, \sigma_1^2)$와 $\mathcal{N}(\mu_2, \sigma_2^2)$ 사이의 KL 발산은 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1105)
:end_tab:
