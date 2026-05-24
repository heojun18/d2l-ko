# 시퀀스 다루기
:label:`sec_sequence`

지금까지 저희는 입력이 단일 특징 벡터 $\mathbf{x} \in \mathbb{R}^d$로
구성된 모델에 초점을 맞추어 왔습니다.
시퀀스를 처리할 수 있는 모델을 개발할 때 관점의 가장 큰 변화는,
이제 저희가 특징 벡터들의 순서 있는 목록
$\mathbf{x}_1, \dots, \mathbf{x}_T$로 이루어진 입력에 초점을 맞춘다는 점입니다.
여기서 각 특징 벡터 $\mathbf{x}_t$는
$\mathbb{R}^d$에 속하며
타임스텝 $t \in \mathbb{Z}^+$로 인덱싱됩니다.

일부 데이터셋은 거대한 단일 시퀀스로 구성됩니다.
예를 들어, 기후 과학자들에게 제공될 수 있는
극도로 긴 센서 측정값 스트림을 생각해 보세요.
이러한 경우, 저희는 미리 정해진 길이의 부분 시퀀스를
무작위로 샘플링하여 학습 데이터셋을 만들 수 있습니다.
더 흔하게는, 저희의 데이터가 시퀀스들의 모음으로 도착합니다.
다음 예시들을 생각해 봅시다.
(i) 문서들의 모음. 각 문서는 자신만의 단어 시퀀스로 표현되고,
각각 자신만의 길이 $T_i$를 가집니다.
(ii) 병원 입원 환자들의 시퀀스 표현.
여기서 각 입원은 여러 사건으로 구성되며,
시퀀스 길이는 대략 입원 기간에 따라 달라집니다.


이전에 개별 입력을 다룰 때, 저희는
그 입력들이 동일한 기저 분포 $P(X)$에서
독립적으로 샘플링된다고 가정했습니다.
저희는 여전히 전체 시퀀스
(예: 전체 문서나 환자 궤적)가 독립적으로 샘플링된다고
가정하지만,
각 타임스텝에서 도착하는 데이터들이
서로 독립적이라고는 가정할 수 없습니다.
예를 들어, 문서 뒷부분에 등장할 가능성이 있는 단어는
그 문서 앞부분에 등장한 단어에 크게 좌우됩니다.
환자가 병원 방문 10일째에 받을 가능성이 있는 약은
이전 9일 동안 일어난 일에 크게 좌우됩니다.

이는 놀라운 일이 아닐 것입니다.
시퀀스의 요소들이 관련되어 있다고 믿지 않았다면,
애초에 그것들을 시퀀스로 모델링할 이유가 없었을 것입니다.
검색 도구나 현대 이메일 클라이언트에서 인기 있는
자동 완성 기능의 유용성을 생각해 보세요.
이것들이 유용한 이유는 정확히, 어떤 초기 접두사가 주어졌을 때
시퀀스가 어떻게 이어질지를 예측하는 것이
(불완전하지만, 무작위 추측보다는 더 잘) 종종 가능하기 때문입니다.
대부분의 시퀀스 모델에서,
저희는 시퀀스의 독립성이나
정상성(stationarity)조차도 요구하지 않습니다.
대신, 저희가 요구하는 것은 오직
시퀀스 자체가 전체 시퀀스에 대한
어떤 고정된 기저 분포에서 샘플링된다는 점뿐입니다.

이 유연한 접근 방식은 다음과 같은 현상을 허용합니다.
(i) 문서가 처음과 끝에서 상당히 다르게 보이는 경우,
(ii) 환자 상태가 입원 기간 동안
회복 또는 사망 쪽으로 진행되는 경우,
(iii) 추천 시스템과의 지속적인 상호작용 과정에서
고객의 취향이 예측 가능한 방식으로 변화하는 경우입니다.


때때로 저희는 순차적으로 구조화된 입력이 주어졌을 때
고정된 타깃 $y$를 예측하고자 합니다
(예: 영화 리뷰를 바탕으로 한 감성 분류).
또 다른 경우에는, 고정된 입력이 주어졌을 때
순차적으로 구조화된 타깃 ($y_1, \ldots, y_T$)을
예측하고자 합니다 (예: 이미지 캡셔닝).
또 다른 경우에는, 순차적으로 구조화된 입력을 바탕으로
순차적으로 구조화된 타깃을 예측하는 것이 목표입니다
(예: 기계 번역 또는 비디오 캡셔닝).
이러한 시퀀스 대 시퀀스 과제는 두 가지 형태를 띕니다.
(i) *정렬됨(aligned)*: 각 타임스텝에서의 입력이
해당하는 타깃과 정렬되는 경우 (예: 품사 태깅).
(ii) *비정렬됨(unaligned)*: 입력과 타깃이
반드시 한 스텝 한 스텝 대응을 보이지는 않는 경우
(예: 기계 번역).

어떤 종류의 타깃을 다루는 것을 걱정하기 전에,
가장 단순한 문제부터 다룰 수 있습니다.
바로 비지도 밀도 모델링(*시퀀스 모델링*이라고도 합니다)입니다.
여기서 시퀀스들의 모음이 주어졌을 때,
저희의 목표는 어떤 주어진 시퀀스를 볼 확률이 얼마인지,
즉 $p(\mathbf{x}_1, \ldots, \mathbf{x}_T)$를 알려주는
확률 질량 함수를 추정하는 것입니다.

```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=9}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

## 자기회귀 모델 (Autoregressive Models)


순차적으로 구조화된 데이터를 다루도록 설계된
전문 신경망을 소개하기 전에,
실제 시퀀스 데이터를 살펴보고
기본적인 직관과 통계 도구를 쌓아 봅시다.
특히 저희는 FTSE 100 지수의 주가 데이터에
초점을 맞출 것입니다 (:numref:`fig_ftse100`).
각 *타임스텝* $t \in \mathbb{Z}^+$에서,
저희는 그 시점의 지수 가격 $x_t$를 관측합니다.


![약 30년에 걸친 FTSE 100 지수.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`


이제 한 트레이더가 단기 매매를 하고 싶어 한다고 가정해 봅시다.
지수가 다음 타임스텝에 상승할지 하락할지에 대한
자신의 판단에 따라
전략적으로 지수에 들어가거나 빠져나오려고 합니다.
다른 어떤 특징
(뉴스, 재무 보고 데이터 등)도 없을 때,
다음 값을 예측하는 데 사용할 수 있는 유일한 신호는
지금까지의 가격 이력입니다.
따라서 트레이더는 다음 타임스텝에 지수가 가질 수 있는
가격에 대한 확률 분포

$$P(x_t \mid x_{t-1}, \ldots, x_1)$$

를 알고 싶어 합니다.
연속값을 가지는 확률 변수에 대한
전체 분포를 추정하는 것은 어려울 수 있지만,
트레이더는 분포의 몇 가지 핵심 통계량,
특히 기댓값과 분산에 초점을 맞추는 것으로 만족할 것입니다.
조건부 기댓값

$$\mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)],$$

을 추정하는 한 가지 간단한 전략은
선형 회귀 모델을 적용하는 것입니다
(:numref:`sec_linear_regression`을 떠올려 보세요).
한 신호의 값을 그 같은 신호의 이전 값들로 회귀하는
이러한 모델은 자연스럽게 *자기회귀 모델(autoregressive models)* 이라고 불립니다.
한 가지 큰 문제가 있습니다. 입력의 개수
$x_{t-1}, \ldots, x_1$이 $t$에 따라 달라진다는 것입니다.
다시 말해, 입력의 개수는 저희가 마주치는
데이터의 양에 따라 증가합니다.
따라서 과거 데이터를 학습 세트로 다루고자 한다면,
각 예시가 서로 다른 개수의 특징을 가진다는
문제에 부딪힙니다.
이 장에서 이어지는 내용의 상당 부분은,
관심 대상이 $P(x_t \mid x_{t-1}, \ldots, x_1)$
또는 이 분포의 어떤 통계량인
이러한 *자기회귀* 모델링 문제에 임할 때
이러한 어려움을 극복하기 위한 기법들에 관한 것입니다.

몇 가지 전략이 자주 반복됩니다.
우선, 긴 시퀀스
$x_{t-1}, \ldots, x_1$이 사용 가능하다고 하더라도,
가까운 미래를 예측할 때
이력을 그렇게 멀리까지 거슬러 살펴볼
필요는 없다고 믿을 수도 있습니다.
이 경우에는 길이 $\tau$의 어떤 윈도우로 조건화하고
$x_{t-1}, \ldots, x_{t-\tau}$ 관측값만 사용하는 것에
만족할 수 있습니다.
즉각적인 이점은 이제 인수의 개수가
적어도 $t > \tau$에 대해서는 항상 같다는 점입니다.
이를 통해 저희는 입력으로 고정 길이 벡터가 필요한
어떤 선형 모델이나 심층 신경망도 학습할 수 있습니다.
둘째, 과거 관측값의 어떤 요약 $h_t$를 유지하고
(:numref:`fig_sequence-model` 참고)
동시에 예측 $\hat{x}_t$와 더불어
$h_t$를 업데이트하는 모델을 개발할 수도 있습니다.
이로 인해 $\hat{x}_t = P(x_t \mid h_{t})$로 $x_t$를 추정할 뿐만 아니라
$h_t = g(h_{t-1}, x_{t-1})$ 형태의 업데이트도 함께 하는
모델이 만들어집니다.
$h_t$가 결코 관측되지 않기 때문에,
이러한 모델은 *잠재 자기회귀 모델(latent autoregressive models)* 이라고도 불립니다.

![잠재 자기회귀 모델.](../img/sequence-model.svg)
:label:`fig_sequence-model`

과거 데이터로부터 학습 데이터를 구성하기 위해,
일반적으로 윈도우를 무작위로 샘플링하여 예시를 만듭니다.
일반적으로 저희는 시간이 멈춰 있을 것이라고는 기대하지 않습니다.
그러나 $x_t$의 구체적인 값은 변할 수 있어도,
이전 관측값이 주어졌을 때 각 후속 관측값을 생성하는
동역학은 변하지 않는다고 종종 가정합니다.
통계학자들은 변하지 않는 동역학을 *정상(stationary)* 이라고 부릅니다.



## 시퀀스 모델 (Sequence Models)

때때로, 특히 언어를 다룰 때,
저희는 전체 시퀀스의 결합 확률을
추정하고자 합니다.
이는 단어와 같은 이산 *토큰*으로 구성된
시퀀스를 다룰 때 흔한 과제입니다.
일반적으로 이렇게 추정된 함수를 *시퀀스 모델(sequence models)* 이라고 부르며,
자연어 데이터에 대해서는 *언어 모델(language models)* 이라고 부릅니다.
시퀀스 모델링 분야가 자연어 처리에 의해 매우 강하게 이끌려 와서,
저희는 비언어 데이터를 다루는 경우에도
종종 시퀀스 모델을 "언어 모델"이라고 부릅니다.
언어 모델은 온갖 이유로 유용함이 입증되어 있습니다.
때때로 저희는 문장의 가능도를 평가하고 싶어 합니다.
예를 들어, 기계 번역 시스템이나 음성 인식 시스템이
생성한 두 후보 출력의 자연스러움을 비교하고
싶을 수도 있습니다.
그러나 언어 모델링은 가능도를 *평가*하는 능력뿐만 아니라,
시퀀스를 *샘플링*하는 능력,
나아가 가장 가능도가 높은 시퀀스를 위한 최적화 능력까지
저희에게 제공합니다.

언어 모델링이 언뜻 보기에는 자기회귀 문제처럼
보이지 않을 수 있지만,
시퀀스의 결합 밀도 $p(x_1, \ldots, x_T)$를
확률의 연쇄 법칙을 적용하여
좌에서 우 방향으로 조건부 밀도의 곱으로 분해함으로써
언어 모델링을 자기회귀 예측으로 환원할 수 있습니다.

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

단어와 같은 이산 신호를 다루는 경우,
자기회귀 모델은 확률적 분류기여야 하며,
주어진 좌측 문맥에서 다음에 올 단어가 무엇이든 그에 대해
어휘 전체에 걸친 완전한 확률 분포를
출력해야 한다는 점에 유의하세요.



### 마르코프 모델 (Markov Models)
:label:`subsec_markov-models`


이제 위에서 언급한 전략, 즉 전체 시퀀스 이력
$x_{t-1}, \ldots, x_1$ 대신
이전 $\tau$개의 타임스텝
$x_{t-1}, \ldots, x_{t-\tau}$만으로 조건화하는 전략을
사용하고 싶다고 가정해 봅시다.
이전 $\tau$ 스텝 너머의 이력을 예측력의 손실 없이
버릴 수 있을 때마다,
저희는 그 시퀀스가 *마르코프 조건(Markov condition)* 을 만족한다고 말합니다.
즉, *최근 이력이 주어졌을 때 미래가 과거에 대해 조건부 독립* 이라는 의미입니다.
$\tau = 1$일 때, 저희는 데이터가
*1차 마르코프 모델(first-order Markov model)* 로 특징지어진다고 말하며,
$\tau = k$일 때, 데이터가
$k^{\textrm{th}}$차 마르코프 모델로 특징지어진다고 말합니다.
1차 마르코프 조건이 성립할 때 ($\tau = 1$),
저희 결합 확률의 인수분해는 이전 *단어*가 주어졌을 때
각 단어의 확률들의 곱이 됩니다.

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$

저희는 마르코프 조건이 *근사적으로만* 참임을 알 때조차도,
그것이 만족되는 것처럼 진행되는 모델을 가지고 작업하는 것이
유용하다고 자주 느낍니다.
실제 텍스트 문서에서는 더 많은 좌측 문맥을 포함할수록
저희는 계속해서 정보를 얻습니다.
그러나 이러한 이득은 빠르게 줄어듭니다.
따라서 때때로 저희는 $k^{\textrm{th}}$차 마르코프 조건에
유효성이 의존하는 모델을 학습하여,
계산상 및 통계상의 어려움을 회피하면서 타협합니다.
오늘날의 거대한 RNN 및 Transformer 기반 언어 모델조차도
수천 단어를 넘는 문맥을 포함하는 경우는 드뭅니다.


이산 데이터에서 진정한 마르코프 모델은
단순히 각 단어가 각 문맥에서 발생한 횟수를 세어,
$P(x_t \mid x_{t-1})$의 상대 빈도 추정값을 만들어 냅니다.
(언어에서처럼) 데이터가 이산값만을 갖는 경우에는 언제든,
가장 가능도 높은 단어 시퀀스를 동적 계획법을 사용하여
효율적으로 계산할 수 있습니다.


### 디코딩의 순서

저희가 왜 텍스트 시퀀스의 인수분해 $P(x_1, \ldots, x_T)$를
좌에서 우로 진행하는 조건부 확률의 사슬로 표현했는지
궁금하실 수 있습니다.
왜 우에서 좌나 어떤 다른, 겉보기에 무작위인 순서는 아닌가요?
원칙적으로 $P(x_1, \ldots, x_T)$를 역순으로
펼치는 데 잘못된 점은 없습니다.
그 결과는 유효한 인수분해입니다.

$$P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$


그러나 언어 모델링 과제에서는 텍스트를
저희가 읽는 방향
(대부분의 언어에서 좌에서 우, 그러나 아랍어와 히브리어에서는 우에서 좌)과
같은 방향으로 인수분해하는 것이 선호되는 데에는 많은 이유가 있습니다.
첫째, 이는 단순히 저희가 생각하기에 더 자연스러운 방향이기 때문입니다.
결국 저희 모두 매일 텍스트를 읽고,
이 과정은 어떤 단어와 구가 다음에 올 가능성이 있는지
예측하는 능력에 의해 이끌어집니다.
다른 사람의 문장을 얼마나 많이 완성해 보셨는지 생각해 보세요.
따라서 그러한 순방향 디코딩을 선호할 다른 이유가 없다고 하더라도,
저희가 이 순서로 예측할 때 무엇이 가능성 있는지에 대한
더 나은 직관을 가지고 있다는 점만으로도 그것들은 유용할 것입니다.

둘째, 순서대로 인수분해함으로써,
저희는 동일한 언어 모델을 사용하여
임의로 긴 시퀀스에 확률을 할당할 수 있습니다.
스텝 $1$부터 $t$에 대한 확률을 단어 $t+1$까지 확장된 확률로
변환하기 위해, 저희는 단순히 이전 토큰들이 주어졌을 때
추가 토큰의 조건부 확률을 곱합니다.
$P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1)$.

셋째, 저희는 임의의 다른 위치에 있는 단어를 예측하는 것보다
인접한 단어를 예측하기 위한 더 강력한 예측 모델을 가지고 있습니다.
인수분해의 모든 순서가 유효하지만,
그것들이 반드시 모두 동일하게 쉬운
예측 모델링 문제를 나타내는 것은 아닙니다.
이는 언어뿐만 아니라 다른 종류의 데이터에서도 참인데,
예를 들어 데이터가 인과적으로 구조화되어 있을 때 그렇습니다.
예를 들어 저희는 미래의 사건이 과거에 영향을 줄 수 없다고 믿습니다.
따라서 $x_t$를 바꾼다면, 저희는 앞으로 $x_{t+1}$에 일어나는 일에는
영향을 줄 수 있지만 그 반대는 아닙니다.
즉, $x_t$를 바꾼다면, 과거 사건에 대한 분포는 변하지 않을 것입니다.
어떤 맥락에서는, 이것이 $P(x_t \mid x_{t+1})$을 예측하는 것보다
$P(x_{t+1} \mid x_t)$을 예측하기를 더 쉽게 만듭니다.
예를 들어, 어떤 경우에는 어떤 가산 잡음 $\epsilon$에 대해
$x_{t+1} = f(x_t) + \epsilon$임을 찾을 수 있는 반면,
그 반대는 참이 아닙니다 :cite:`Hoyer.Janzing.Mooij.ea.2009`.
이는 좋은 소식인데, 일반적으로 저희가 추정에 관심을 두는 것이
바로 순방향이기 때문입니다.
:citet:`Peters.Janzing.Scholkopf.2017`의 책에 이 주제에 관한
더 많은 내용이 담겨 있습니다.
저희는 이 주제의 표면만 겨우 긁었을 뿐입니다.


## 학습

저희의 관심을 텍스트 데이터에 집중시키기 전에,
먼저 연속값을 가지는 합성 데이터를 가지고
이를 시험해 봅시다.

(**여기서 저희의 1000개 합성 데이터는
타임스텝의 0.01배에 적용된
삼각함수 `sin` 함수를 따를 것입니다.
문제를 조금 더 흥미롭게 만들기 위해,
저희는 각 샘플을 가산 잡음으로 오염시킵니다.**)
이 시퀀스로부터 저희는 학습 예시를 추출하며,
각각은 특징과 레이블로 구성됩니다.

```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2
        if tab.selected('jax'):
            key = d2l.get_key()
            self.x = d2l.sin(0.01 * self.time) + jax.random.normal(key,
                                                                   [T]) * 0.2
```

```{.python .input}
%%tab all
data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

먼저, 데이터가 $\tau^{\textrm{th}}$차 마르코프 조건을 만족하는 것처럼 동작하는,
따라서 과거 $\tau$개의 관측값만을 사용하여 $x_t$를 예측하는
모델을 시도해 봅시다.
[**따라서 각 타임스텝마다 저희는 레이블 $y = x_t$와 특징
$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$을 가지는 예시를 갖습니다.**]
예리한 독자라면 이로 인해 $1000-\tau$개의 예시가 생긴다는 점을
알아채셨을 수도 있습니다. 왜냐하면 $y_1, \ldots, y_\tau$에 대해서는
충분한 이력이 부족하기 때문입니다.
첫 $\tau$개의 시퀀스를 0으로 패딩할 수도 있지만,
일을 간단하게 유지하기 위해 지금은 그것들을 버리겠습니다.
결과 데이터셋은 $T - \tau$개의 예시를 포함하며,
각 모델 입력의 시퀀스 길이는 $\tau$입니다.
저희는 sin 함수의 한 주기를 다루는
(**첫 600개의 예시에 대한 데이터 이터레이터를 만듭니다**).

```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

이 예제에서 저희 모델은 표준 선형 회귀가 될 것입니다.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## 예측

[**저희 모델을 평가하기 위해, 먼저 모델이
1스텝-앞 예측에서 얼마나 잘 동작하는지 확인합니다**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

```{.python .input}
%%tab jax
onestep_preds = model.apply({'params': trainer.state.params}, data.features)
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

이러한 예측은 $t=1000$ 근처의 끝부분에서조차
좋아 보입니다.

그러나 저희가 타임스텝 604 (`n_train + tau`)까지만
시퀀스 데이터를 관측했고 여러 스텝 미래로
예측을 하고자 한다면 어떻게 될까요?
안타깝게도 저희는 타임스텝 609에 대한
1스텝-앞 예측을 직접 계산할 수 없습니다.
왜냐하면 $x_{604}$까지만 본 상태에서,
해당하는 입력을 알지 못하기 때문입니다.
저희는 이 문제를 이전의 예측을 후속 예측을 위한
모델의 입력으로 대입하여 다룰 수 있는데,
원하는 타임스텝에 도달할 때까지 한 번에 한 스텝씩
앞으로 투영해 나갑니다.

$$\begin{aligned}
\hat{x}_{605} &= f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} &= f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} &= f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} &= f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} &= f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
&\vdots\end{aligned}$$

일반적으로, 관측된 시퀀스 $x_1, \ldots, x_t$에 대해,
타임스텝 $t+k$에서의 예측된 출력 $\hat{x}_{t+k}$를
$k$*-스텝-앞 예측($k$-step-ahead prediction)* 이라고 합니다.
$x_{604}$까지 관측했으므로,
그것의 $k$-스텝-앞 예측은 $\hat{x}_{604+k}$입니다.
다시 말해, 저희는 다중 스텝 앞 예측을 하기 위해
저희 자신의 예측을 계속 사용해야 할 것입니다.
이것이 얼마나 잘 되는지 봅시다.

```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab jax
multistep_preds = d2l.zeros(data.T)
multistep_preds = multistep_preds.at[:].set(data.x)
for i in range(data.num_train + data.tau, data.T):
    pred = model.apply({'params': trainer.state.params},
                       d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
    multistep_preds = multistep_preds.at[i].set(pred.item())
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

안타깝게도, 이 경우 저희는 보기 좋게 실패합니다.
예측은 몇 스텝이 지난 후 꽤 빠르게
상수로 감쇠합니다.
왜 알고리즘이 미래로 더 멀리 예측할 때
훨씬 더 나쁘게 동작했을까요?
궁극적으로, 이는 오차가 누적된다는
사실로 귀결됩니다.
스텝 1 이후에 어떤 오차 $\epsilon_1 = \bar\epsilon$이 있다고 해 봅시다.
이제 스텝 2의 *입력*이 $\epsilon_1$만큼 교란되고,
따라서 어떤 상수 $c$에 대해
$\epsilon_2 = \bar\epsilon + c \epsilon_1$ 정도의 오차를 겪고,
이런 식으로 계속됩니다.
예측은 실제 관측값으로부터 빠르게 발산할 수 있습니다.
이 흔한 현상이 이미 익숙하실 수도 있습니다.
예를 들어 다음 24시간 동안의 일기 예보는
꽤 정확한 편이지만 그 이후로는
정확도가 빠르게 떨어집니다.
저희는 이 장 전체와 그 이후에서
이를 개선하기 위한 방법을 논의할 것입니다.

$k = 1, 4, 16, 64$에 대해 전체 시퀀스에 대한 예측을 계산하여
[**$k$-스텝-앞 예측의 어려움을 좀 더 자세히 살펴봅시다**].

```{.python .input}
%%tab pytorch, mxnet, tensorflow
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab jax
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model.apply({'params': trainer.state.params},
                            d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

이는 저희가 미래로 더 멀리 예측하려고 할 때
예측의 품질이 어떻게 변하는지를 명확하게 보여줍니다.
4-스텝-앞 예측은 여전히 좋아 보이지만,
그 너머는 거의 쓸모가 없습니다.

## 요약

내삽(interpolation)과 외삽(extrapolation) 사이에는
어려움에 있어서 상당한 차이가 있습니다.
따라서 시퀀스를 가지고 있다면, 학습할 때
데이터의 시간 순서를 항상 존중하세요.
즉, 미래 데이터로는 결코 학습하지 마세요.
이러한 종류의 데이터가 주어졌을 때,
시퀀스 모델은 추정을 위해 전문화된 통계 도구를 필요로 합니다.
인기 있는 두 가지 선택지는 자기회귀 모델과
잠재 변수 자기회귀 모델입니다.
인과적 모델(예: 시간이 앞으로 흐르는 경우)에서는,
순방향을 추정하는 것이 일반적으로
역방향보다 훨씬 더 쉽습니다.
타임스텝 $t$까지 관측된 시퀀스에 대해,
타임스텝 $t+k$에서의 예측된 출력은
$k$*-스텝-앞 예측* 입니다.
$k$를 증가시켜 시간상 더 멀리 예측할수록,
오차가 누적되고 예측의 품질이 저하되며,
종종 극적으로 그렇게 됩니다.

## 연습문제

1. 이 절의 실험에 있는 모델을 개선하세요.
    1. 과거 네 개를 초과하는 관측값을 포함시켜 보세요. 실제로는 얼마나 많이 필요한가요?
    1. 만약 잡음이 없었다면 얼마나 많은 과거 관측값이 필요할까요? 힌트: $\sin$과 $\cos$를 미분 방정식으로 쓸 수 있습니다.
    1. 총 특징의 개수를 일정하게 유지하면서 더 오래된 관측값을 포함시킬 수 있나요? 이것이 정확도를 향상시키나요? 왜 그런가요?
    1. 신경망 구조를 바꾸고 성능을 평가하세요. 새 모델을 더 많은 에포크로 학습할 수도 있습니다. 무엇을 관찰하나요?
1. 한 투자자가 매수할 좋은 증권을 찾고자 합니다.
   그들은 어떤 것이 잘 될 가능성이 있는지 결정하기 위해 과거 수익률을 봅니다.
   이 전략에서 무엇이 잘못될 수 있을까요?
1. 인과성이 텍스트에도 적용되나요? 어느 정도까지 그런가요?
1. 데이터의 동역학을 포착하기 위해 잠재 자기회귀 모델이
   필요할 수 있는 경우의 예를 들어 보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18010)
:end_tab:
