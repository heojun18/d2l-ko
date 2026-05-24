# 언어 모델 (Language Models)
:label:`sec_language-model`

:numref:`sec_text-sequence`에서 저희는 텍스트 시퀀스를 토큰으로 매핑하는 방법을 봤는데, 이러한 토큰은 단어 또는 문자와 같은 이산 관측값의 시퀀스로 볼 수 있습니다. 길이 $T$의 텍스트 시퀀스에 있는 토큰이 차례로 $x_1, x_2, \ldots, x_T$라고 가정해 봅시다.
*언어 모델(language models)* 의 목표는
전체 시퀀스의 결합 확률을 추정하는 것입니다.

$$P(x_1, x_2, \ldots, x_T),$$

여기서는 :numref:`sec_sequence`의
통계 도구를 적용할 수 있습니다.

언어 모델은 매우 유용합니다. 예를 들어, 이상적인 언어 모델은 단순히 한 번에 하나의 토큰을 $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$로 추출함으로써 스스로 자연스러운 텍스트를 생성할 수 있어야 합니다.
타자기를 사용하는 원숭이와는 사뭇 달리, 그러한 모델에서 나오는 모든 텍스트는 자연어, 예를 들어 영어 텍스트로 통할 것입니다. 더 나아가, 이전 대화 단편으로 텍스트를 조건화하는 것만으로도 의미 있는 대화를 생성하기에 충분할 것입니다.
이러한 시스템을 설계하는 것에 저희가 여전히 매우 멀리 떨어져 있다는 점은 분명한데, 왜냐하면 그것은 단지 문법적으로 합리적인 콘텐츠를 생성하는 것을 넘어 텍스트를 *이해*해야 하기 때문입니다.

그럼에도 불구하고, 언어 모델은 제한된 형태에서조차 큰 도움이 됩니다.
예를 들어, "to recognize speech"와 "to wreck a nice beach"라는 구절은 매우 비슷하게 들립니다.
이는 음성 인식에서 모호함을 일으킬 수 있는데,
이는 두 번째 번역을 터무니없다고 거부하는 언어 모델을 통해 쉽게 해결됩니다.
마찬가지로, 문서 요약 알고리즘에서
"dog bites man"이 "man bites dog"보다 훨씬 더 빈번하다는 것이나, "I want to eat grandma"는 다소 불안한 문장인 반면 "I want to eat, grandma"는 훨씬 더 무해하다는 것을 아는 것이 유용합니다.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## 언어 모델 학습하기

당연한 질문은 저희가 문서, 혹은 심지어 토큰의 시퀀스를 어떻게 모델링해야 하는지입니다.
텍스트 데이터를 단어 수준에서 토큰화한다고 가정해 봅시다.
기본적인 확률 규칙을 적용하는 것부터 시작해 봅시다.

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

예를 들어,
네 단어를 포함하는 텍스트 시퀀스의 확률은 다음과 같이 주어집니다.

$$\begin{aligned}&P(\textrm{deep}, \textrm{learning}, \textrm{is}, \textrm{fun}) \\
=&P(\textrm{deep}) P(\textrm{learning}  \mid  \textrm{deep}) P(\textrm{is}  \mid  \textrm{deep}, \textrm{learning}) P(\textrm{fun}  \mid  \textrm{deep}, \textrm{learning}, \textrm{is}).\end{aligned}$$

### 마르코프 모델과 $n$-그램
:label:`subsec_markov-models-and-n-grams`

:numref:`sec_sequence`의 시퀀스 모델 분석 가운데,
언어 모델링에 마르코프 모델을 적용해 봅시다.
시퀀스에 대한 분포가 1차 마르코프 성질을 만족하는 것은 $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$일 때입니다. 더 높은 차수는 더 긴 의존성에 해당합니다. 이는 저희가 시퀀스를 모델링하기 위해 적용할 수 있는 여러 가지 근사로 이어집니다.

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

하나, 둘, 셋의 변수를 포함하는 확률 공식은 각각 일반적으로 *유니그램(unigram)*, *바이그램(bigram)*, *트라이그램(trigram)* 모델이라고 합니다.
언어 모델을 계산하려면, 단어의 확률과
이전 몇 개 단어가 주어졌을 때
한 단어의 조건부 확률을 계산해야 합니다.
이러한 확률들이
언어 모델의 파라미터라는 점에
유의하세요.



### 단어 빈도

여기서 저희는 학습 데이터셋이
모든 위키피디아 항목, [프로젝트 구텐베르크(Project Gutenberg)](https://en.wikipedia.org/wiki/Project_Gutenberg),
그리고 웹에 게시된 모든 텍스트와 같은
큰 텍스트 코퍼스라고 가정합니다.
단어의 확률은 학습 데이터셋에서 주어진 단어의
상대적인 단어 빈도로부터 계산할 수 있습니다.
예를 들어, 추정값 $\hat{P}(\textrm{deep})$은
"deep"이라는 단어로 시작하는 어떤 문장의 확률로 계산할 수 있습니다.
약간 덜 정확한 접근은 "deep"이라는 단어의 모든 발생을 세어서
코퍼스의 총 단어 수로 나누는 것입니다.
이것은 특히 빈도가 높은 단어에 대해서는 꽤 잘 작동합니다.
계속해서, 저희는 다음을 추정해 볼 수 있습니다.

$$\hat{P}(\textrm{learning} \mid \textrm{deep}) = \frac{n(\textrm{deep, learning})}{n(\textrm{deep})},$$

여기서 $n(x)$와 $n(x, x')$는 각각 단일 단어와
연속된 단어 쌍의 발생 횟수입니다.
안타깝게도,
단어 쌍의 확률을 추정하는 것은 다소 더 어려운데,
왜냐하면 "deep learning"의 발생이 훨씬 덜 빈번하기 때문입니다.
특히, 일부 흔치 않은 단어 조합에 대해서는
정확한 추정을 얻기에 충분한 발생을 찾는 것이 까다로울 수 있습니다.
:numref:`subsec_natural-lang-stat`의 경험적 결과가 시사하듯이,
세 단어 조합과 그 이상에 대해서는 상황이 더 악화됩니다.
저희의 데이터셋에서 보지 못할 것 같은 그럴듯한 세 단어 조합이 많이 있을 것입니다.
저희가 그러한 단어 조합에 0이 아닌 개수를 할당하는 어떤 해결책을 제공하지 않는 한, 그것들을 언어 모델에서 사용할 수 없을 것입니다. 데이터셋이 작거나 단어가 매우 드물다면, 저희는 그것들 중 단 하나조차도 찾지 못할 수 있습니다.

### 라플라스 평활화 (Laplace Smoothing)

흔한 전략은 어떤 형태의 *라플라스 평활화(Laplace smoothing)* 를 수행하는 것입니다.
해결책은 모든 개수에 작은 상수를 더하는 것입니다.
학습 세트의 총 단어 수를 $n$으로,
고유한 단어의 수를 $m$으로 표기합시다.
이 해결책은 다음과 같은 식을 통해 단일 단어에 도움이 됩니다.

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

여기서 $\epsilon_1,\epsilon_2$, $\epsilon_3$은 하이퍼파라미터입니다.
$\epsilon_1$을 예로 들어 봅시다.
$\epsilon_1 = 0$일 때, 평활화가 적용되지 않습니다.
$\epsilon_1$이 양의 무한대에 가까워질 때,
$\hat{P}(x)$는 균일 확률 $1/m$에 가까워집니다.
위는 다른 기법이 달성할 수 있는 것의
다소 원시적인 변형입니다 :cite:`Wood.Gasthaus.Archambeau.ea.2011`.


안타깝게도, 이와 같은 모델은 다음의 이유로
다소 빠르게 다루기 힘들어집니다.
첫째,
:numref:`subsec_natural-lang-stat`에서 논의했듯이,
많은 $n$-그램이 매우 드물게 발생하여,
라플라스 평활화를 언어 모델링에 다소 부적합하게 만듭니다.
둘째, 모든 개수를 저장해야 합니다.
셋째, 이는 단어의 의미를 완전히 무시합니다. 예를 들어,
"cat"과 "feline"은 관련된 문맥에서 발생해야 합니다.
이러한 모델을 추가적인 문맥에 맞게 조정하는 것은 꽤 어려운 반면,
딥러닝 기반 언어 모델은 이를 고려하기에
적합합니다.
마지막으로, 긴 단어 시퀀스는 거의 확실하게 새로울 것이므로,
이전에 본 단어 시퀀스의 빈도를 단순히 세는 모델은
거기서 형편없이 동작할 수밖에 없습니다.
따라서, 저희는 이 장의 나머지 부분에서
언어 모델링을 위해 신경망을 사용하는 데 초점을 맞춥니다.


## 펄플렉서티 (Perplexity)
:label:`subsec_perplexity`

다음으로, 언어 모델의 품질을 측정하는 방법에 대해 논의해 봅시다. 그런 다음 이를 사용하여 후속 절에서 저희의 모델을 평가할 것입니다.
한 가지 방법은 텍스트가 얼마나 놀라운지 확인하는 것입니다.
좋은 언어 모델은 다음에 오는 토큰을 높은 정확도로 예측할 수 있습니다.
다른 언어 모델이 제안한 다음과 같은 "It is raining" 구절의 이어짐을 생각해 봅시다.

1. "It is raining outside"
1. "It is raining banana tree"
1. "It is raining piouw;kcj pwepoiut"

품질 면에서 예시 1이 분명히 가장 좋습니다. 단어들은 합리적이고 논리적으로 일관됩니다.
어떤 단어가 의미상 따라오는지를 꽤 정확하게 반영하지는 않을 수 있지만 ("in San Francisco"와 "in winter"는 완벽하게 합리적인 확장이었을 것입니다), 모델은 어떤 종류의 단어가 따라오는지를 포착할 수 있습니다.
예시 2는 무의미한 확장을 만들어내며 상당히 더 나쁩니다. 그럼에도 불구하고, 적어도 모델은 단어의 철자를 어떻게 쓰는지와 단어 사이의 어느 정도의 상관관계를 배웠습니다. 마지막으로, 예시 3은 데이터에 제대로 적합하지 않은, 잘 학습되지 않은 모델을 나타냅니다.

저희는 시퀀스의 가능도를 계산하여 모델의 품질을 측정할 수도 있습니다.
안타깝게도 이는 이해하기 어렵고 비교하기 어려운 숫자입니다.
결국, 짧은 시퀀스가 긴 시퀀스보다 훨씬 더 발생하기 쉬우므로,
톨스토이의 대작
*전쟁과 평화*에서 모델을 평가하는 것은 필연적으로, 가령 생텍쥐페리의 중편 소설 *어린 왕자*에서보다 훨씬 더 작은 가능도를 만들어낼 것입니다. 빠진 것은 평균에 해당하는 것입니다.

여기서 정보 이론이 유용합니다.
저희는 소프트맥스 회귀를 소개할 때
엔트로피, 놀라움(surprisal), 교차 엔트로피를 정의했습니다
(:numref:`subsec_info_theory_basics`).
텍스트를 압축하고자 한다면, 저희는 현재 토큰의 집합이 주어졌을 때
다음 토큰을 예측하는 것에 관해 물어볼 수 있습니다.
더 나은 언어 모델은 저희가 다음 토큰을 더 정확하게 예측할 수 있게 해야 합니다.
따라서, 그것은 저희가 시퀀스를 압축하는 데 더 적은 비트를 쓸 수 있게 해야 합니다.
그래서 저희는 그것을 시퀀스의 모든 $n$개 토큰에 걸쳐 평균낸
교차 엔트로피 손실로 측정할 수 있습니다.

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

여기서 $P$는 언어 모델에 의해 주어지고 $x_t$는 시퀀스에서 타임스텝 $t$에 관측된 실제 토큰입니다.
이는 다양한 길이의 문서에서의 성능을 비교할 수 있게 만듭니다. 역사적인 이유로, 자연어 처리 분야의 과학자들은 *펄플렉서티(perplexity)* 라고 불리는 양을 사용하는 것을 선호합니다. 한마디로, 그것은 :eqref:`eq_avg_ce_for_lm`의 지수함수입니다.

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

펄플렉서티는 다음에 어느 토큰을 고를지 결정할 때 저희가 가지는 실제 선택지의 수의 기하 평균의 역수로 가장 잘 이해될 수 있습니다. 여러 경우를 살펴봅시다.

* 최선의 경우 시나리오에서는, 모델이 항상 타깃 토큰의 확률을 1로 완벽하게 추정합니다. 이 경우 모델의 펄플렉서티는 1입니다.
* 최악의 경우 시나리오에서는, 모델이 항상 타깃 토큰의 확률을 0으로 예측합니다. 이 상황에서, 펄플렉서티는 양의 무한대입니다.
* 베이스라인에서는, 모델이 어휘의 모든 사용 가능한 토큰에 대해 균일 분포를 예측합니다. 이 경우, 펄플렉서티는 어휘의 고유 토큰의 수와 같습니다. 사실, 만약 저희가 어떤 압축도 없이 시퀀스를 저장한다면, 이것이 인코딩에 대해 저희가 할 수 있는 최선일 것입니다. 따라서, 이는 어떤 유용한 모델이든 반드시 이겨야 하는 자명하지 않은 상한을 제공합니다.

## 시퀀스 분할
:label:`subsec_partitioning-seqs`

저희는 신경망을 사용하여 언어 모델을 설계하고
텍스트 시퀀스에서 현재 토큰의 집합이 주어졌을 때
다음 토큰을 예측하는 데 모델이 얼마나 좋은지
평가하기 위해 펄플렉서티를 사용할 것입니다.
모델을 소개하기 전에,
모델이 한 번에 미리 정의된 길이의
시퀀스 미니배치를 처리한다고 가정해 봅시다.
이제 질문은 [**입력 시퀀스와 타깃 시퀀스의 미니배치를 무작위로 읽는**] 방법입니다.


데이터셋이 `corpus`에 $T$개의 토큰 인덱스 시퀀스 형태로 되어 있다고 가정합시다.
저희는 그것을 부분 시퀀스로 분할할 것이며, 각 부분 시퀀스는 $n$개의 토큰(타임스텝)을 가집니다.
각 에포크마다
전체 데이터셋의 (거의) 모든 토큰을 순회하고
가능한 모든 길이-$n$ 부분 시퀀스를 얻기 위해,
저희는 무작위성을 도입할 수 있습니다.
더 구체적으로,
각 에포크의 시작에서,
처음 $d$개의 토큰을 버립니다.
여기서 $d\in [0,n)$는 균일하게 무작위로 샘플링됩니다.
그런 다음 나머지 시퀀스는
$m=\lfloor (T-d)/n \rfloor$개의 부분 시퀀스로 분할됩니다.
타임스텝 $t$의 토큰 $x_t$에서 시작하는 길이-$n$ 부분 시퀀스를 $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$로 표기합시다.
결과로 나온 $m$개의 분할된 부분 시퀀스는
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}$입니다.
각 부분 시퀀스는 언어 모델로의 입력 시퀀스로 사용될 것입니다.


언어 모델링에서,
목표는 지금까지 본 토큰을 바탕으로 다음 토큰을 예측하는 것이며, 따라서 타깃(레이블)은 한 토큰씩 이동된 원래 시퀀스입니다.
어떤 입력 시퀀스 $\mathbf x_t$에 대한 타깃 시퀀스는
길이 $n$의 $\mathbf x_{t+1}$입니다.

![분할된 길이-5 부분 시퀀스로부터 입력 시퀀스와 타깃 시퀀스의 다섯 쌍을 얻기.](../img/lang-model-data.svg)
:label:`fig_lang_model_data`

:numref:`fig_lang_model_data`는 $n=5$이고 $d=2$일 때 입력 시퀀스와 타깃 시퀀스의 다섯 쌍을 얻는 예시를 보여줍니다.

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = d2l.tensor([corpus[i:i+num_steps+1] 
                        for i in range(len(corpus)-num_steps)])
    self.X, self.Y = array[:,:-1], array[:,1:]
```

언어 모델을 학습시키기 위해,
저희는 미니배치 단위로 입력 시퀀스와 타깃 시퀀스의 쌍을
무작위로 샘플링할 것입니다.
다음 데이터 로더는 매번 데이터셋에서 무작위로 미니배치를 생성합니다.
인자 `batch_size`는 각 미니배치의 부분 시퀀스 예시의 개수를 지정하고
`num_steps`는 토큰 단위의 부분 시퀀스 길이입니다.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

다음에서 볼 수 있듯이,
타깃 시퀀스의 미니배치는
입력 시퀀스를 한 토큰씩
이동시킴으로써 얻을 수 있습니다.

```{.python .input  n=7}
%%tab all
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
```

## 요약 및 논의

언어 모델은 텍스트 시퀀스의 결합 확률을 추정합니다. 긴 시퀀스의 경우, $n$-그램은 의존성을 절단함으로써 편리한 모델을 제공합니다. 그러나 구조는 많지만 라플라스 평활화를 통해 빈도가 낮은 단어 조합을 효율적으로 다루기에는 빈도가 충분하지 않습니다. 따라서, 저희는 후속 절에서 신경망 언어 모델링에 초점을 맞출 것입니다.
언어 모델을 학습시키기 위해, 저희는 미니배치 단위로 입력 시퀀스와 타깃 시퀀스의 쌍을 무작위로 샘플링할 수 있습니다. 학습 후, 저희는 언어 모델의 품질을 측정하기 위해 펄플렉서티를 사용할 것입니다.

언어 모델은 데이터 크기, 모델 크기, 학습 컴퓨팅 양의 증가로 확장될 수 있습니다. 큰 언어 모델은 입력 텍스트 지시에 대한 출력 텍스트를 예측함으로써 원하는 작업을 수행할 수 있습니다. 저희가 나중에 논의할 것처럼 (예: :numref:`sec_large-pretraining-transformers`),
현재 시점에
큰 언어 모델은 다양한 작업에 걸친 최첨단 시스템의 기반을 이룹니다.


## 연습문제

1. 학습 데이터셋에 100,000개의 단어가 있다고 가정해 봅시다. 4-그램은 얼마만큼의 단어 빈도와 다중 단어 인접 빈도를 저장해야 할까요?
1. 대화를 어떻게 모델링하시겠습니까?
1. 긴 시퀀스 데이터를 읽기 위한 다른 어떤 방법을 떠올릴 수 있나요?
1. 각 에포크의 시작에서 처음 몇 개의 토큰을 균일하게 무작위인 개수만큼 버리는 저희의 방법을 생각해 봅시다.
    1. 그것이 정말로 문서의 시퀀스에 대한 완벽하게 균일한 분포로 이어지나요?
    1. 더욱 균일하게 만들기 위해서는 무엇을 해야 할까요?
1. 시퀀스 예시가 완전한 문장이 되기를 원한다면, 이는 미니배치 샘플링에서 어떤 종류의 문제를 도입할까요? 이를 어떻게 고칠 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18012)
:end_tab:
