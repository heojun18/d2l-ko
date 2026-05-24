```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 다층 퍼셉트론
:label:`sec_mlp`

:numref:`sec_softmax`에서 저희는
소프트맥스 회귀를 소개하면서,
처음부터 직접 알고리즘을 구현해 보고
(:numref:`sec_softmax_scratch`) 고수준 API를 사용해
(:numref:`sec_softmax_concise`) 구현해 보았습니다. 이를 통해 저희는
저해상도 이미지에서 10가지 의류 범주를 인식할 수 있는
분류기를 훈련할 수 있었습니다.
그 과정에서 데이터를 다루는 법,
출력을 유효한 확률 분포로 강제하는 법,
적절한 손실 함수를 적용하는 법,
그리고 모델 파라미터에 대해 손실을 최소화하는 법을 배웠습니다.
이제 이러한 메커니즘을
단순한 선형 모델의 맥락에서 충분히 익혔으므로,
저희는 이 책이 주로 다루는
훨씬 풍부한 모델 부류인
심층 신경망의 탐험을 시작할 수 있습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from jax import grad, vmap
```

## 은닉층

저희는 :numref:`subsec_linear_model`에서
아핀 변환을 편향이 추가된 선형 변환으로 설명했습니다.
먼저, :numref:`fig_softmaxreg`에 설명된
소프트맥스 회귀 예제에 해당하는
모델 구조를 떠올려 봅시다.
이 모델은 단 한 번의 아핀 변환과 그에 이어지는 소프트맥스 연산을 통해
입력을 출력으로 곧바로 사상합니다.
저희의 레이블이 단순한 아핀 변환을 통해
입력 데이터와 정말로 관련되어 있다면,
이 접근법으로 충분할 것입니다.
하지만 (아핀 변환에서의) 선형성은 *강한* 가정입니다.

### 선형 모델의 한계

예를 들어, 선형성은 *단조성*이라는
*더 약한* 가정을 함의합니다. 즉,
저희의 특징이 증가하면 모델 출력이
항상 증가하거나((대응하는 가중치가 양수일 때))
항상 감소해야 한다는 것입니다((대응하는 가중치가 음수일 때)).
때로는 이것이 타당합니다.
예를 들어, 어떤 개인이 대출을 상환할지를 예측하려 한다면,
다른 조건이 모두 같다면
소득이 더 높은 신청자가 항상
소득이 더 낮은 신청자보다 상환할 가능성이 높다고
합리적으로 가정할 수 있습니다.
단조롭기는 하지만, 이 관계는 상환 확률과
선형적으로 연관되지는 않을 가능성이
큽니다. 소득이 \$0에서 \$50,000으로 증가하는 것은
\$100만에서 \$105만으로 증가하는 것보다
상환 가능성을 더 크게 증가시킬
것입니다.
이 문제를 해결하는 한 가지 방법은 결과에 후처리를 적용해
선형성이 더 그럴듯해 보이도록 만드는 것입니다.
가령 로지스틱 사상((그리고 따라서 결과 확률의 로그))을 사용하는 식입니다.

단조성을 위배하는 예제는
쉽게 떠올릴 수 있습니다.
예를 들어 체온의 함수로 건강을
예측하려 한다고 해봅시다.
정상 체온이 37°C((98.6°F)) 이상인 사람의 경우,
더 높은 체온은 더 큰 위험을 나타냅니다.
하지만 체온이 37°C 아래로 떨어지면,
더 낮은 체온이 더 큰 위험을 나타냅니다!
이 경우에도 영리한 전처리, 예를 들어 37°C로부터의 거리를
특징으로 사용하는 식으로 문제를 해결할 수 있습니다.


그렇다면 고양이와 개의 이미지를 분류하는 것은 어떨까요?
위치 (13, 17)의 픽셀 강도를 증가시키면
이미지가 개를 나타낼 가능성이
항상 증가하거나((또는 항상 감소))해야 할까요?
선형 모델에 의존한다는 것은 고양이와 개를 구별하기 위한
유일한 요건이 개별 픽셀의 밝기를 평가하는 것이라는
암묵적인 가정에 해당합니다.
이미지를 반전시켜도 범주가 보존되는 세계에서
이 접근법은 실패할 수밖에 없습니다.

그런데 여기서 선형성의 명백한 부조리에도 불구하고,
앞서 살펴본 예제들과 달리,
간단한 전처리로 이 문제를
해결할 수 있을 것 같지는 않습니다.
이는 어떤 픽셀의 의미가
그 맥락((주변 픽셀의 값))에 따라 복잡한 방식으로
달라지기 때문입니다.
저희 특징 사이의 관련 상호작용을 고려한
어떤 데이터 표현이 존재할 수 있고
그 위에서는 선형 모델이 적절할 수 있겠지만,
저희는 그것을 손으로 계산하는 방법을 알지 못합니다.
심층 신경망을 이용하면 저희는 관측 데이터로
은닉층을 통한 표현과 그 표현 위에서 작동하는
선형 예측기를 함께 학습합니다.

이 비선형성 문제는 적어도 한 세기 동안 연구되어 왔습니다 :cite:`Fisher.1928`. 예를 들어, 결정 트리는
가장 기본적인 형태에서 일련의 이진 결정을 통해
클래스 소속을 결정합니다 :cite:`quinlan2014c4`. 마찬가지로 커널
방법은 수십 년 동안 비선형 의존성을 모델링하는 데 사용되어 왔습니다
:cite:`Aronszajn.1950`. 이는 비모수적 스플라인 모델 :cite:`Wahba.1990`과
커널 방법 :cite:`Scholkopf.Smola.2002`에까지 자리 잡았습니다. 이는 또한 뇌가
매우 자연스럽게 해결하는 문제이기도 합니다. 결국 뉴런은 다른 뉴런에게 신호를 보내고,
그 뉴런은 다시 또 다른 뉴런에게 신호를 보냅니다 :cite:`Cajal.Azoulay.1894`.
결과적으로 저희는 비교적 단순한 변환의 연속을 가지게 됩니다.

### 은닉층 도입

저희는 하나 이상의 은닉층을 도입함으로써
선형 모델의 한계를 극복할 수 있습니다.
이를 위한 가장 쉬운 방법은 여러 개의 완전 연결 층을
서로 쌓아 올리는 것입니다.
각 층은 그 위의 층으로 신호를 보내고,
이는 출력을 생성할 때까지 계속됩니다.
저희는 처음 $L-1$개의 층을
표현으로 생각할 수 있고, 마지막 층을
선형 예측기로 생각할 수 있습니다.
이 구조는 흔히
*다층 퍼셉트론*이라고 불리며,
종종 *MLP*로 줄여서 부릅니다 (:numref:`fig_mlp`).

![5개의 은닉 유닛을 가진 은닉층이 있는 MLP.](../img/mlp.svg)
:label:`fig_mlp`

이 MLP는 4개의 입력, 3개의 출력을 가지며,
은닉층에는 5개의 은닉 유닛이 있습니다.
입력층에서는 어떤 계산도 수행되지 않으므로,
이 네트워크로 출력을 생성하려면
은닉층과 출력층의 계산을
모두 구현해야 합니다.
따라서 이 MLP의 층 수는 2개입니다.
두 층 모두 완전 연결되어 있다는 점에 유의하세요.
모든 입력은 은닉층의 모든 뉴런에 영향을 미치고,
이들은 다시 출력층의 모든 뉴런에
영향을 미칩니다. 하지만, 아직 끝난 것이
아닙니다.

### 선형에서 비선형으로

이전과 마찬가지로 저희는 행렬 $\mathbf{X} \in \mathbb{R}^{n \times d}$로
각 예제가 $d$개의 입력((특징))을 가지는 $n$개의 미니배치 예제를 나타냅니다.
은닉층이 $h$개의 은닉 유닛을 갖는 단일 은닉층 MLP의 경우,
저희는 은닉층의 출력을 $\mathbf{H} \in \mathbb{R}^{n \times h}$로 나타내며,
이는 *은닉 표현*입니다.
은닉층과 출력층이 모두 완전 연결되어 있으므로,
은닉층 가중치 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$와 편향 $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$,
그리고 출력층 가중치 $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$와 편향 $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$가 있습니다.
이를 통해 단일 은닉층 MLP의 출력 $\mathbf{O} \in \mathbb{R}^{n \times q}$를
다음과 같이 계산할 수 있습니다.

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

은닉층을 추가한 후, 저희 모델은 이제
추가적인 파라미터 집합을 추적하고 업데이트해야 한다는 점에 유의하세요.
그렇다면 그 대신 저희는 무엇을 얻었을까요?
놀랍게도 (위에서 정의한 모델에서는) *저희의 수고에 비해
얻은 것이 전혀 없습니다*!
그 이유는 분명합니다.
위의 은닉 유닛은
입력의 아핀 함수로 주어지고,
출력((소프트맥스 이전))은 단지
은닉 유닛의 아핀 함수일 뿐입니다.
아핀 함수의 아핀 함수는
그 자체로 아핀 함수입니다.
게다가 저희의 선형 모델은 이미
어떤 아핀 함수든 표현할 수 있었습니다.

이를 형식적으로 보기 위해, 위 정의에서 은닉층을 그냥 합쳐 버리면
파라미터 $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$와 $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$를 갖는
동등한 단일층 모델을 얻을 수 있습니다.

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

다층 구조의 잠재력을 실현하려면,
또 하나의 핵심 요소가 필요합니다. 바로
아핀 변환 후에 각 은닉 유닛에 적용할
비선형 *활성화 함수* $\sigma$입니다. 예를 들어, 널리 사용되는
선택지는 ReLU((rectified linear unit)) 활성화 함수 :cite:`Nair.Hinton.2010`이며,
이는 인수에 원소 단위로 작용하는 $\sigma(x) = \mathrm{max}(0, x)$입니다.
활성화 함수 $\sigma(\cdot)$의 출력은
*활성화*라고 불립니다.
일반적으로 활성화 함수가 적용된 상태에서는
저희의 MLP를 선형 모델로 합쳐 버릴 수 없습니다.

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

$\mathbf{X}$의 각 행이 미니배치의 한 예제에 해당하므로,
다소 표기를 남용하여 저희는 비선형 함수
$\sigma$가 입력에 행 단위로,
즉 한 번에 한 예제씩 적용되도록 정의합니다.
:numref:`subsec_softmax_vectorization`에서 행 단위 연산을 나타낼 때
저희가 소프트맥스에 동일한 표기를 사용했다는 점에 유의하세요.
저희가 사용하는 활성화 함수는 단순히 행 단위가 아니라
원소 단위로 적용되는 경우가 꽤 자주 있습니다. 즉, 층의 선형 부분을 계산한 후
다른 은닉 유닛이 취하는 값을 보지 않고도
각 활성화를 계산할 수 있습니다.

더 일반적인 MLP를 만들기 위해, 저희는
$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$,
$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$ 등으로
이러한 은닉층을 서로 위에 계속 쌓아
점점 더 표현력이 풍부한 모델을 만들 수 있습니다.

### 보편 근사기

저희는 뇌가 매우 정교한 통계 분석이 가능하다는 것을 알고 있습니다. 그래서
심층 네트워크가 *얼마나 강력할 수* 있는지 묻는 것은 가치 있는 일입니다. 이 질문은
여러 번 답해져 왔는데, 예를 들어 MLP 맥락에서는 :citet:`Cybenko.1989`에서,
재생 커널 힐베르트 공간의 맥락에서는 :citet:`micchelli1984interpolation`에서 답해졌으며,
이는 단일 은닉층을 가진 방사 기저 함수((RBF)) 네트워크로 볼 수 있는 방식이었습니다.
이러한 ((그리고 관련된)) 결과들은 단일 은닉층 네트워크라도
충분한 노드((아마도 터무니없이 많은))와
적절한 가중치 집합이 주어진다면,
어떤 함수든 모델링할 수 있음을 시사합니다.
하지만 실제로 그 함수를 학습하는 것은 어려운 부분입니다.
여러분은 신경망을
C 프로그래밍 언어와 비슷한 것으로 생각해 볼 수 있습니다.
이 언어는, 다른 어떤 현대 언어와 마찬가지로,
계산 가능한 모든 프로그램을 표현할 수 있습니다.
하지만 실제로 여러분의 명세를 충족하는
프로그램을 만들어 내는 것은 어려운 부분입니다.

게다가 단일 은닉층 네트워크가
어떤 함수든 학습할 *수* 있다고 해서
모든 문제를 그 하나로
풀려고 시도해야 한다는 의미는 아닙니다. 사실, 이 경우 커널 방법이
훨씬 더 효과적입니다. 무한 차원 공간에서도
문제를 *정확하게* 풀 수 있기 때문입니다 :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`.
사실, 저희는 ((더 넓은 대신)) 더 깊은 네트워크를 사용하여 많은 함수를
훨씬 더 간결하게 근사할 수 있습니다 :cite:`Simonyan.Zisserman.2014`.
이후 장들에서 더 엄밀한 논의를 다룰 것입니다.


## 활성화 함수
:label:`subsec_activation-functions`

활성화 함수는 가중합을 계산하고 거기에 편향을 더한 후
뉴런을 활성화할지 말지를 결정합니다.
이들은 입력 신호를 출력으로 변환하는 미분 가능한 연산자이며,
대부분이 비선형성을 추가합니다.
활성화 함수는 딥러닝의 기초이므로,
(**흔히 사용되는 몇 가지를 간단히 살펴봅시다**).

### ReLU 함수

구현의 단순함과
다양한 예측 작업에서의 좋은 성능 덕분에
가장 인기 있는 선택은
*rectified linear unit* (*ReLU*) :cite:`Nair.Hinton.2010`입니다.
[**ReLU는 매우 단순한 비선형 변환을 제공합니다**].
원소 $x$가 주어지면, 함수는 그 원소와 $0$의
최댓값으로 정의됩니다.

$$\operatorname{ReLU}(x) = \max(x, 0).$$

비공식적으로 말하자면, ReLU 함수는 양수 원소만 유지하고
음수 원소는 모두 대응하는 활성화를 0으로 설정함으로써
버립니다.
직관을 얻기 위해 함수를 그려볼 수 있습니다.
보시다시피 활성화 함수는 구간별 선형입니다.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

입력이 음수일 때
ReLU 함수의 도함수는 0이고,
입력이 양수일 때
ReLU 함수의 도함수는 1입니다.
입력이 정확히 0의 값을 가질 때
ReLU 함수는 미분 가능하지 않다는 점에 유의하세요.
이러한 경우 저희는 기본적으로 왼쪽
도함수를 사용하고 입력이 0일 때 도함수가 0이라고 말합니다.
입력이 실제로 정확히 0이 되는 경우는 결코 없을 수 있기 때문에
저희는 이 점에서 그냥 넘어갈 수 있습니다 ((수학자들은
이것이 측도가 0인 집합 위에서 미분 불가능하다고 말할 것입니다)).
미묘한 경계 조건이 중요하다면 저희가
((진짜)) 수학을 하고 있는 것이지 공학을 하고 있는 것이 아니라는 옛 격언이 있습니다.
그 관습적인 지혜가 여기에 적용될 수도 있고, 적어도
저희가 제약 최적화를 수행하고 있지는 않다는 사실이 그렇습니다 :cite:`Mangasarian.1965,Rockafellar.1970`.
아래에 ReLU 함수의 도함수를 그려봅니다.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

ReLU를 사용하는 이유는
그 도함수가 특히 잘 거동하기 때문입니다.
도함수가 사라지거나, 아니면 단순히 인수를 그대로 통과시킵니다.
이는 최적화를 더 잘 거동하게 만들고
이전 버전의 신경망을 괴롭혔던
잘 알려진 기울기 소실 문제를 완화했습니다 ((이에 대해서는 나중에 더 다룹니다)).

ReLU 함수에는 *parametrized ReLU* (*pReLU*) 함수 :cite:`He.Zhang.Ren.ea.2015`를 비롯한
많은 변형이 있다는 점에 유의하세요.
이 변형은 ReLU에 선형 항을 추가하여
인수가 음수일 때에도
일부 정보가 여전히 통과할 수 있도록 합니다.

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### 시그모이드 함수

[**시그모이드 함수는 입력값을**]
$\mathbb{R}$ 정의역에서 (**구간 (0, 1) 위의 출력으로 변환합니다.**)
이러한 이유로 시그모이드는
종종 *압축 함수*라고 불립니다.
이는 (-inf, inf) 범위의 어떤 입력이든
(0, 1) 범위의 어떤 값으로 압축합니다.

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

초기 신경망에서, 과학자들은
*발화*하거나 *발화하지 않는* 생물학적 뉴런을
모델링하는 데 관심이 있었습니다.
따라서 이 분야의 선구자들, 멀게는
인공 뉴런의 발명자인 McCulloch와 Pitts에 이르기까지,
모두 임계값 유닛에 초점을 맞췄습니다 :cite:`McCulloch.Pitts.1943`.
임계값 활성화는 입력이 어떤 임계값 아래일 때
값 0을 취하고
입력이 임계값을 초과할 때 값 1을 취합니다.

기울기 기반 학습으로 관심이 옮겨가면서,
시그모이드 함수는 임계값 유닛에 대한
부드럽고 미분 가능한 근사이기 때문에
자연스러운 선택이 되었습니다.
저희가 이진 분류 문제에서 출력을 확률로 해석하고자 할 때
시그모이드는 여전히 출력 유닛의
활성화 함수로 널리 사용됩니다. 시그모이드는 소프트맥스의 특수한 경우라고 생각할 수 있습니다.
하지만 시그모이드는 은닉층에서의 대부분의 용도에서
더 단순하고 훈련하기 쉬운 ReLU에
크게 자리를 내주었습니다. 이는 주로
시그모이드가 최적화에 어려움을 일으킨다는 사실
:cite:`LeCun.Bottou.Orr.ea.1998`과 관련이 있는데, 큰 양수와 음수 인수 *모두*에 대해 그 기울기가 사라지기 때문입니다.
이는 벗어나기 어려운 평탄한 영역으로 이어질 수 있습니다.
그럼에도 시그모이드는 중요합니다. 순환 신경망에 관한 이후 장들((예: :numref:`sec_lstm`))에서
저희는 시간에 걸친 정보 흐름을 제어하는 데
시그모이드 유닛을 활용하는 구조를 설명할 것입니다.

아래에서 시그모이드 함수를 그려봅니다.
입력이 0에 가까울 때
시그모이드 함수가 선형 변환에
가까워진다는 점에 유의하세요.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

시그모이드 함수의 도함수는 다음 식으로 주어집니다.

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


시그모이드 함수의 도함수를 아래에 그려봅니다.
입력이 0일 때
시그모이드 함수의 도함수가
최댓값 0.25에 도달한다는 점에 유의하세요.
입력이 어느 방향으로든 0에서 멀어지면,
도함수는 0에 가까워집니다.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### Tanh 함수
:label:`subsec_tanh`

시그모이드 함수와 마찬가지로, [**tanh((쌍곡 탄젠트))
함수 또한 입력을 압축하여**]
구간 (**$-1$과 $1$ 사이의**) 원소로 변환합니다.

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

아래에 tanh 함수를 그려봅니다. 입력이 0에 가까워지면 tanh 함수는 선형 변환에 가까워진다는 점에 유의하세요. 함수의 모양은 시그모이드 함수와 비슷하지만, tanh 함수는 좌표계의 원점에 대해 점대칭을 나타냅니다 :cite:`Kalman.Kwasny.1992`.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

tanh 함수의 도함수는 다음과 같습니다.

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

이를 아래에 그려봅니다.
입력이 0에 가까워지면
tanh 함수의 도함수는 최댓값 1에 가까워집니다.
그리고 저희가 시그모이드 함수에서 보았듯이,
입력이 어느 방향으로든 0에서 멀어지면,
tanh 함수의 도함수는 0에 가까워집니다.

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## 요약 및 논의

이제 저희는 표현력 있는 다층 신경망 구조를 만들기 위해
비선형성을 어떻게 도입하는지 알게 되었습니다.
참고로 여러분의 지식은 이미
1990년경의 실무자가 가지고 있던 도구 모음과
유사한 것을 다룰 수 있는 수준입니다.
어떤 면에서 여러분은
그 당시 일하던 누구보다도 유리한데,
강력한 오픈 소스 딥러닝 프레임워크를 활용하여
단 몇 줄의 코드로 빠르게 모델을 만들 수 있기 때문입니다.
이전에는 이러한 네트워크를 훈련하기 위해 연구자들이
C, Fortran, 또는 ((LeNet의 경우)) Lisp으로
층과 도함수를 명시적으로 코딩해야 했습니다.

두 번째 이점은 ReLU가 시그모이드나 tanh 함수보다
최적화에 훨씬 더 적합하다는 것입니다. 이것이 지난 10년간
딥러닝의 부활을 도운 핵심 혁신 중 하나라고
주장할 수도 있습니다. 다만 활성화 함수에 관한 연구가
멈춘 것은 아니라는 점에 유의하세요.
예를 들어,
:citet:`Hendrycks.Gimpel.2016`의 GELU((Gaussian error linear unit))
활성화 함수 $x \Phi(x)$ (($\Phi(x)$는
표준 가우시안 누적 분포 함수))와
:citet:`Ramachandran.Zoph.Le.2017`에서 제안된
Swish 활성화 함수
$\sigma(x) = x \operatorname{sigmoid}(\beta x)$는 많은 경우 더 나은 정확도를
낼 수 있습니다.

## 연습문제

1. *선형* 심층 네트워크, 즉 비선형성 $\sigma$가 없는 네트워크에 층을 추가하는 것이
   결코 네트워크의 표현력을 증가시킬 수 없다는 것을 보이세요.
   오히려 표현력을 줄이는 예제를 들어보세요.
1. pReLU 활성화 함수의 도함수를 계산하세요.
1. Swish 활성화 함수 $x \operatorname{sigmoid}(\beta x)$의 도함수를 계산하세요.
1. ReLU ((또는 pReLU))만 사용하는 MLP가
   연속 구간별 선형 함수를 구성한다는 것을 보이세요.
1. 시그모이드와 tanh는 매우 비슷합니다.
    1. $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$임을 보이세요.
    1. 두 비선형성으로 매개변수화된 함수 부류가 동일함을 증명하세요. 힌트: 아핀 층은 편향 항도 가집니다.
1. 배치 정규화 :cite:`Ioffe.Szegedy.2015`와 같이 한 번에 하나의 미니배치에 적용되는 비선형성이 있다고 가정합시다. 이것이 어떤 종류의 문제를 일으킬 것으로 예상하나요?
1. 시그모이드 활성화 함수에 대해 기울기가 사라지는 예제를 제시하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17984)
:end_tab:
