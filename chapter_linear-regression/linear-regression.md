```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형 회귀
:label:`sec_linear_regression`

*회귀(Regression)* 문제는 수치 값을 예측하고자 할 때마다 등장합니다.
일반적인 예로는 가격 예측(주택, 주식 등),
입원 기간 예측(병원 환자의 경우),
수요 예측(소매 판매의 경우) 등 수많은 사례가 있습니다.
모든 예측 문제가 고전적인 회귀 문제는 아닙니다.
이후에는 분류 문제를 다룰 텐데,
이는 어떤 범주들의 집합 중 어디에 속하는지를 예측하는 것이 목표입니다.

진행 예시로, 면적(평방피트)과 연식(년) 정보를 바탕으로
주택 가격(달러)을 추정하고자 한다고 가정해 봅시다.
주택 가격을 예측하는 모델을 개발하려면,
각 주택의 판매 가격, 면적, 연식이 포함된
데이터를 확보해야 합니다.
머신러닝 용어로 이 데이터셋은 *훈련 데이터셋(training dataset)*
또는 *훈련 세트(training set)*라고 부르며,
각 행(한 건의 판매에 해당하는 데이터를 담고 있는)은
*예제(example)*(또는 *데이터 포인트(data point)*, *인스턴스(instance)*, *샘플(sample)*)라고 합니다.
저희가 예측하고자 하는 대상(가격)은
*레이블(label)*(또는 *타깃(target)*)이라고 합니다.
예측의 근거가 되는 변수들(연식과 면적)은
*특징(feature)*(또는 *공변량(covariate)*)이라고 합니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from jax import numpy as jnp
import math
import time
```

## 기초

*선형 회귀(Linear regression)*는 회귀 문제를 다루는
표준 도구 중에서 가장 단순하면서도 가장 널리 사용되는 방법입니다.
19세기 초로 거슬러 올라가는 :cite:`Legendre.1805,Gauss.1809`
선형 회귀는 몇 가지 단순한 가정에서 출발합니다.
첫째, 특징 $\mathbf{x}$와 타깃 $y$ 사이의 관계가
대략적으로 선형이라고 가정합니다.
즉, 조건부 평균 $E[Y \mid X=\mathbf{x}]$가
특징 $\mathbf{x}$의 가중합으로 표현될 수 있다고 봅니다.
이러한 설정은 관측 잡음으로 인해 타깃 값이
기대값에서 벗어날 수도 있음을 허용합니다.
다음으로, 이러한 잡음이 가우시안 분포를 따르며
잘 거동한다는 가정을 추가할 수 있습니다.
일반적으로 $n$을 데이터셋의 예제 개수를 나타내는 데 사용합니다.
샘플과 타깃을 열거할 때는 위 첨자를 사용하고,
좌표를 인덱싱할 때는 아래 첨자를 사용합니다.
좀 더 구체적으로,
$\mathbf{x}^{(i)}$는 $i$번째 샘플을 나타내고,
$x_j^{(i)}$는 그 샘플의 $j$번째 좌표를 나타냅니다.

### 모델
:label:`subsec_linear_model`

모든 해법의 중심에는 특징을 어떻게
타깃의 추정값으로 변환할지를 기술하는 모델이 있습니다.
선형성 가정은 타깃의 기대값(가격)이
특징(면적과 연식)의 가중합으로 표현될 수 있음을 의미합니다.

$$\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b.$$
:eqlabel:`eq_price-area`

여기서 $w_{\textrm{area}}$와 $w_{\textrm{age}}$는
*가중치(weight)*라고 부르고, $b$는 *편향(bias)*
(또는 *오프셋(offset)*이나 *절편(intercept)*)이라고 부릅니다.
가중치는 각 특징이 예측에 미치는 영향을 결정합니다.
편향은 모든 특징이 0일 때 추정값이 어떤 값을 가질지를 결정합니다.
신축 주택 중 면적이 정확히 0인 집을 보는 일은 결코 없겠지만,
저희가 특징의 모든 선형 함수를 표현할 수 있도록 해 주기 때문에
(원점을 지나는 직선으로만 제한되지 않도록) 편향은 여전히 필요합니다.
엄밀히 말하면 :eqref:`eq_price-area`는 입력 특징의 *아핀 변환(affine transformation)*이며,
가중합을 통한 특징의 *선형 변환(linear transformation)*과 더해진 편향에 의한 *평행 이동(translation)*으로 특징지어집니다.
데이터셋이 주어졌을 때, 저희의 목표는 평균적으로 모델의 예측이
데이터에서 관측된 실제 가격에 가능한 한 가깝게 들어맞도록
가중치 $\mathbf{w}$와 편향 $b$를 선택하는 것입니다.


몇 개의 특징만 있는 데이터셋에 초점을 맞추는 것이 일반적인 분야에서는,
:eqref:`eq_price-area`에서처럼
모델을 명시적으로 길게 풀어 표현하는 것이 일반적입니다.
머신러닝에서는 보통 고차원 데이터셋을 다루기 때문에,
간결한 선형대수 표기를 사용하는 것이 더 편리합니다.
입력이 $d$개의 특징으로 이루어져 있을 때,
각각에 인덱스($1$부터 $d$까지)를 부여하고
예측 $\hat{y}$를 다음과 같이 표현할 수 있습니다
(일반적으로 "hat" 기호는 추정값을 나타냅니다).

$$\hat{y} = w_1  x_1 + \cdots + w_d  x_d + b.$$

모든 특징을 벡터 $\mathbf{x} \in \mathbb{R}^d$에,
모든 가중치를 벡터 $\mathbf{w} \in \mathbb{R}^d$에 모으면,
$\mathbf{w}$와 $\mathbf{x}$의 내적을 사용해
저희 모델을 간결하게 표현할 수 있습니다.

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

:eqref:`eq_linreg-y`에서 벡터 $\mathbf{x}$는
하나의 예제의 특징에 해당합니다.
저희는 $n$개의 예제로 구성된 전체 데이터셋의 특징을
*설계 행렬(design matrix)* $\mathbf{X} \in \mathbb{R}^{n \times d}$로
참조하는 것이 종종 편리합니다.
여기서 $\mathbf{X}$는 모든 예제에 대해 하나의 행과
모든 특징에 대해 하나의 열을 가집니다.
특징들의 모음 $\mathbf{X}$에 대해,
예측값 $\hat{\mathbf{y}} \in \mathbb{R}^n$은
행렬--벡터 곱으로 표현할 수 있습니다.

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$
:eqlabel:`eq_linreg-y-vec`

여기서 합산 과정에 브로드캐스팅(:numref:`subsec_broadcasting`)이 적용됩니다.
훈련 데이터셋의 특징 $\mathbf{X}$와
대응하는 (알려진) 레이블 $\mathbf{y}$가 주어졌을 때,
선형 회귀의 목표는 $\mathbf{X}$와 같은 분포에서
샘플링된 새로운 데이터 예제의 특징이 주어졌을 때
그 예제의 레이블이 (기대값 측면에서) 가장 작은 오차로 예측되도록
가중치 벡터 $\mathbf{w}$와 편향 항 $b$를 찾는 것입니다.

$\mathbf{x}$가 주어졌을 때 $y$를 예측하는 최선의 모델이
선형이라고 믿는다고 하더라도,
모든 $1 \leq i \leq n$에 대해 $y^{(i)}$가
$\mathbf{w}^\top \mathbf{x}^{(i)}+b$와 정확히 일치하는
$n$개 예제의 실제 데이터셋을 찾을 수 있다고 기대해서는 안 됩니다.
예를 들어, 특징 $\mathbf{X}$와 레이블 $\mathbf{y}$를 관측하는 데
어떤 기기를 사용하든 약간의 측정 오차가 있을 수 있습니다.
따라서 기저 관계가 선형이라고 확신하는 경우에도,
그러한 오차를 설명하기 위해 잡음 항을 포함시킬 것입니다.

최적의 *매개변수(parameter)*(또는 *모델 매개변수(model parameter)*)
$\mathbf{w}$와 $b$를 찾아 나서기 전에,
두 가지가 더 필요합니다.
(i) 주어진 모델의 품질을 측정하는 척도,
(ii) 품질을 개선하기 위해 모델을 갱신하는 절차입니다.

### 손실 함수
:label:`subsec_linear-regression-loss-function`

자연스럽게, 모델을 데이터에 맞추려면
어떤 *적합도(fitness)*(또는 동등하게 *부적합도(unfitness)*) 척도에
합의해야 합니다.
*손실 함수(loss function)*는 타깃의 *실제* 값과 *예측* 값 사이의
거리를 정량화합니다.
손실은 보통 작은 값일수록 좋은 음이 아닌 수이며,
완벽한 예측은 0의 손실을 낳습니다.
회귀 문제에서 가장 일반적인 손실 함수는 제곱 오차입니다.
예제 $i$에 대한 예측이 $\hat{y}^{(i)}$이고
대응하는 실제 레이블이 $y^{(i)}$일 때,
*제곱 오차(squared error)*는 다음과 같이 주어집니다.

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

상수 $\frac{1}{2}$는 실질적인 차이를 만들지는 않지만,
손실의 도함수를 취할 때 상쇄되기 때문에
표기상 편리한 것으로 드러납니다.
훈련 데이터셋은 저희에게 주어진 것이고
따라서 저희가 통제할 수 있는 것이 아니므로,
경험적 오차(empirical error)는 모델 매개변수만의 함수입니다.
:numref:`fig_fit_linreg`에서는 1차원 입력을 가진 문제에서
선형 회귀 모델의 적합 결과를 시각화합니다.

![1차원 데이터에 선형 회귀 모델을 적합한 결과.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

추정값 $\hat{y}^{(i)}$와 타깃 $y^{(i)}$ 사이의 큰 차이는
손실의 이차 형태로 인해 더 큰 손실 기여로 이어진다는 점에
유의하세요(이러한 이차성은 양날의 검이 될 수 있습니다.
모델이 큰 오차를 피하도록 장려하기는 하지만,
이상치 데이터에 과도하게 민감하게 만들 수도 있습니다).
$n$개 예제로 이루어진 전체 데이터셋에서 모델의 품질을 측정하려면,
단순히 훈련 세트의 손실을 평균(또는 동등하게 합산)합니다.

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

모델을 훈련할 때, 저희는 모든 훈련 예제 전반에 걸친
총 손실을 최소화하는 매개변수 ($\mathbf{w}^*, b^*$)를 찾습니다.

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### 해석적 해

저희가 다룰 대부분의 모델과 달리,
선형 회귀는 놀랍도록 쉬운 최적화 문제를 제공합니다.
특히, 다음과 같이 간단한 공식을 적용하여
최적 매개변수(훈련 데이터에서 평가된)를
해석적으로 찾을 수 있습니다.
먼저, 모두 1로 구성된 열을 설계 행렬에 추가함으로써
편향 $b$를 매개변수 $\mathbf{w}$에 통합할 수 있습니다.
그러면 저희의 예측 문제는 $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$를 최소화하는 것이 됩니다.
설계 행렬 $\mathbf{X}$가 풀랭크(full rank)인 한
(어떤 특징도 다른 특징들에 선형 종속이 아닌 한),
손실 곡면 위에는 단 하나의 임계점만 존재할 것이며,
이것이 전체 도메인에서의 손실의 최솟값에 해당합니다.
$\mathbf{w}$에 대한 손실의 도함수를 취하고
이를 0으로 두면 다음과 같습니다.

$$\begin{aligned}
    \partial_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 =
    2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0
    \textrm{ and hence }
    \mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}.
\end{aligned}$$

$\mathbf{w}$에 대해 풀면 최적화 문제의 최적해가 제공됩니다.
이 해

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$$

은 $\mathbf X^\top \mathbf X$ 행렬이 가역일 때,
즉 설계 행렬의 열들이 선형 독립일 때만
유일하다는 점에 유의하세요 :cite:`Golub.Van-Loan.1996`.



선형 회귀 같은 단순한 문제는
해석적 해를 가질 수 있지만,
이러한 행운에 익숙해져서는 안 됩니다.
해석적 해는 멋진 수학적 분석을 가능하게 하지만,
해석적 해의 존재라는 요건은 너무나 제한적이어서
딥러닝의 흥미로운 측면 대부분을 배제하게 됩니다.

### 미니배치 확률적 경사 하강법

다행히도 모델을 해석적으로 풀 수 없는 경우에도,
실제로는 모델을 효과적으로 훈련할 수 있는 경우가 많습니다.
나아가, 많은 작업에서 이러한 최적화하기 어려운 모델들이
훨씬 더 나은 것으로 드러나기 때문에,
이들을 훈련하는 방법을 알아내는 것은
충분히 그 수고를 들일 만한 가치가 있습니다.

거의 모든 딥러닝 모델을 최적화하기 위한 핵심 기법으로,
이 책 전반에 걸쳐 활용할 것은
손실 함수를 점진적으로 낮추는 방향으로 매개변수를 갱신하여
반복적으로 오차를 줄이는 것입니다.
이 알고리즘을 *경사 하강법(gradient descent)*이라고 합니다.

경사 하강법의 가장 단순한 적용은
데이터셋의 모든 단일 예제에서 계산된 손실의 평균인
손실 함수의 도함수를 취하는 것입니다.
실제로 이것은 매우 느릴 수 있습니다.
업데이트 단계가 매우 강력하다 하더라도
한 번의 업데이트를 하기 전에 전체 데이터셋을 훑어야 합니다 :cite:`Liu.Nocedal.1989`.
설상가상으로 훈련 데이터에 중복이 많다면,
전체 업데이트의 이점은 제한적입니다.

다른 극단은 한 번에 단일 예제만 고려하여
한 번에 하나의 관측치를 기반으로 업데이트 단계를 취하는 것입니다.
그 결과로 나오는 알고리즘인 *확률적 경사 하강법(stochastic gradient descent, SGD)*은
대규모 데이터셋에서도 효과적인 전략이 될 수 있습니다 :cite:`Bottou.2010`.
불행히도 SGD는 계산적, 통계적으로 모두 단점이 있습니다.
한 가지 문제는 프로세서가 주 메모리에서 프로세서 캐시로
데이터를 옮기는 것보다 숫자를 곱하고 더하는 데
훨씬 빠르다는 사실에서 비롯됩니다.
대응하는 수의 벡터--벡터 연산을 수행하는 것보다
행렬--벡터 곱셈을 수행하는 것이
최대 한 자릿수 더 효율적일 수 있습니다.
이는 한 번에 하나의 샘플을 처리하는 것이
전체 배치에 비해 훨씬 오래 걸릴 수 있음을 의미합니다.
두 번째 문제는 배치 정규화(:numref:`sec_batch_norm`에서 설명될 예정)와 같은
일부 층이 한 번에 하나 이상의 관측치에 접근할 수 있을 때만
잘 동작한다는 것입니다.

두 문제 모두의 해법은 중간 전략을 선택하는 것입니다.
전체 배치나 한 번에 하나의 샘플만 취하는 대신,
관측치의 *미니배치(minibatch)*를 취합니다 :cite:`Li.Zhang.Chen.ea.2014`.
미니배치의 구체적인 크기 선택은
메모리 양, 가속기 수, 층의 선택, 전체 데이터셋 크기 등
많은 요인에 따라 달라집니다.
그럼에도 불구하고, 32에서 256 사이의 수,
가급적 $2$의 큰 거듭제곱의 배수가 좋은 출발점입니다.
이것이 저희를 *미니배치 확률적 경사 하강법(minibatch stochastic gradient descent)*으로 이끕니다.

가장 기본적인 형태에서, 각 반복 $t$마다
먼저 고정된 수 $|\mathcal{B}|$개의 훈련 예제로 구성된
미니배치 $\mathcal{B}_t$를 무작위로 샘플링합니다.
그런 다음 미니배치에 대한 평균 손실의 모델 매개변수에 대한
도함수(경사)를 계산합니다.
마지막으로, 경사에 *학습률(learning rate)*이라고 부르는
미리 정해진 작은 양수 값 $\eta$를 곱하고,
그 결과 항을 현재 매개변수 값에서 뺍니다.
업데이트를 다음과 같이 표현할 수 있습니다.

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

요약하면, 미니배치 SGD는 다음과 같이 진행됩니다.
(i) 일반적으로 무작위로 모델 매개변수의 값을 초기화한다.
(ii) 데이터에서 반복적으로 무작위 미니배치를 샘플링하면서
음의 경사 방향으로 매개변수를 갱신한다.
이차 손실과 아핀 변환의 경우,
이는 다음과 같은 닫힌 형태의 전개를 가집니다.

$$\begin{aligned} \mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_b l^{(i)}(\mathbf{w}, b) &&  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

미니배치 $\mathcal{B}$를 고르기 때문에
그 크기 $|\mathcal{B}|$로 정규화해야 합니다.
미니배치 크기와 학습률은 자주 사용자가 정의합니다.
훈련 루프에서 업데이트되지 않는 그러한 조정 가능한 매개변수를
*하이퍼파라미터(hyperparameter)*라고 부릅니다.
이들은 베이지안 최적화 :cite:`Frazier.2018` 같은
여러 기법을 통해 자동으로 조정될 수 있습니다.
결국 해의 품질은 일반적으로 별도의
*검증 데이터셋(validation dataset)*(또는 *검증 세트(validation set)*)에서 평가됩니다.

미리 정해진 횟수만큼 반복하여 훈련한 후
(또는 다른 어떤 정지 기준이 충족될 때까지),
저희는 추정된 모델 매개변수를 기록하며,
이를 $\hat{\mathbf{w}}, \hat{b}$로 표시합니다.
저희 함수가 진정으로 선형이고 잡음이 없더라도,
이 매개변수들이 손실의 정확한 최소화 지점이 아니며
결정론적이지도 않다는 점에 유의하세요.
알고리즘은 최소화 지점을 향해 천천히 수렴하지만
일반적으로 유한한 단계 안에서 정확히 그것들을 찾지는 못합니다.
게다가 매개변수 갱신에 사용되는 미니배치 $\mathcal{B}$는
무작위로 선택됩니다.
이는 결정론을 깨뜨립니다.

선형 회귀는 마침 전역 최솟값을 가진 학습 문제이기도 합니다
($\mathbf{X}$가 풀랭크일 때마다, 또는 동등하게
$\mathbf{X}^\top \mathbf{X}$가 가역일 때마다).
그러나 심층 신경망의 손실 곡면은 많은 안장점과 최솟값을 포함합니다.
다행히도 저희는 일반적으로 매개변수의 정확한 집합을 찾는 것이 아니라,
정확한 예측을 (따라서 낮은 손실을) 이끌어내는
어떤 매개변수 집합이든 찾으면 됩니다.
실제로 딥러닝 실무자들은
*훈련 세트에서* 손실을 최소화하는 매개변수를
찾는 데에 거의 어려움을 겪지 않습니다
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018,Frankle.Carbin.2018`.
더 만만찮은 과제는 이전에 본 적 없는 데이터에서
정확한 예측으로 이어지는 매개변수를 찾는 것이며,
이는 *일반화(generalization)*라고 부르는 과제입니다.
이러한 주제들은 책 전반에 걸쳐 다시 다룰 것입니다.

### 예측

모델 $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$가 주어졌을 때,
이제 새로운 예제에 대한 *예측(prediction)*을 할 수 있습니다.
예를 들어, 면적 $x_1$과 연식 $x_2$가 주어진,
이전에 본 적 없는 주택의 판매 가격을 예측할 수 있습니다.
딥러닝 실무자들은 예측 단계를 *추론(inference)*이라고
부르는 경향이 있지만, 이는 다소 잘못된 표현입니다.
(*추론*은 매개변수의 값과 본 적 없는 인스턴스에 대한 가능한 레이블 모두를 포함하여,
증거를 바탕으로 도달한 어떤 결론이든 폭넓게 지칭합니다.)
오히려 통계 문헌에서 *추론*은 매개변수 추론을 더 자주 가리키며,
이러한 용어의 중복은 딥러닝 실무자가 통계학자와 이야기할 때
불필요한 혼란을 야기합니다.
이어지는 내용에서는 가능한 한 *예측*이라는 용어를 고수할 것입니다.



## 속도를 위한 벡터화

모델을 훈련할 때, 저희는 일반적으로
예제의 전체 미니배치를 동시에 처리하고자 합니다.
이를 효율적으로 수행하려면 (**저희가**) (~~해야 합니다~~)
(**Python에서 비용이 많이 드는 for 루프를 작성하기보다는
계산을 벡터화하고 빠른 선형대수 라이브러리를 활용해야 합니다.**)

이것이 왜 그렇게 중요한지 보기 위해,
(**벡터를 더하는 두 가지 방법을 고려해 봅시다.**)
시작하기 위해, 모두 1로 채워진
두 개의 10,000차원 벡터를 인스턴스화합니다.
첫 번째 방법에서는 Python의 for 루프로 벡터를 순회합니다.
두 번째에서는 `+`를 한 번 호출하는 것에 의존합니다.

```{.python .input}
%%tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

이제 작업량을 벤치마크할 수 있습니다.
먼저, [**for 루프를 사용해 한 번에 한 좌표씩 더해 봅니다.**]

```{.python .input}
%%tab mxnet, pytorch
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab tensorflow
c = tf.Variable(d2l.zeros(n))
t = time.time()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab jax
# JAX arrays are immutable, meaning that once created their contents
# cannot be changed. For updating individual elements, JAX provides
# an indexed update syntax that returns an updated copy
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c = c.at[i].set(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

(**대안으로, 원소별 합을 계산하기 위해 재정의된 `+` 연산자에 의존합니다.**)

```{.python .input}
%%tab all
t = time.time()
d = a + b
f'{time.time() - t:.5f} sec'
```

두 번째 방법은 첫 번째보다 극적으로 빠릅니다.
코드를 벡터화하면 종종 자릿수 수준의 속도 향상을 얻습니다.
게다가 더 많은 수학을 라이브러리에 떠넘김으로써
저희가 직접 작성해야 하는 계산이 줄어들고,
오류 가능성을 줄이며 코드의 이식성을 높입니다.


## 정규 분포와 제곱 손실
:label:`subsec_normal_distribution_and_squared_loss`

지금까지 저희는 제곱 손실 목적 함수에 대한
다소 기능적인 동기를 제시했습니다.
기저 패턴이 진정으로 선형일 때 최적 매개변수가
조건부 기대값 $E[Y\mid X]$를 반환하며,
손실은 이상치에 큰 페널티를 부여한다는 것이었습니다.
잡음의 분포에 대해 확률적 가정을 함으로써
제곱 손실 목적 함수에 대한
더 형식적인 동기를 제공할 수도 있습니다.

선형 회귀는 19세기 전환기에 발명되었습니다.
가우스와 르장드르 중 누가 먼저 그 아이디어를 떠올렸는지에 대해서는
오랫동안 논쟁이 있었지만,
정규 분포(또한 *가우시안*이라고도 불림)를 발견한 것은
가우스이기도 했습니다.
정규 분포와 제곱 손실을 사용한 선형 회귀가
공통의 기원 그 이상의 더 깊은 연결을
공유한다는 사실이 밝혀졌습니다.

먼저, 평균 $\mu$와 분산 $\sigma^2$(표준 편차 $\sigma$)을 가진
정규 분포는 다음과 같이 주어진다는 것을 떠올려 봅시다.

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

아래에서 [**정규 분포를 계산하는 함수를 정의합니다**].

```{.python .input}
%%tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    if tab.selected('jax'):
        return p * jnp.exp(-0.5 * (x - mu)**2 / sigma**2)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
```

이제 (**정규 분포를 시각화**)할 수 있습니다.

```{.python .input}
%%tab mxnet
# Use NumPy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.python .input}

%%tab pytorch, tensorflow, jax
if tab.selected('jax'):
    # Use JAX NumPy for visualization
    x = jnp.arange(-7, 7, 0.01)
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    # Use NumPy again for visualization
    x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

평균을 바꾸면 $x$축을 따라 분포가 이동하는 것에 해당하고,
분산을 늘리면 분포가 펼쳐지면서
정점이 낮아진다는 점에 유의하세요.

제곱 손실을 사용한 선형 회귀를 동기 부여하는 한 가지 방법은
관측치가 잡음이 섞인 측정에서 발생하며,
잡음 $\epsilon$이 정규 분포 $\mathcal{N}(0, \sigma^2)$를
따른다고 가정하는 것입니다.

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \textrm{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

따라서 이제 주어진 $\mathbf{x}$에 대해 특정 $y$를 관측할
*가능도(likelihood)*를 다음과 같이 적을 수 있습니다.

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

이처럼 가능도는 인수분해됩니다.
*최대 가능도의 원리(principle of maximum likelihood)*에 따르면,
매개변수 $\mathbf{w}$와 $b$의 최선의 값은
전체 데이터셋의 *가능도*를 최대화하는 값입니다.

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)} \mid \mathbf{x}^{(i)}).$$

이 등식은 모든 쌍 $(\mathbf{x}^{(i)}, y^{(i)})$가
서로 독립적으로 추출되었기 때문에 성립합니다.
최대 가능도의 원리에 따라 선택된 추정량을
*최대 가능도 추정량(maximum likelihood estimator)*이라고 부릅니다.
많은 지수 함수의 곱을 최대화하는 것이 어려워 보일 수 있지만,
대신 가능도의 로그를 최대화함으로써
목적 함수를 바꾸지 않고도 상황을 크게 단순화할 수 있습니다.
역사적인 이유로 최적화는 최대화보다는
최소화로 표현되는 것이 더 흔합니다.
따라서 아무것도 바꾸지 않고도
*음의 로그 가능도(negative log-likelihood)*를 *최소화*할 수 있으며,
이는 다음과 같이 표현됩니다.

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

$\sigma$가 고정되어 있다고 가정하면,
$\mathbf{w}$나 $b$에 의존하지 않기 때문에
첫 번째 항은 무시할 수 있습니다.
두 번째 항은 승법 상수 $\frac{1}{\sigma^2}$를 제외하면
앞서 소개한 제곱 오차 손실과 동일합니다.
다행히 해는 $\sigma$에도 의존하지 않습니다.
이로부터 평균 제곱 오차를 최소화하는 것이
가산적 가우시안 잡음 가정 하의 선형 모델에 대한
최대 가능도 추정과 동등하다는 결론이 따라옵니다.


## 신경망으로서의 선형 회귀

선형 모델은 이 책에서 소개할 많은 복잡한 신경망을
표현할 만큼 충분히 풍부하지는 않지만,
(인공) 신경망은 모든 특징이 입력 뉴런으로 표현되고
그 모두가 출력에 직접 연결된 신경망으로서
선형 모델을 포섭할 만큼 충분히 풍부합니다.

:numref:`fig_single_neuron`은 선형 회귀를
신경망으로 묘사합니다.
이 다이어그램은 각 입력이 출력에 어떻게 연결되어 있는지와 같은
연결 패턴을 강조하지만,
가중치나 편향이 취하는 구체적인 값은 강조하지 않습니다.

![선형 회귀는 단일 층 신경망입니다.](../img/singleneuron.svg)
:label:`fig_single_neuron`

입력은 $x_1, \ldots, x_d$입니다.
$d$를 입력층의 *입력 수* 또는
*특징 차원성(feature dimensionality)*이라고 합니다.
신경망의 출력은 $o_1$입니다.
저희는 단일 수치 값을 예측하려고 하기 때문에,
출력 뉴런은 하나뿐입니다.
입력 값은 모두 *주어진다*는 점에 유의하세요.
*계산되는* 뉴런은 단 하나뿐입니다.
요약하면, 선형 회귀를 단일 층의 완전 연결 신경망으로
생각할 수 있습니다.
이후 장에서는 훨씬 더 많은 층을 가진
신경망을 만나게 될 것입니다.

### 생물학

선형 회귀가 계산 신경과학보다 앞서기 때문에,
선형 회귀를 신경망의 관점에서 설명하는 것이
시대착오적으로 보일 수 있습니다.
그럼에도 불구하고, 인공 뉴런 모델을 개발하기 시작했을 때
사이버네틱스 학자이자 신경생리학자인
워런 매컬록과 월터 피츠에게는
자연스러운 출발점이었습니다.
:numref:`fig_Neuron`에 나오는 만화 같은
생물학적 뉴런 그림을 살펴보세요.
이는 *수상돌기(dendrites)*(입력 단자),
*핵(nucleus)*(CPU), *축삭(axon)*(출력 와이어),
*축삭 말단(axon terminals)*(출력 단자)으로 구성되어 있으며,
*시냅스(synapse)*를 통해 다른 뉴런들과의 연결을 가능하게 합니다.

![실제 뉴런(출처: 미국 국립 암 연구소의 감시, 역학 및 최종 결과(SEER) 프로그램의 "Anatomy and Physiology").](../img/neuron.svg)
:label:`fig_Neuron`

다른 뉴런(또는 환경 센서)에서 도착하는 정보 $x_i$가
수상돌기에서 수신됩니다.
특히 그 정보는 *시냅스 가중치(synaptic weight)* $w_i$에 의해 가중되어,
예를 들어 $x_i w_i$ 곱을 통해 활성화 또는 억제 같은
입력의 효과를 결정합니다.
여러 출처에서 도착하는 가중 입력은
핵에서 가중합 $y = \sum_i x_i w_i + b$로 집계되며,
함수 $\sigma(y)$를 통한 비선형 후처리를 받기도 합니다.
이 정보는 그런 다음 축삭을 통해 축삭 말단으로 보내져,
목적지(예를 들어 근육 같은 작동체)에 도달하거나
다른 뉴런의 수상돌기를 통해 그 뉴런으로 입력됩니다.

확실히, 그러한 많은 단위들이 올바른 연결성과 학습 알고리즘만 갖춘다면
하나의 뉴런만으로 표현할 수 있는 것보다 훨씬 흥미롭고 복잡한
거동을 만들어내기 위해 결합될 수 있다는 고차원적 아이디어는
실제 생물학적 신경계에 대한 저희의 연구에서 비롯됩니다.
동시에, 오늘날 딥러닝의 대부분의 연구는
훨씬 더 광범위한 출처에서 영감을 얻습니다.
저희는 비행기가 새에서 *영감*을 받았을 수는 있지만
조류학이 수 세기 동안 항공학 혁신의
주요 추동력이었던 것은 아니라고 지적한
:citet:`Russell.Norvig.2016`를 인용합니다.
마찬가지로 요즘 딥러닝의 영감은
수학, 언어학, 심리학, 통계학, 컴퓨터과학 등 수많은 분야로부터
동등하거나 더 큰 비중으로 옵니다.

## 요약

이 절에서 저희는 전통적인 선형 회귀를 소개했는데,
이는 선형 함수의 매개변수가 훈련 세트에서
제곱 손실을 최소화하도록 선택되는 방법입니다.
또한 몇 가지 실용적인 고려 사항과
선형성 및 가우시안 잡음 가정 하의
최대 가능도 추정으로서의 선형 회귀 해석을 통해
이러한 목적 함수 선택에 동기를 부여했습니다.
계산적 고려 사항과 통계와의 연결을 논의한 후,
저희는 그러한 선형 모델이 입력이 출력에 직접 연결된
단순한 신경망으로 어떻게 표현될 수 있는지 보여주었습니다.
저희는 곧 선형 모델을 완전히 넘어설 것이지만,
선형 모델은 저희의 모든 모델이 필요로 하는
대부분의 구성 요소를 소개하기에 충분합니다.
매개변수 형태, 미분 가능한 목적 함수,
미니배치 확률적 경사 하강법을 통한 최적화,
그리고 궁극적으로 이전에 본 적 없는 데이터에서의 평가가 그것입니다.



## 연습문제

1. 어떤 데이터 $x_1, \ldots, x_n \in \mathbb{R}$이 있다고 가정합니다. 저희의 목표는 $\sum_i (x_i - b)^2$가 최소화되는 상수 $b$를 찾는 것입니다.
    1. $b$의 최적 값에 대한 해석적 해를 찾으세요.
    1. 이 문제와 그 해는 정규 분포와 어떻게 관련됩니까?
    1. 손실을 $\sum_i (x_i - b)^2$에서 $\sum_i |x_i-b|$로 바꾸면 어떻게 됩니까? $b$에 대한 최적 해를 찾을 수 있습니까?
1. $\mathbf{x}^\top \mathbf{w} + b$로 표현될 수 있는 아핀 함수가 $(\mathbf{x}, 1)$에 대한 선형 함수와 동등함을 증명하세요.
1. $\mathbf{x}$의 이차 함수, 즉 $f(\mathbf{x}) = b + \sum_i w_i x_i + \sum_{j \leq i} w_{ij} x_{i} x_{j}$를 찾고자 한다고 가정합니다. 이를 심층 신경망에서 어떻게 정식화하겠습니까?
1. 선형 회귀 문제가 풀릴 수 있는 조건 중 하나가 설계 행렬 $\mathbf{X}^\top \mathbf{X}$가 풀랭크여야 한다는 것이었음을 떠올려 보세요.
    1. 그렇지 않은 경우 어떻게 됩니까?
    1. 어떻게 해결할 수 있습니까? $\mathbf{X}$의 모든 항목에 좌표별로 독립인 작은 양의 가우시안 잡음을 추가하면 어떻게 됩니까?
    1. 이 경우 설계 행렬 $\mathbf{X}^\top \mathbf{X}$의 기대값은 무엇입니까?
    1. $\mathbf{X}^\top \mathbf{X}$가 풀랭크가 아닐 때 확률적 경사 하강법에는 어떤 일이 일어납니까?
1. 가산 잡음 $\epsilon$을 지배하는 잡음 모델이 지수 분포라고 가정합니다. 즉, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$입니다.
    1. 모델 하의 데이터에 대한 음의 로그 가능도 $-\log P(\mathbf y \mid \mathbf X)$를 작성하세요.
    1. 닫힌 형태의 해를 찾을 수 있습니까?
    1. 이 문제를 풀기 위한 미니배치 확률적 경사 하강법 알고리즘을 제안하세요. 무엇이 잘못될 수 있습니까(힌트: 매개변수를 계속 갱신함에 따라 정류점 근처에서 어떤 일이 일어납니까)? 이를 해결할 수 있습니까?
1. 두 개의 선형 층을 조합하여 두 층짜리 신경망을 설계하고자 한다고 가정합니다. 즉, 첫 번째 층의 출력이 두 번째 층의 입력이 됩니다. 그러한 단순한 합성이 왜 작동하지 않겠습니까?
1. 주택이나 주식 가격의 현실적인 가격 추정에 회귀를 사용하려면 어떻게 됩니까?
    1. 가산적 가우시안 잡음 가정이 적절하지 않음을 보이세요. 힌트: 음의 가격을 가질 수 있습니까? 변동은 어떻습니까?
    1. 가격의 로그에 대한 회귀, 즉 $y = \log \textrm{price}$가 훨씬 더 나은 이유는 무엇입니까?
    1. 페니 주식, 즉 매우 낮은 가격의 주식을 다룰 때 무엇을 걱정해야 합니까? 힌트: 가능한 모든 가격에 거래할 수 있습니까? 이것이 저가 주식에 더 큰 문제인 이유는 무엇입니까? 더 많은 정보는 옵션 가격 결정에 관한 유명한 블랙(Black)--숄즈(Scholes) 모델 :cite:`Black.Scholes.1973`을 검토해 보세요.
1. 식료품점에서 판매되는 사과의 *수*를 추정하기 위해 회귀를 사용하려고 한다고 가정합니다.
    1. 가우시안 가산 잡음 모델의 문제는 무엇입니까? 힌트: 당신은 사과를 팔지, 석유를 파는 것이 아닙니다.
    1. [푸아송 분포](https://en.wikipedia.org/wiki/Poisson_distribution)는 카운트에 대한 분포를 나타냅니다. 이는 $p(k \mid \lambda) = \lambda^k e^{-\lambda}/k!$로 주어집니다. 여기서 $\lambda$는 율 함수이고 $k$는 보게 되는 사건의 수입니다. $\lambda$가 카운트 $k$의 기대값임을 증명하세요.
    1. 푸아송 분포와 관련된 손실 함수를 설계하세요.
    1. 대신 $\log \lambda$를 추정하기 위한 손실 함수를 설계하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
