```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 배치 정규화
:label:`sec_batch_norm`

심층 신경망을 학습시키는 것은 어렵습니다.
합리적인 시간 안에 이들을 수렴하게 만드는 것은 까다로울 수 있습니다.
이 절에서, 저희는 심층 네트워크의 수렴을 일관되게 가속화하는 인기 있고 효과적인 기법인
*배치 정규화*를 설명합니다 :cite:`Ioffe.Szegedy.2015`.
잔차 블록과 함께(나중에 :numref:`sec_resnet`에서 다룸), 배치 정규화는
실무자들이 100층 이상의 네트워크를 일상적으로 학습할 수 있게 만들었습니다.
배치 정규화의 부차적인 (우연한) 이점은 그 내재된 정규화에 있습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## 심층 네트워크 학습

데이터로 작업할 때, 저희는 종종 학습 전에 전처리를 합니다.
데이터 전처리에 관한 선택은 종종 최종 결과에 엄청난 차이를 만듭니다.
주택 가격 예측에 대한 MLP의 적용(:numref:`sec_kaggle_house`)을 떠올려 보세요.
실제 데이터로 작업할 때 저희의 첫 단계는
여러 관측치에 걸쳐 입력 특성을 평균 $\boldsymbol{\mu} = 0$과 단위 분산 $\boldsymbol{\Sigma} = \boldsymbol{1}$을 갖도록 표준화하는 것이었습니다 :cite:`friedman1987exploratory`. 후자는 종종 대각선이 단위가 되도록(즉, $\Sigma_{ii} = 1$) 재스케일됩니다.
또 다른 전략은 벡터를 단위 길이로 재스케일하는 것이며, 가능하면 *관측당* 평균을 0으로 합니다.
이는 예를 들어 공간 센서 데이터에 잘 작동할 수 있습니다. 이러한 전처리 기법과 다른 많은 것들은
추정 문제를 잘 제어하는 데 유익합니다.
특성 선택 및 추출에 대한 리뷰는 예를 들어 :citet:`guyon2008feature`의 논문을 참조하세요.
벡터를 표준화하는 것은 또한 그 위에서 작용하는 함수의 함수 복잡도를 제약하는 좋은 부수 효과를 가집니다. 예를 들어, 서포트 벡터 머신의 유명한 radius-margin 경계 :cite:`Vapnik95`와 퍼셉트론 수렴 정리 :cite:`Novikoff62`는 유한 노름의 입력에 의존합니다.

직관적으로, 이 표준화는 사전에 파라미터를 유사한 스케일에 놓기 때문에
저희의 옵티마이저와 잘 어울립니다.
이와 같이, 심층 네트워크 *내부*에 해당하는 정규화 단계가
유익하지 않을지 묻는 것은 자연스러운 일입니다. 이는 배치 정규화 :cite:`Ioffe.Szegedy.2015`의 발명으로 이어진 추론은 아니지만, 이를 그리고 그 사촌인 레이어 정규화 :cite:`Ba.Kiros.Hinton.2016`을 통합된 프레임워크 내에서 이해하는 유용한 방법입니다.

둘째, 일반적인 MLP나 CNN의 경우, 저희가 학습할 때,
중간 층의 변수
(예: MLP의 아핀 변환 출력)는
입력에서 출력까지의 층을 따라, 같은 층의 단위 간에,
그리고 모델 파라미터에 대한 저희의 업데이트로 인한 시간에 따라
크게 다른 크기의 값을 가질 수 있습니다.
배치 정규화의 발명자들은 이러한 변수의 분포에서의 이러한 표류가
네트워크의 수렴을 방해할 수 있다고 비공식적으로 가정했습니다.
직관적으로, 한 층이
다른 층의 100배인 변수 활성화를 가진다면,
이는 학습률에서의 보상적 조정을 필요하게 할 수 있다고 추측할 수 있습니다. AdaGrad :cite:`Duchi.Hazan.Singer.2011`, Adam :cite:`Kingma.Ba.2014`, Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`, 또는 Distributed Shampoo :cite:`anil2020scalable`와 같은
적응형 솔버는 예를 들어 이차 방법의 측면을 추가하는 등 최적화의 관점에서 이를 해결하려고 합니다.
대안은 단순히 적응형 정규화를 통해 문제가 발생하는 것을 방지하는 것입니다.

셋째, 더 깊은 네트워크는 복잡하고 과적합에 더 취약한 경향이 있습니다.
이는 정규화가 더 중요해진다는 것을 의미합니다. 정규화를 위한 일반적인 기법은 노이즈
주입입니다. 이는 오랫동안 알려져 있었습니다, 예를 들어 입력에 대한 노이즈 주입과 관련하여
:cite:`Bishop.1995`. 또한 이는 :numref:`sec_dropout`의 드롭아웃의 기초를 형성합니다. 알고 보니, 꽤 우연히, 배치 정규화는 세 가지 이점 모두를 전달합니다: 전처리, 수치적 안정성, 그리고 정규화.

배치 정규화는 개별 층이나, 선택적으로 모든 층에 적용됩니다:
각 학습 반복에서,
저희는 먼저 입력(배치 정규화의)을
평균을 빼고
표준 편차로 나누어 정규화하며,
이 둘은 모두 현재 미니배치의 통계를 기반으로 추정됩니다.
다음으로, 잃어버린 자유도를 복구하기 위해 스케일 계수와 오프셋을 적용합니다.
*배치* 통계에 기반한 이 *정규화* 때문에
*배치 정규화*가 그 이름을 얻습니다.

크기 1의 미니배치로 배치 정규화를 적용하려고 시도한다면,
저희는 아무것도 학습할 수 없을 것이라는 점에 주목하세요.
이는 평균을 뺀 후,
각 은닉 단위가 값 0을 갖게 되기 때문입니다.
짐작하시겠지만, 배치 정규화에 전체 절을 할애하는 만큼,
충분히 큰 미니배치로 그 접근법은 효과적이고 안정적임이 입증됩니다.
여기서 한 가지 시사점은 배치 정규화를 적용할 때,
배치 크기의 선택이
배치 정규화가 없을 때보다 훨씬 더 중요하며, 적어도
배치 크기를 조정함에 따라 적절한 보정이 필요하다는 것입니다.

미니배치를 $\mathcal{B}$로 표시하고 $\mathbf{x} \in \mathcal{B}$를
배치 정규화($\textrm{BN}$)에 대한 입력이라고 합시다. 이 경우 배치 정규화는 다음과 같이 정의됩니다:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm`에서,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$는 미니배치 $\mathcal{B}$의 표본 평균이고
$\hat{\boldsymbol{\sigma}}_\mathcal{B}$는 표본 표준 편차입니다.
표준화를 적용한 후,
결과 미니배치는
평균이 0이고 단위 분산을 가집니다.
단위 분산(다른 어떤 마법의 숫자가 아니라)의 선택은 임의적입니다. 저희는 $\mathbf{x}$와 같은 모양을 가진
요소별 *스케일 파라미터* $\boldsymbol{\gamma}$와 *시프트 파라미터* $\boldsymbol{\beta}$를 포함하여
이 자유도를 복구합니다. 둘 다 모델 학습의 일부로
학습되어야 할 파라미터입니다.

중간 층의 변수 크기는
배치 정규화가 적극적으로 이들을 주어진 평균과 크기로 중심화하고 다시 스케일링하기 때문에
학습 중에 발산할 수 없습니다($\hat{\boldsymbol{\mu}}_\mathcal{B}$와 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$를 통해).
실제 경험은 특성 재스케일링을 논의할 때 암시한 것처럼, 배치 정규화가 더 공격적인 학습률을 허용하는 것 같다고 확인합니다.
저희는 :eqref:`eq_batchnorm`의 $\hat{\boldsymbol{\mu}}_\mathcal{B}$와 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$를 다음과 같이 계산합니다:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

저희는 경험적 분산 추정치가 매우 작거나 사라질 수 있는 경우에도
0으로 나누는 것을 절대로 시도하지 않도록 분산 추정치에
작은 상수 $\epsilon > 0$을 추가합니다.
추정치 $\hat{\boldsymbol{\mu}}_\mathcal{B}$와 ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$는
평균과 분산의 노이즈가 있는 추정치를 사용하여
스케일링 문제를 상쇄합니다.
이 노이즈가 문제가 되어야 한다고 생각할 수도 있습니다.
반대로, 이는 실제로 유익합니다.

이는 딥러닝에서 반복되는 주제임이 밝혀졌습니다.
아직 이론적으로 잘 규명되지 않은 이유로,
최적화에서의 다양한 노이즈 소스는
종종 더 빠른 학습과 더 적은 과적합으로 이어집니다:
이러한 변동은 정규화의 한 형태로 작용하는 것 같습니다.
:citet:`Teye.Azizpour.Smith.2018`과 :citet:`Luo.Wang.Shao.ea.2018`은
배치 정규화의 속성을 각각 베이지안 사전 확률과 페널티에 관련시켰습니다.
특히, 이는 배치 정규화가 50(100 범위의 적당한 미니배치 크기에서 가장 잘 작동하는 이유)에 대한 수수께끼에
약간의 빛을 비춥니다.
이 특정한 미니배치 크기는 $\hat{\boldsymbol{\sigma}}$를 통한 스케일과 $\hat{\boldsymbol{\mu}}$를 통한 오프셋 모두의 측면에서 층당 "딱 맞는 양"의 노이즈를 주입하는 것 같습니다.
더 큰 미니배치는 더 안정적인 추정치로 인해 덜 정규화하는 반면, 매우 작은 미니배치는
높은 분산으로 인해 유용한 신호를 파괴합니다. 이 방향을 더 탐색하면서, 대안적인 유형의
전처리와 필터링을 고려하면 다른 효과적인 유형의 정규화로 이어질 수 있습니다.

학습된 모델을 고정시키면,
평균과 분산을 추정하기 위해
전체 데이터셋을 사용하는 것을 선호할 것이라고 생각할 수 있습니다.
학습이 완료된 후, 왜 동일한 이미지가
그것이 속하는 배치에 따라
다르게 분류되기를 원하겠습니까?
학습 중에는, 모든 데이터 예제에 대한 중간 변수가
저희가 모델을 업데이트할 때마다 변경되기 때문에
그러한 정확한 계산은 실현 불가능합니다.
그러나, 모델이 학습되면,
저희는 전체 데이터셋을 기반으로
각 층 변수의 평균과 분산을 계산할 수 있습니다.
실제로 이는 배치 정규화를 채택하는 모델의 표준 관행입니다;
따라서 배치 정규화 층은 노이즈가 학습 중에만 주입되는
:numref:`sec_dropout`의 드롭아웃 정규화의 동작과 유사하게
*학습 모드*(미니배치 통계에 의한 정규화)에서와 *예측 모드*(데이터셋 통계에 의한 정규화)에서
다르게 작동합니다.


## 배치 정규화 층

완전 연결 층과 합성곱 층에 대한 배치 정규화 구현은
약간 다릅니다.
배치 정규화와 다른 층 사이의 한 가지 주요 차이점은
전자가 한 번에 전체 미니배치에서 작동하기 때문에,
저희가 다른 층을 도입할 때 이전에 했던 것처럼 배치 차원을 단순히 무시할 수 없다는 것입니다.

### 완전 연결 층

완전 연결 층에 배치 정규화를 적용할 때,
:citet:`Ioffe.Szegedy.2015`는 그들의 원래 논문에서 아핀 변환 다음 *그리고* 비선형 활성화 함수 *앞*에 배치 정규화를 삽입했습니다. 이후의 응용은 활성화 함수 *바로 다음에* 배치 정규화를 삽입하는 것을 실험했습니다.
완전 연결 층의 입력을 $\mathbf{x}$로,
아핀 변환을 $\mathbf{W}\mathbf{x} + \mathbf{b}$(가중치 파라미터 $\mathbf{W}$와 편향 파라미터 $\mathbf{b}$를 가짐)로,
그리고 활성화 함수를 $\phi$로 표시하면,
배치 정규화가 활성화된 완전 연결 층 출력 $\mathbf{h}$의 계산을 다음과 같이 표현할 수 있습니다:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

평균과 분산은 변환이 적용되는
*동일한* 미니배치에서 계산됨을 기억하세요.

### 합성곱 층

마찬가지로, 합성곱 층에서도 합성곱 후 비선형 활성화 함수 전에
배치 정규화를 적용할 수 있습니다. 완전 연결 층에서의 배치 정규화와의 주요 차이점은
저희가 *모든 위치에 걸쳐* 채널별로 연산을 적용한다는 것입니다. 이는 합성곱으로 이어진 이동
불변성에 대한 저희의 가정과 호환됩니다: 저희는 이해의 목적상 이미지 내 패턴의
특정 위치가 중요하지 않다고 가정했습니다.

저희의 미니배치가 $m$개의 예제를 포함하고
각 채널에 대해, 합성곱의 출력이 높이 $p$와 너비 $q$를 가진다고 가정합니다.
합성곱 층의 경우, 저희는 각 배치 정규화를
출력 채널당 $m \cdot p \cdot q$개의 요소에 대해 동시에 수행합니다.
따라서, 평균과 분산을 계산할 때
모든 공간 위치에서 값을 수집하고
결과적으로 주어진 채널 내에서 각 공간 위치의 값을 정규화하기 위해
동일한 평균과 분산을 적용합니다.
각 채널은 자체 스케일과 시프트 파라미터를 가지며,
둘 다 스칼라입니다.

### 레이어 정규화
:label:`subsec_layer-normalization-in-bn`

합성곱의 맥락에서 배치 정규화는 크기 1의 미니배치에 대해서도 잘 정의됨에 유의하세요:
결국, 저희는 평균을 낼 수 있는 이미지 전체의 모든 위치를 가지고 있습니다. 결과적으로,
평균과 분산은 단일 관측치 내에서도 잘 정의됩니다. 이 고려사항은
:citet:`Ba.Kiros.Hinton.2016`이 *레이어 정규화*의 개념을 도입하도록 이끌었습니다. 이는 한 번에 하나의 관측치에 적용된다는 점만 제외하면,
배치 정규화와 똑같이 작동합니다. 결과적으로 오프셋과 스케일링 인자 모두 스칼라입니다. $n$차원 벡터 $\mathbf{x}$에 대해, 레이어 정규화는 다음과 같이 주어집니다.

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

여기서 스케일링과 오프셋은 계수별로 적용되며
다음과 같이 주어집니다.

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

이전과 마찬가지로 0으로 나누는 것을 방지하기 위해 작은 오프셋 $\epsilon > 0$을 추가합니다. 레이어 정규화 사용의 주요 이점 중 하나는 발산을 방지한다는 것입니다. 결국, $\epsilon$을 무시하면, 레이어 정규화의 출력은 스케일에 독립적입니다. 즉, 어떤 $\alpha \neq 0$의 선택에 대해서도 $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$입니다. 이는 $|\alpha| \to \infty$에 대해 등식이 됩니다(근사 등식은 분산에 대한 오프셋 $\epsilon$ 때문입니다).

레이어 정규화의 또 다른 이점은 미니배치 크기에 의존하지 않는다는 것입니다. 또한 학습 또는 테스트 체제에 있는지 여부에 독립적입니다. 즉, 이는 단순히 활성화를 주어진 스케일로 표준화하는 결정론적 변환입니다. 이는 최적화에서 발산을 방지하는 데 매우 유익할 수 있습니다. 자세한 내용은 생략하고 관심 있는 독자는 원본 논문을 참고하기를 권장합니다.

### 예측 중의 배치 정규화

앞서 언급한 바와 같이, 배치 정규화는 일반적으로 학습 모드와 예측 모드에서 다르게 작동합니다.
첫째, 미니배치에서 각각을 추정함으로써 발생하는
표본 평균과 표본 분산의 노이즈는
모델을 학습시킨 후에는 더 이상 바람직하지 않습니다.
둘째, 저희는 배치별 정규화 통계를 계산하는
사치를 가질 수 없을 수도 있습니다.
예를 들어,
저희는 한 번에 하나의 예측을 하기 위해 모델을 적용해야 할 수 있습니다.

일반적으로, 학습 후, 저희는 변수 통계의 안정적인 추정치를 계산하기 위해
전체 데이터셋을 사용하고
그런 다음 예측 시점에 이를 고정합니다.
따라서, 배치 정규화는 학습 중과 테스트 시점에 다르게 작동합니다.
드롭아웃도 이러한 특성을 보인다는 것을 떠올려 보세요.

## (**처음부터 구현**)

배치 정규화가 실제로 어떻게 작동하는지 보기 위해, 저희는 아래에서 처음부터 하나를 구현합니다.

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        # `linen.Module.variables` have a `value` attribute containing the array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

이제 저희는 [**적절한 `BatchNorm` 층을 만들 수 있습니다.**]
저희의 층은 학습 과정에서 모두 업데이트될
스케일 `gamma`와 시프트 `beta`에 대한 적절한 파라미터를 유지합니다.
또한, 저희의 층은 모델 예측 중 후속 사용을 위해
평균과 분산의 이동 평균을 유지할 것입니다.

알고리즘적 세부 사항을 제쳐두고,
층 구현의 기초가 되는 설계 패턴에 주목하세요.
일반적으로, 저희는 별도의 함수에 수학을 정의합니다, 예를 들어 `batch_norm`.
그런 다음 이 기능을 사용자 정의 층에 통합하며,
그 코드는 대부분 올바른 장치 컨텍스트로 데이터 이동,
필요한 변수의 할당 및 초기화,
이동 평균(여기서는 평균과 분산에 대한) 추적 등과 같은 부기 문제를 처리합니다.
이 패턴은 수학과 보일러플레이트 코드의 깔끔한 분리를 가능하게 합니다.
또한 편의를 위해
여기서는 입력 모양을 자동으로 추론하는 것에 대해 걱정하지 않았음에 주목하세요;
따라서 저희는 특성의 수를 전체에 걸쳐 지정해야 합니다.
지금까지 모든 현대 딥러닝 프레임워크는 고수준 배치 정규화 API에서
크기와 모양의 자동 감지를 제공합니다(실제로 저희는 이것 대신 이를 사용할 것입니다).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

저희는 `momentum`을 사용하여 과거 평균 및 분산 추정치에 대한 집계를 제어했습니다. 이는 최적화의 *모멘텀* 항과는 전혀 관련이 없으므로 다소 잘못된 이름이라고 할 수 있습니다. 그럼에도 불구하고, 이는 이 항에 대해 일반적으로 채택된 이름이며 API 명명 규칙을 존중하여 저희도 코드에서 동일한 변수 이름을 사용합니다.

## [**배치 정규화를 적용한 LeNet**]

`BatchNorm`을 맥락에서 적용하는 방법을 보기 위해,
아래에서 저희는 이를 전통적인 LeNet 모델(:numref:`sec_lenet`)에 적용합니다.
배치 정규화는 합성곱 층이나 완전 연결 층 다음에,
하지만 해당 활성화 함수 전에 적용된다는 것을 떠올려 보세요.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
`BatchNorm` 층은 배치 통계(평균과 분산)를 계산해야 하므로,
Flax는 모든 미니배치로 이들을 업데이트하며 `batch_stats` 사전을 추적합니다.
`batch_stats`와 같은 컬렉션은 :numref:`oo-design-training`에 정의된
`d2l.Trainer` 클래스의 `TrainState` 객체에 속성으로 저장될 수 있으며 모델의 순방향 패스 중에,
Flax가 변이된 변수를 반환하도록 이들을 `mutable` 인수에 전달해야 합니다.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

이전과 마찬가지로, 저희는 [**Fashion-MNIST 데이터셋에서 저희 네트워크를 학습합니다**].
이 코드는 LeNet을 처음 학습시켰을 때의 것과 사실상 동일합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

첫 번째 배치 정규화 층에서 학습된 [**스케일 파라미터 `gamma`
와 시프트 파라미터 `beta`를**] 살펴봅시다.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**간결한 구현**]

저희가 방금 정의한 `BatchNorm` 클래스와 비교하여,
딥러닝 프레임워크의 고수준 API에 정의된 `BatchNorm` 클래스를 직접 사용할 수 있습니다.
코드는 차원을 올바르게 얻기 위해 추가 인수를 더 이상 제공할 필요가 없다는 점을 제외하면,
저희 위의 구현과 사실상 동일하게 보입니다.

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

아래에서, 저희는 [**저희의 모델을 학습시키기 위해 동일한 하이퍼파라미터를 사용합니다.**]
평소와 같이 고수준 API 변형은 그 코드가 C++ 또는 CUDA로 컴파일된 반면,
저희의 사용자 정의 구현은 Python에 의해 해석되어야 하기 때문에
훨씬 더 빠르게 실행된다는 점에 주목하세요.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## 논의

직관적으로, 배치 정규화는
최적화 환경을 더 부드럽게 만든다고 생각됩니다.
그러나, 저희는 심층 모델을 학습할 때 관찰되는 현상에 대해
추측에 근거한 직관과 진정한 설명을
신중하게 구별해야 합니다.
저희는 더 단순한
심층 신경망(MLP와 기존 CNN)이
애초에 왜 잘 일반화되는지조차 모른다는 점을 떠올려 보세요.
드롭아웃과 가중치 감쇠에도 불구하고,
이들은 본 적 없는 데이터로 일반화하는 능력이
훨씬 더 정교한 학습 이론적 일반화 보장이 필요한 만큼 유연합니다.

배치 정규화를 제안한 원래 논문 :cite:`Ioffe.Szegedy.2015`은, 강력하고 유용한 도구를 도입한 것 외에도,
이것이 작동하는 이유에 대한 설명을 제공했습니다:
*내부 공변량 변화*를 줄임으로써.
추측컨대 *내부 공변량 변화*로 그들이 의미했던 것은
위에 표현된 직관과 같은 것이었습니다(학습 과정 동안
변수 값의 분포가 변한다는 개념).
그러나, 이 설명에는 두 가지 문제가 있었습니다:
i) 이 표류는 *공변량 변화*와 매우 다르므로,
이름을 잘못된 명칭으로 만듭니다. 굳이 말하자면, 개념 표류에 더 가깝습니다.
ii) 이 설명은 충분히 명시되지 않은 직관을 제공하지만
*왜 정확히 이 기법이 작동하는가*라는 질문은
엄격한 설명을 기다리는 열린 질문으로 남겨둡니다.
이 책 전반에 걸쳐, 저희는 실무자들이
심층 신경망의 개발을 안내하기 위해 사용하는 직관을 전달하는 것을 목표로 합니다.
그러나, 저희는 이러한 안내 직관을
확립된 과학적 사실과 분리하는 것이
중요하다고 믿습니다.
결국, 이 자료를 마스터하고
자신의 연구 논문을 작성하기 시작할 때
기술적 주장과 직감을 구분하는 데
명확하기를 원할 것입니다.

배치 정규화의 성공에 이어,
*내부 공변량 변화* 측면에서의 그 설명은
기술 문헌과 머신러닝 연구를 제시하는 방법에 대한
더 광범위한 담론에서 토론에 반복적으로 등장했습니다.
2017 NeurIPS 컨퍼런스에서
Test of Time Award를 수상하면서 한 기억에 남는 연설에서,
Ali Rahimi는 *내부 공변량 변화*를
딥러닝의 현대적 관행을 연금술에 비유하는
주장의 초점으로 사용했습니다.
이후, 그 예는 머신러닝의 문제가 되는 추세를 개괄하는
포지션 페이퍼 :cite:`Lipton.Steinhardt.2018`에서 자세히 다시 다루어졌습니다.
다른 저자들은
배치 정규화의 성공에 대한 대안적인 설명을 제안했으며,
일부 :cite:`Santurkar.Tsipras.Ilyas.ea.2018`는
배치 정규화의 성공이 원래 논문에서 주장된 것과
어떤 면에서 반대되는 행동을 보임에도 불구하고 온다고 주장합니다.


저희는 *내부 공변량 변화*가
매년 기술 머신러닝 문헌에서 이루어지는
수천 개의 비슷하게 모호한 주장보다 더 비판받을 가치가 없다고 말합니다.
아마도, 이러한 토론의 초점으로서의 그 공명은
대상 청중에 대한 그것의 광범위한 인식 가능성에 기인할 것입니다.
배치 정규화는 거의 모든 배포된 이미지 분류기에 적용된
필수 불가결한 방법임이 입증되어,
그 기법을 도입한 논문에 수만 건의 인용을 가져다주었습니다. 그러나, 저희는 노이즈 주입을 통한 정규화,
재스케일링을 통한 가속, 그리고 마지막으로 전처리의 안내 원칙이
미래에 층과 기법의 추가 발명으로 이어질 수 있다고 추측합니다.

더 실용적인 관점에서, 배치 정규화에 대해 기억할 만한 측면이 몇 가지 있습니다:

* 모델 학습 중에, 배치 정규화는 미니배치의 평균과 표준 편차를 활용하여
  네트워크의 중간 출력을 지속적으로 조정함으로써, 신경망 전체에서 각 층의
  중간 출력 값이 더 안정적이 됩니다.
* 배치 정규화는 완전 연결 층의 경우와 합성곱 층의 경우 약간 다릅니다. 실제로,
  합성곱 층의 경우, 레이어 정규화가 때때로 대안으로 사용될 수 있습니다.
* 드롭아웃 층처럼, 배치 정규화 층은 학습 모드와 예측 모드에서
  다른 동작을 가집니다.
* 배치 정규화는 정규화와 최적화에서 수렴을 향상시키는 데 유용합니다. 대조적으로,
  내부 공변량 변화를 줄이려는 원래 동기는 유효한 설명이 아닌 것 같습니다.
* 입력 섭동에 덜 민감한 더 견고한 모델의 경우, 배치 정규화 제거를 고려하세요 :cite:`wang2022removing`.

## 연습문제

1. 배치 정규화 전에 완전 연결 층 또는 합성곱 층에서 편향 파라미터를 제거해야 하나요? 왜 그런가요?
1. 배치 정규화가 있는 LeNet과 없는 LeNet의 학습률을 비교하세요.
    1. 검증 정확도의 증가를 도표로 나타내세요.
    1. 두 경우 모두에서 최적화가 실패하기 전에 학습률을 얼마나 크게 만들 수 있나요?
1. 모든 층에서 배치 정규화가 필요한가요? 실험해 보세요.
1. 평균만 제거하거나, 대안적으로 분산만 제거하는 배치 정규화의 "라이트" 버전을
   구현하세요. 어떻게 작동하나요?
1. 파라미터 `beta`와 `gamma`를 고정하세요. 결과를 관찰하고 분석하세요.
1. 드롭아웃을 배치 정규화로 대체할 수 있나요? 동작은 어떻게 변하나요?
1. 연구 아이디어: 적용할 수 있는 다른 정규화 변환을 생각해 보세요:
    1. 확률 적분 변환을 적용할 수 있나요?
    1. 풀랭크 공분산 추정치를 사용할 수 있나요? 왜 그렇게 하지 않는 것이 좋을까요?
    1. 다른 컴팩트한 행렬 변형(블록 대각, 저-변위 랭크, Monarch 등)을 사용할 수 있나요?
    1. 희소화 압축이 정규화기로 작동하나요?
    1. 사용할 수 있는 다른 투영(예: 볼록 콘, 대칭 그룹별 변환)이 있나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18005)
:end_tab:
