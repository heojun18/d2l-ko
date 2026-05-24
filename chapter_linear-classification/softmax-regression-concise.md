```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 소프트맥스 회귀의 간결한 구현
:label:`sec_softmax_concise`



선형 회귀를 더 쉽게 구현할 수 있도록 해 준
고수준 딥러닝 프레임워크
(:numref:`sec_linear_concise` 참고)와 마찬가지로,
여기서도 그것들은 마찬가지로 편리합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
import jax
from jax import numpy as jnp
import optax
```

## 모델 정의

:numref:`sec_linear_concise`에서와 같이,
저희는 내장 층을 사용하여
완전 연결층을 구성합니다.
그러면 내장된 `__call__` 메서드는 신경망을 어떤 입력에 적용해야 할 때마다
`forward`를 호출합니다.

:begin_tab:`mxnet`
입력 `X`가 4차 텐서임에도 불구하고,
내장 `Dense` 층은
첫 번째 축을 따른 차원을 변경하지 않은 채로 유지함으로써
`X`를 자동으로 2차 텐서로 변환합니다.
:end_tab:

:begin_tab:`pytorch`
저희는 `Flatten` 층을 사용하여 첫 번째 축을 따른 차원을 변경하지 않은 채로 유지함으로써
4차 텐서 `X`를 2차로 변환합니다.

:end_tab:

:begin_tab:`tensorflow`
저희는 `Flatten` 층을 사용하여 첫 번째 축을 따른 차원을 변경하지 않은 채로 유지함으로써
4차 텐서 `X`를 변환합니다.
:end_tab:

:begin_tab:`jax`
Flax는 사용자가 `@nn.compact` 데코레이터를 사용하여 더 간결한 방식으로
신경망 클래스를 작성할 수 있게 해 줍니다. `@nn.compact`를 사용하면
데이터클래스에서 표준 `setup` 메서드를 정의할 필요 없이
모든 신경망 로직을 하나의 "순전파(forward pass)" 메서드 안에
간단히 작성할 수 있습니다.
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## 소프트맥스 다시 살펴보기
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch`에서 저희는 모델의 출력을 계산하고
교차 엔트로피 손실을 적용했습니다. 이것이 수학적으로 완벽히
합리적이지만, 지수 함수에서의 수치적 언더플로우와
오버플로우 때문에 계산적으로는 위험합니다.

소프트맥스 함수가 확률을 $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$를 통해
계산함을 떠올려 보시기 바랍니다.
일부 $o_k$가 매우 크다면, 즉 매우 양수라면,
$\exp(o_k)$는 특정 데이터 타입에서 가질 수 있는
가장 큰 수보다 클 수도 있습니다. 이를 *오버플로우(overflow)*라고 합니다. 마찬가지로,
모든 인수가 매우 큰 음수라면, *언더플로우(underflow)*가 발생할 것입니다.
예를 들어, 단정밀도 부동 소수점 수는 대략 $10^{-38}$에서 $10^{38}$ 범위를
다룹니다. 따라서 $\mathbf{o}$의 가장 큰 항이
구간 $[-90, 90]$ 밖에 있다면, 결과는 안정적이지 않을 것입니다.
이 문제를 우회하는 방법은 모든 항목에서
$\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$를 빼는 것입니다.

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

구성 방식에 의해 모든 $j$에 대해 $o_j - \bar{o} \leq 0$임을 알고 있습니다. 따라서 $q$-클래스
분류 문제에서, 분모는 구간 $[1, q]$에 포함됩니다. 또한
분자는 결코 $1$을 초과하지 않으므로 수치적 오버플로우를 방지합니다. 수치적 언더플로우는
$\exp(o_j - \bar{o})$가 수치적으로 $0$으로 평가될 때만 발생합니다. 그럼에도 불구하고, 몇 단계
뒤에서 저희는 $\log \hat{y}_j$를 $\log 0$으로 계산하고자 할 때 곤란해질 수 있습니다.
특히 역전파에서,
저희는 두려운 `NaN`(Not a Number) 결과가 화면을 가득 채우는
상황에 직면할 수도 있습니다.

다행스럽게도, 지수 함수를 계산하고 있긴 하지만,
궁극적으로는 (교차 엔트로피 손실을 계산할 때)
그 로그를 취하려는 의도라는 사실 덕분에 저희는 구원받습니다.
소프트맥스와 교차 엔트로피를 결합함으로써,
저희는 수치적 안정성 문제를 모두 피할 수 있습니다. 다음을 얻습니다.

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

이는 오버플로우와 언더플로우를 모두 피합니다.
저희는 모델의 출력 확률을 평가하고자 할 때를 대비하여 관례적인 소프트맥스 함수를 가까이 두고 싶을 것입니다.
그러나 소프트맥스 확률을 새로운 손실 함수에 전달하는 대신,
저희는 단지
[**로짓을 전달하고 교차 엔트로피 손실 함수 내부에서
소프트맥스와 그 로그를 한꺼번에 계산합니다.**]
이는 ["LogSumExp 기법"](https://en.wikipedia.org/wiki/LogSumExp)과 같은 똑똑한 일들을 수행합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## 학습

다음으로 모델을 학습합니다. 저희는 784차원 특성 벡터로 평탄화된 Fashion-MNIST 이미지를 사용합니다.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

이전과 마찬가지로, 이 알고리즘은 합리적으로 정확한 해로 수렴하지만,
이번에는 이전보다 더 적은 코드 줄로 그렇게 합니다.


## 요약

고수준 API는 수치적 안정성과 같은 잠재적으로 위험한 측면들을 사용자에게서 숨기는 데 매우 편리합니다. 더욱이, 매우 적은 코드 줄로 모델을 간결하게 설계할 수 있도록 해 줍니다. 이는 축복이자 저주입니다. 명백한 이점은 통계 수업을 평생 한 번도 들어본 적이 없는 엔지니어(사실 이들은 이 책의 대상 독자 중 일부입니다)에게도 매우 접근하기 쉽게 만든다는 것입니다. 그러나 날카로운 모서리를 숨기는 데에는 대가도 따릅니다. 즉, 새로운 다른 구성 요소를 직접 추가하는 것에 대한 동기 부여가 약해진다는 점입니다. 그러한 일을 하기 위한 손에 익은 감각이 거의 없기 때문입니다. 게다가 프레임워크의 보호 패딩이 모든 모서리 사례를 완전히 다루지 못할 때마다 무언가를 *수정*하는 것을 더 어렵게 만듭니다. 다시 말하지만, 이는 친숙함의 부족 때문입니다.

따라서 저희는 이후 따라올 많은 구현의 골자만 남긴 버전과 우아한 버전을 *모두* 검토할 것을 강력히 권고합니다. 저희는 이해의 용이성을 강조하지만, 구현은 그럼에도 불구하고 일반적으로 꽤 성능이 좋습니다(여기서 합성곱은 큰 예외입니다). 어떤 프레임워크도 제공할 수 없는 새로운 것을 발명할 때 여러분이 이를 토대로 삼을 수 있도록 하는 것이 저희의 의도입니다.


## 연습문제

1. 딥러닝은 FP64 배정밀도(매우 드물게 사용됨),
FP32 단정밀도, BFLOAT16(압축된 표현에 적합), FP16(매우 불안정), TF32(NVIDIA의 새로운 형식), INT8 등 다양한 수 형식을 사용합니다. 결과가 수치적 언더플로우나 오버플로우로 이어지지 않는 지수 함수의 가장 작은 인수와 가장 큰 인수를 계산하시오.
1. INT8은 $1$에서 $255$까지의 0이 아닌 수로 구성된 매우 제한된 형식입니다. 더 많은 비트를 사용하지 않고 그 동적 범위를 어떻게 확장할 수 있을까요? 표준 곱셈과 덧셈이 여전히 작동할까요?
1. 학습을 위한 에폭 수를 늘려 보시기 바랍니다. 왜 일정 시간 후에 검증 정확도가 감소할 수도 있을까요? 이를 어떻게 고칠 수 있을까요?
1. 학습률을 늘리면 어떻게 되나요? 여러 학습률에 대한 손실 곡선을 비교해 보시기 바랍니다. 어떤 것이 더 잘 작동하나요? 언제 그러한가요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17983)
:end_tab:
