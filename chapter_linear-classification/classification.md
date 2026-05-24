```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 기본 분류 모델
:label:`sec_classification`

회귀의 경우 처음부터 구현한 코드와 프레임워크 기능을 사용한 간결한 구현이 상당히 유사하다는 것을 알아차렸을 수 있습니다. 분류의 경우에도 마찬가지입니다. 이 책의 많은 모델이 분류를 다루기 때문에, 이 설정을 구체적으로 지원하는 기능을 추가하는 것이 가치 있습니다. 이 절에서는 향후 코드를 단순화하기 위해 분류 모델을 위한 기본 클래스를 제공합니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## `Classifier` 클래스

:begin_tab:`pytorch, mxnet, tensorflow`
저희는 아래에 `Classifier` 클래스를 정의합니다. `validation_step`에서는 검증 배치에 대한 손실 값과 분류 정확도를 모두 보고합니다. `num_val_batches` 배치마다 업데이트를 가져옵니다. 이는 전체 검증 데이터에 대해 평균을 낸 손실과 정확도를 생성한다는 이점이 있습니다. 마지막 배치가 더 적은 예제를 포함하는 경우 이 평균값은 정확하지 않지만, 코드를 단순하게 유지하기 위해 이 사소한 차이는 무시합니다.
:end_tab:


:begin_tab:`jax`
저희는 아래에 `Classifier` 클래스를 정의합니다. `validation_step`에서는 검증 배치에 대한 손실 값과 분류 정확도를 모두 보고합니다. `num_val_batches` 배치마다 업데이트를 가져옵니다. 이는 전체 검증 데이터에 대해 평균을 낸 손실과 정확도를 생성한다는 이점이 있습니다. 마지막 배치가 더 적은 예제를 포함하는 경우 이 평균값은 정확하지 않지만, 코드를 단순하게 유지하기 위해 이 사소한 차이는 무시합니다.

또한 JAX의 경우 `training_step` 메서드를 재정의합니다. 나중에 `Classifier`를 서브클래스화하는 모든 모델이 보조 데이터를 반환하는 손실을 갖기 때문입니다.
이 보조 데이터는 배치 정규화를 사용하는 모델(:numref:`sec_batch_norm`에서 설명될 예정)에 사용될 수 있으며, 그 외의 모든 경우에는 보조 데이터를 표현하기 위해 손실이 자리 표시자(빈 딕셔너리)도 반환하게 할 것입니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

기본적으로 저희는 선형 회귀의 맥락에서 그랬던 것처럼 미니배치를 다루는 확률적 경사 하강법 옵티마이저를 사용합니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## 정확도

예측된 확률 분포 `y_hat`가 주어졌을 때,
하드 예측(hard prediction)을 출력해야 할 때는
일반적으로 가장 높은 예측 확률을 가진 클래스를 선택합니다.
실제로 많은 응용 분야에서는 저희가 선택을 해야 합니다.
예를 들어, Gmail은 이메일을 "기본", "소셜", "프로모션", "포럼", "스팸" 중 하나로 분류해야 합니다.
내부적으로는 확률을 추정할 수 있지만,
결국에는 클래스 중 하나를 선택해야 합니다.

예측이 레이블 클래스 `y`와 일치할 때, 그것은 정확한 예측입니다.
분류 정확도는 모든 예측 중 정확한 예측이 차지하는 비율입니다.
정확도를 직접 최적화하는 것은 어려울 수 있지만(미분 가능하지 않음),
그것이 종종 저희가 가장 신경 쓰는 성능 지표입니다. 이는 흔히 벤치마크에서
관심 있는 정량적 지표이기도 합니다. 따라서 분류기를 학습할 때 저희는 거의 항상 정확도를 보고할 것입니다.

정확도는 다음과 같이 계산됩니다.
먼저, `y_hat`이 행렬이라면,
저희는 두 번째 차원이 각 클래스에 대한 예측 점수를 저장한다고 가정합니다.
저희는 `argmax`를 사용하여 각 행의 가장 큰 항목의 인덱스로 예측된 클래스를 얻습니다.
그런 다음 [**예측된 클래스를 실제값 `y`와 원소별로 비교합니다.**]
등호 연산자 `==`는 데이터 타입에 민감하므로,
저희는 `y_hat`의 데이터 타입을 `y`의 데이터 타입과 일치하도록 변환합니다.
결과는 0(거짓)과 1(참)의 항목을 포함하는 텐서입니다.
합을 취하면 정확한 예측의 수가 산출됩니다.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## 요약

분류는 자체 편의 함수가 필요할 만큼 충분히 흔한 문제입니다. 분류에서 핵심적으로 중요한 것은 분류기의 *정확도(accuracy)*입니다. 저희가 주로 정확도에 관심을 갖는 경우가 많지만, 통계적이고 계산적인 이유로 분류기를 학습할 때는 다양한 다른 목적을 최적화한다는 점에 유의하시기 바랍니다. 그러나 학습 중에 어떤 손실 함수가 최소화되었는지에 관계없이, 분류기의 정확도를 경험적으로 평가하기 위한 편의 메서드를 갖는 것은 유용합니다.


## 연습문제

1. 검증 손실을 $L_\textrm{v}$, 이 절에서 손실 함수의 평균화로 계산된 빠르고 거친 추정값을 $L_\textrm{v}^\textrm{q}$로 표기하시오. 마지막으로, 마지막 미니배치의 손실을 $l_\textrm{v}^\textrm{b}$로 표기하시오. $L_\textrm{v}$를 $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$, 그리고 표본 크기와 미니배치 크기로 표현하시오.
1. 빠르고 거친 추정값 $L_\textrm{v}^\textrm{q}$가 편향되지 않음을 보이시오. 즉, $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$임을 보이시오. 그럼에도 불구하고 왜 $L_\textrm{v}$를 사용하고 싶을까요?
1. 다중 클래스 분류 손실이 주어졌을 때, $y$를 보고 $y'$를 추정하는 데 대한 페널티를 $l(y,y')$로 표기하고 확률 $p(y \mid x)$가 주어졌을 때, $y'$의 최적 선택을 위한 규칙을 정식화하시오. 힌트: $l$과 $p(y \mid x)$를 사용하여 기대 손실을 표현하시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17981)
:end_tab:
