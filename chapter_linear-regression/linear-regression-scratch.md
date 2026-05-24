```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형 회귀의 처음부터 구현하기
:label:`sec_linear_scratch`

이제 저희는 선형 회귀의 완전히 작동하는 구현을
하나씩 살펴볼 준비가 되었습니다.
이 절에서는 (**전체 방법을 처음부터 구현할 것입니다.
여기에는 (i) 모델, (ii) 손실 함수,
(iii) 미니배치 확률적 경사 하강법 옵티마이저,
(iv) 이 모든 조각을 한데 엮는 훈련 함수가
포함됩니다.**)
마지막으로 :numref:`sec_synthetic-regression-data`의
합성 데이터 생성기를 실행하고
그 결과 데이터셋에 모델을 적용할 것입니다.
현대 딥러닝 프레임워크는 이 작업의 거의 전부를
자동화할 수 있지만, 처음부터 구현하는 것은
여러분이 무엇을 하고 있는지 정말로 안다고 확신할 수 있는 유일한 방법입니다.
게다가 모델을 커스터마이즈해 자신만의 층이나 손실 함수를
정의할 때가 되면, 내부에서 어떻게 작동하는지를 이해하는 것이
유용하게 쓰일 것입니다.
이 절에서는 텐서와 자동 미분에만 의존할 것입니다.
나중에 아래에 이어지는 내용의 구조를 유지하면서
딥러닝 프레임워크의 부가 기능을 활용한
더 간결한 구현을 소개하겠습니다.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## 모델 정의

[**미니배치 SGD로 모델의 매개변수를 최적화하기 시작하기 전에,**]
(**우선 매개변수가 좀 있어야 합니다.**)
다음에서 저희는 평균 0과 표준 편차 0.01인
정규 분포에서 난수를 추출해 가중치를 초기화합니다.
마법의 숫자 0.01은 실제로 종종 잘 작동하지만,
인수 `sigma`를 통해 다른 값을 지정할 수도 있습니다.
또한 편향을 0으로 설정합니다.
객체 지향 설계를 위해, `d2l.Module`(:numref:`subsec_oo-design-models`에서 소개됨)의
하위 클래스의 `__init__` 메서드에 코드를 추가한다는 점에 유의하세요.

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

다음으로 저희는 [**모델을 정의하고,
입력과 매개변수를 출력에 연결해야 합니다.**]
선형 모델에 대해 :eqref:`eq_linreg-y-vec`와 같은 표기를 사용하여,
저희는 단순히 입력 특징 $\mathbf{X}$와 모델 가중치 $\mathbf{w}$의
행렬--벡터 곱을 취하고, 각 예제에 오프셋 $b$를 더합니다.
곱 $\mathbf{Xw}$는 벡터이고 $b$는 스칼라입니다.
브로드캐스팅 메커니즘(:numref:`subsec_broadcasting` 참조) 때문에,
벡터와 스칼라를 더할 때 스칼라는 벡터의 각 성분에 더해집니다.
그 결과 `forward` 메서드는
`add_to_class`(:numref:`oo-design-utilities`에서 소개됨)를 통해
`LinearRegressionScratch` 클래스에 등록됩니다.

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## 손실 함수 정의

[**모델을 갱신하려면 손실 함수의 경사를 취해야 하기 때문에,**]
(**먼저 손실 함수를 정의해야 합니다.**)
여기서는 :eqref:`eq_mse`의 제곱 손실 함수를 사용합니다.
구현에서는 참 값 `y`를 예측값의 형태 `y_hat`으로 변환해야 합니다.
다음 메서드가 반환하는 결과도 `y_hat`과 같은 형태를 가질 것입니다.
또한 미니배치의 모든 예제에 대한 평균 손실 값을 반환합니다.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## 최적화 알고리즘 정의

:numref:`sec_linear_regression`에서 논의했듯이,
선형 회귀는 닫힌 형태의 해를 가집니다.
그러나 여기서 저희의 목표는 더 일반적인 신경망을
어떻게 훈련하는지 설명하는 것이며,
이를 위해서는 여러분에게 미니배치 SGD를 어떻게 사용하는지
가르쳐 드려야 합니다.
따라서 이 기회를 빌려 첫 번째 작동하는 SGD 예제를 소개하겠습니다.
각 단계에서, 저희 데이터셋에서 무작위로 추출한 미니배치를 사용하여
매개변수에 대한 손실의 경사를 추정합니다.
다음으로 손실을 줄일 수 있는 방향으로 매개변수를 갱신합니다.

다음 코드는 매개변수 집합과 학습률 `lr`이 주어졌을 때
갱신을 적용합니다.
저희 손실은 미니배치에 대한 평균으로 계산되기 때문에,
배치 크기에 따라 학습률을 조정할 필요는 없습니다.
이후 장에서는 분산 대규모 학습에서 등장하는 매우 큰 미니배치의 경우
학습률을 어떻게 조정해야 하는지 살펴볼 것입니다.
지금은 이 의존성을 무시할 수 있습니다.

:begin_tab:`mxnet`
저희는 내장 SGD 옵티마이저와 비슷한 API를 갖도록
`d2l.HyperParameters`(:numref:`oo-design-utilities`에서 소개됨)의
하위 클래스인 `SGD` 클래스를 정의합니다.
`step` 메서드에서 매개변수를 갱신합니다.
이는 무시될 수 있는 `batch_size` 인수를 받습니다.
:end_tab:

:begin_tab:`pytorch`
저희는 내장 SGD 옵티마이저와 비슷한 API를 갖도록
`d2l.HyperParameters`(:numref:`oo-design-utilities`에서 소개됨)의
하위 클래스인 `SGD` 클래스를 정의합니다.
`step` 메서드에서 매개변수를 갱신합니다.
`zero_grad` 메서드는 모든 경사를 0으로 설정하며,
역전파 단계 전에 실행되어야 합니다.
:end_tab:

:begin_tab:`tensorflow`
저희는 내장 SGD 옵티마이저와 비슷한 API를 갖도록
`d2l.HyperParameters`(:numref:`oo-design-utilities`에서 소개됨)의
하위 클래스인 `SGD` 클래스를 정의합니다.
`apply_gradients` 메서드에서 매개변수를 갱신합니다.
이는 매개변수와 경사 쌍의 리스트를 받습니다.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

다음으로 `SGD` 클래스의 인스턴스를 반환하는 `configure_optimizers` 메서드를 정의합니다.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## 훈련

이제 모든 부품(매개변수, 손실 함수, 모델, 옵티마이저)이 제자리에 있으므로,
저희는 [**주요 훈련 루프를 구현할**] 준비가 되었습니다.
이 코드를 완전히 이해하는 것은 매우 중요한데,
이 책에서 다루는 다른 모든 딥러닝 모델에 대해
비슷한 훈련 루프를 사용할 것이기 때문입니다.
각 *에포크(epoch)*에서, 저희는 전체 훈련 데이터셋을 순회하며
모든 예제를 한 번씩 통과합니다(예제 수가 배치 크기로 나누어떨어진다고 가정).
각 *반복(iteration)*에서, 훈련 예제의 미니배치를 가져와
모델의 `training_step` 메서드를 통해 손실을 계산합니다.
그런 다음 각 매개변수에 대한 경사를 계산합니다.
마지막으로, 모델 매개변수를 갱신하기 위해
최적화 알고리즘을 호출합니다.
요약하면, 다음 루프를 실행할 것입니다.

* 매개변수 $(\mathbf{w}, b)$ 초기화
* 완료될 때까지 반복
    * 경사 $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$ 계산
    * 매개변수 $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$ 갱신
 
저희가 :numref:``sec_synthetic-regression-data``에서 생성한 합성 회귀 데이터셋은
검증 데이터셋을 제공하지 않는다는 점을 떠올려 보세요.
그러나 대부분의 경우 저희는 모델 품질을 측정하기 위한
검증 데이터셋을 원할 것입니다.
여기서는 모델 성능을 측정하기 위해 각 에포크에서
검증 데이터로더를 한 번씩 통과시킵니다.
저희의 객체 지향 설계에 따라,
`prepare_batch`와 `fit_epoch` 메서드는
`d2l.Trainer` 클래스(:numref:`oo-design-training`에서 소개됨)에 등록됩니다.

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

저희는 모델을 훈련할 준비가 거의 다 되었지만,
먼저 약간의 훈련 데이터가 필요합니다.
여기서는 `SyntheticRegressionData` 클래스를 사용하고
일부 정답 매개변수를 전달합니다.
그런 다음 학습률 `lr=0.03`으로 모델을 훈련하고
`max_epochs=3`으로 설정합니다.
일반적으로 에포크 수와 학습률은 모두 하이퍼파라미터라는 점에 유의하세요.
일반적으로 하이퍼파라미터를 설정하는 것은 까다로우며,
저희는 보통 훈련용, 하이퍼파라미터 선택용, 최종 평가용으로
예비된 세 갈래 분할을 사용하고자 할 것입니다.
지금은 이러한 세부 사항을 생략하지만 나중에 다시 다룰 것입니다.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

저희가 데이터셋을 직접 합성했기 때문에,
참 매개변수가 정확히 무엇인지 알고 있습니다.
따라서 [**참 매개변수와 훈련 루프를 통해 학습한 매개변수를 비교함으로써
훈련의 성공을 평가할**] 수 있습니다.
실제로 둘은 서로 매우 가까운 것으로 드러납니다.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

정답 매개변수를 정확히 복원할 수 있는 능력을 당연시해서는 안 됩니다.
일반적으로 심층 모델에 대해서는 매개변수에 대한 유일한 해가 존재하지 않으며,
선형 모델조차도 매개변수를 정확히 복원하는 것은
어떤 특징도 다른 특징들에 선형 종속이 아닐 때만 가능합니다.
그러나 머신러닝에서 저희는 종종 참 기저 매개변수를 복원하는 데
덜 관심을 두고, 오히려 매우 정확한 예측으로 이어지는 매개변수에 관심을 둡니다 :cite:`Vapnik.1992`.
다행히도 어려운 최적화 문제에서조차
확률적 경사 하강법은 종종 놀랍도록 좋은 해를 찾을 수 있는데,
이는 부분적으로 심층 신경망의 경우 매우 정확한 예측으로 이어지는
매개변수의 구성이 많이 존재한다는 사실 덕분입니다.


## 요약

이 절에서 저희는 완전히 기능하는 신경망 모델과 훈련 루프를 구현함으로써
딥러닝 시스템을 설계하는 데 중요한 한 걸음을 내디뎠습니다.
이 과정에서 저희는 데이터 로더, 모델, 손실 함수, 최적화 절차,
시각화 및 모니터링 도구를 만들었습니다.
모델 훈련을 위한 모든 관련 구성 요소를 포함하는
Python 객체를 합성함으로써 이를 수행했습니다.
이것은 아직 전문가급 구현은 아니지만, 완벽하게 기능하며
이와 같은 코드는 이미 작은 문제를 빠르게 해결하는 데
도움이 될 수 있습니다.
다음 절에서는 이를 *더 간결하게*(보일러플레이트 코드를 피하면서) 
그리고 *더 효율적으로*(GPU를 그 잠재력을 최대한 사용하면서)
하는 방법을 살펴보겠습니다.



## 연습문제

1. 가중치를 0으로 초기화한다면 어떻게 됩니까? 알고리즘이 여전히 작동할까요? 매개변수를 $0.01$ 대신 분산 $1000$으로 초기화하면 어떻게 됩니까?
1. 여러분이 전압과 전류를 관련짓는 저항 모델을 만들고자 하는 [게오르크 시몬 옴(Georg Simon Ohm)](https://en.wikipedia.org/wiki/Georg_Ohm)이라고 가정합니다. 자동 미분을 사용해 모델의 매개변수를 학습할 수 있습니까?
1. 스펙트럼 에너지 밀도를 사용해 물체의 온도를 결정하기 위해 [플랑크 법칙(Planck's Law)](https://en.wikipedia.org/wiki/Planck%27s_law)을 사용할 수 있습니까? 참고로, 흑체에서 방출되는 복사의 스펙트럼 밀도 $B$는
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$로 주어집니다. 여기서
   $\lambda$는 파장, $T$는 온도, $c$는 빛의 속도, $h$는 플랑크 상수, $k$는
   볼츠만 상수입니다. 다양한 파장 $\lambda$에 대해 에너지를 측정하고, 이제 스펙트럼 밀도 곡선을
   플랑크 법칙에 맞춰야 합니다.
1. 손실의 이계 도함수를 계산하려고 한다면 어떤 문제에 부딪힐 수 있습니까? 어떻게
   해결하시겠습니까?
1. `loss` 함수에서 `reshape` 메서드가 필요한 이유는 무엇입니까?
1. 손실 함수 값이 얼마나 빠르게 떨어지는지 알아보기 위해 다른 학습률을 사용해 실험해 보세요. 훈련의 에포크 수를 늘려서
   오차를 줄일 수 있습니까?
1. 예제의 수가 배치 크기로 나누어떨어지지 않으면, 에포크 끝에서 `data_iter`에 어떤 일이 일어납니까?
1. `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()` 같은 절대값 손실처럼 다른 손실 함수를 구현해 보세요.
    1. 일반적인 데이터에 대해 어떤 일이 일어나는지 확인하세요.
    1. $\mathbf{y}$의 일부 항목, 예를 들어 $y_5 = 10000$을 능동적으로 교란시키면 동작에 차이가 있는지 확인하세요.
    1. 제곱 손실과 절대값 손실의 가장 좋은 측면을 결합하기 위한 값싼 해를 생각해 낼 수 있습니까?
       힌트: 정말 큰 경사 값을 어떻게 피할 수 있을까요?
1. 데이터셋을 다시 섞어야 하는 이유는 무엇입니까? 그렇게 하지 않으면 악의적으로 구성된 데이터셋이 최적화 알고리즘을 망가뜨릴 수 있는 경우를 설계할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17976)
:end_tab:
