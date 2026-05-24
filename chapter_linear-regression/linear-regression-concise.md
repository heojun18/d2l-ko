```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 선형 회귀의 간결한 구현
:label:`sec_linear_concise`

지난 10년간 딥러닝은 일종의 캄브리아기 대폭발을 목격했습니다.
그 엄청난 기법, 응용, 알고리즘의 수는 지난 수십 년의 진보를
훨씬 뛰어넘습니다.
이는 여러 요인의 우연한 조합에 기인하며,
그 중 하나는 다수의 오픈 소스 딥러닝 프레임워크가 제공하는
강력한 무료 도구들입니다.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`는
널리 채택된 그러한 모델의
1세대를 대표한다고 할 수 있습니다.
Lisp 같은 프로그래밍 경험을 제공한
SN2(Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988` 같은
이전(선구적인) 작업과 대조적으로,
현대 프레임워크는 자동 미분과 Python의 편리함을 제공합니다.
이러한 프레임워크는 경사 기반 학습 알고리즘을 구현하는
반복적인 작업을 자동화하고 모듈화할 수 있게 해 줍니다.

:numref:`sec_linear_scratch`에서 저희는
(i) 데이터 저장과 선형대수를 위한 텐서와
(ii) 경사를 계산하기 위한 자동 미분에만 의존했습니다.
실제로 데이터 반복자, 손실 함수, 옵티마이저, 신경망 층은
너무나 일반적이기 때문에, 현대 라이브러리들은 이러한 구성 요소들을
저희를 위해 구현해 둡니다.
이 절에서는 :numref:`sec_linear_scratch`의 (**선형 회귀 모델을
딥러닝 프레임워크의 고수준 API를 사용해 간결하게 구현하는**) 방법을
보여드리겠습니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## 모델 정의

:numref:`sec_linear_scratch`에서 선형 회귀를 처음부터 구현했을 때,
저희는 모델 매개변수를 명시적으로 정의하고
기본 선형대수 연산을 사용해 출력을 만들어내는 계산을 코딩했습니다.
여러분은 이것을 어떻게 하는지 *알아야* 합니다.
하지만 모델이 더 복잡해지고
이를 거의 매일 해야 하게 되면,
도움을 받게 되어 기뻐할 것입니다.
이 상황은 자신만의 블로그를 처음부터 코딩하는 것과 비슷합니다.
한두 번 해보는 것은 보람되고 교육적이지만,
바퀴를 재발명하는 데 한 달을 보낸다면 형편없는 웹 개발자일 것입니다.

표준 연산에 대해서는,
[**프레임워크의 미리 정의된 층들을 사용**]할 수 있는데,
이를 통해 그 구현을 걱정하는 대신
모델을 구성하는 데 사용되는 층에 집중할 수 있습니다.
:numref:`fig_single_neuron`에서 묘사된 것처럼
단일 층 신경망의 아키텍처를 떠올려 보세요.
이 층은 *완전 연결(fully connected)*이라고 부르는데,
각 입력이 행렬--벡터 곱셈을 통해
각 출력에 연결되어 있기 때문입니다.

:begin_tab:`mxnet`
Gluon에서 완전 연결 층은 `Dense` 클래스에 정의되어 있습니다.
저희는 단일 스칼라 출력만 생성하고자 하므로,
그 수를 1로 설정합니다.
편의를 위해 Gluon은 각 층에 대해 입력 형태를 지정할 것을
저희에게 요구하지 않는다는 점이 주목할 가치가 있습니다.
따라서 이 선형 층에 얼마나 많은 입력이 들어가는지
Gluon에게 알려줄 필요가 없습니다.
모델을 통해 데이터를 처음 전달할 때,
예를 들어 나중에 `net(X)`를 실행할 때,
Gluon은 자동으로 각 층의 입력 수를 추론하고
따라서 올바른 모델을 인스턴스화합니다.
이것이 어떻게 작동하는지는 나중에 더 자세히 설명하겠습니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 완전 연결 층은 `Linear`와 `LazyLinear` 클래스(버전 1.8.0부터 사용 가능)에 정의되어 있습니다. 
후자는 출력 차원만 지정하도록 사용자에게 허용하는 반면,
전자는 이 층에 얼마나 많은 입력이 들어가는지도
추가로 요구합니다.
입력 형태를 지정하는 것은 불편하며 비자명한 계산을 필요로 할 수 있습니다
(예를 들어 합성곱 층에서처럼).
따라서 단순성을 위해, 가능할 때마다 그러한 "lazy" 층을 사용할 것입니다. 
:end_tab:

:begin_tab:`tensorflow`
Keras에서 완전 연결 층은 `Dense` 클래스에 정의되어 있습니다.
저희는 단일 스칼라 출력만 생성하고자 하므로,
그 수를 1로 설정합니다.
편의를 위해 Keras는 각 층에 대해 입력 형태를 지정할 것을
저희에게 요구하지 않는다는 점이 주목할 가치가 있습니다.
이 선형 층에 얼마나 많은 입력이 들어가는지
Keras에게 알려줄 필요가 없습니다.
모델을 통해 데이터를 처음 전달하려 할 때,
예를 들어 나중에 `net(X)`를 실행할 때,
Keras는 자동으로 각 층의 입력 수를 추론합니다.
이것이 어떻게 작동하는지는 나중에 더 자세히 설명하겠습니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

`forward` 메서드에서 저희는 단지 출력을 계산하기 위해 미리 정의된 층의 내장 `__call__` 메서드를 호출합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## 손실 함수 정의

:begin_tab:`mxnet`
`loss` 모듈은 많은 유용한 손실 함수를 정의합니다.
속도와 편의를 위해 저희만의 구현을 포기하고
대신 내장 `loss.L2Loss`를 선택합니다.
그것이 반환하는 `loss`는 각 예제에 대한 제곱 오차이기 때문에,
미니배치 전체에 걸쳐 손실을 평균하기 위해 `mean`을 사용합니다.
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` 클래스는 (:eqref:`eq_mse`의 $1/2$ 인수 없이) 평균 제곱 오차를 계산합니다.**]
기본적으로 `MSELoss`는 예제에 대한 평균 손실을 반환합니다.
저희만의 구현보다 더 빠릅니다(그리고 사용하기 더 쉽습니다).
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` 클래스는 (:eqref:`eq_mse`의 $1/2$ 인수 없이) 평균 제곱 오차를 계산합니다.
기본적으로 예제에 대한 평균 손실을 반환합니다.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## 최적화 알고리즘 정의

:begin_tab:`mxnet`
미니배치 SGD는 신경망을 최적화하기 위한 표준 도구이므로
Gluon은 `Trainer` 클래스를 통해
이 알고리즘의 여러 변형과 함께 이를 지원합니다.
Gluon의 `Trainer` 클래스는 최적화 알고리즘을 나타내고,
:numref:`sec_oo-design`에서 저희가 만든 `Trainer` 클래스는
훈련 방법, 즉 모델 매개변수를 갱신하기 위해
옵티마이저를 반복적으로 호출하는 것을 포함한다는 점에 유의하세요.
`Trainer`를 인스턴스화할 때,
저희는 최적화할 매개변수(저희 모델 `net`에서 `net.collect_params()`를 통해 얻을 수 있음),
사용하고자 하는 최적화 알고리즘(`sgd`),
저희의 최적화 알고리즘이 요구하는 하이퍼파라미터 딕셔너리를 지정합니다.
:end_tab:

:begin_tab:`pytorch`
미니배치 SGD는 신경망을 최적화하기 위한 표준 도구이므로
PyTorch는 `optim` 모듈에서 이 알고리즘의 여러 변형과 함께 이를 지원합니다.
(**`SGD` 인스턴스를 인스턴스화**)할 때,
저희는 최적화할 매개변수(저희 모델에서 `self.parameters()`를 통해 얻을 수 있음)와
저희의 최적화 알고리즘이 요구하는 학습률(`self.lr`)을 지정합니다.
:end_tab:

:begin_tab:`tensorflow`
미니배치 SGD는 신경망을 최적화하기 위한 표준 도구이므로
Keras는 `optimizers` 모듈에서 이 알고리즘의 여러 변형과 함께 이를 지원합니다.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## 훈련

딥러닝 프레임워크의 고수준 API를 통해 모델을 표현하는 것이
더 적은 코드 줄을 필요로 한다는 점을 알아챘을지 모릅니다.
저희는 매개변수를 개별적으로 할당하거나,
손실 함수를 정의하거나, 미니배치 SGD를 구현할 필요가 없었습니다.
훨씬 더 복잡한 모델을 다루기 시작하면,
고수준 API의 이점은 상당히 커질 것입니다.

이제 모든 기본 조각들이 제자리에 있으므로,
[**훈련 루프 자체는 저희가 처음부터 구현한 것과 같습니다.**]
그래서 모델을 훈련하기 위해 :numref:`sec_linear_scratch`의
`fit_epoch` 메서드 구현에 의존하는
`fit` 메서드(:numref:`oo-design-training`에서 소개됨)를 호출하기만 합니다.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

아래에서 저희는
[**유한한 데이터에 대한 훈련으로 학습된 모델 매개변수와
저희 데이터셋을 생성한 실제 매개변수를 비교**]합니다.
매개변수에 접근하기 위해,
저희가 필요로 하는 층의 가중치와 편향에 접근합니다.
저희의 처음부터 구현한 것과 마찬가지로,
저희의 추정 매개변수가 참 값에 가깝다는 점에 유의하세요.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## 요약

이 절은 MXNet :cite:`Chen.Li.Li.ea.2015`, 
JAX :cite:`Frostig.Johnson.Leary.2018`, 
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, 
Tensorflow :cite:`Abadi.Barham.Chen.ea.2016` 같은 현대 딥러닝 프레임워크가
제공하는 편의를 활용한
(이 책의) 첫 번째 심층 신경망 구현을 담고 있습니다.
저희는 데이터를 로드하고, 층을 정의하고, 손실 함수, 옵티마이저, 훈련 루프를 위해
프레임워크의 기본값을 사용했습니다.
프레임워크가 모든 필요한 기능을 제공할 때는 일반적으로 그것들을 사용하는 것이 좋은 생각입니다.
이러한 구성 요소들의 라이브러리 구현은 성능을 위해 많이 최적화되고
신뢰성을 위해 적절히 시험되는 경향이 있기 때문입니다.
동시에 이러한 모듈들이 직접 구현*될 수* 있다는 사실을
잊지 않도록 노력하세요.
이는 특히 모델 개발의 최전선에 살고자 하는
야심찬 연구자들에게 중요한데,
거기서 여러분은 현재 어떤 라이브러리에도 존재할 수 없는
새로운 구성 요소를 발명하게 될 것이기 때문입니다.

:begin_tab:`mxnet`
Gluon에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하고,
`nn` 모듈은 많은 수의 신경망 층을 정의하며,
`loss` 모듈은 많은 일반적인 손실 함수를 정의합니다.
또한 `initializer`는 매개변수 초기화를 위한
많은 선택지에 접근을 제공합니다.
사용자에게 편리하게도, 차원성과 저장은 자동으로 추론됩니다.
이러한 게으른 초기화의 결과로,
매개변수가 인스턴스화(되고 초기화)되기 전에는
이에 접근하려고 시도해서는 안 됩니다.
:end_tab:

:begin_tab:`pytorch`
PyTorch에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하고,
`nn` 모듈은 많은 수의 신경망 층과 일반적인 손실 함수를 정의합니다.
값을 `_`로 끝나는 메서드로 대체함으로써 매개변수를 초기화할 수 있습니다.
신경망의 입력 차원을 지정해야 한다는 점에 유의하세요.
지금은 이것이 사소하지만, 많은 층이 있는 복잡한 신경망을 설계하고자 할 때
상당한 연쇄 효과를 가질 수 있습니다.
이식성을 허용하기 위해서는 이러한 신경망을 어떻게 매개변수화할지에 대한
신중한 고려가 필요합니다.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow에서 `data` 모듈은 데이터 처리를 위한 도구를 제공하고,
`keras` 모듈은 많은 수의 신경망 층과 일반적인 손실 함수를 정의합니다.
또한 `initializers` 모듈은 모델 매개변수 초기화를 위한 다양한 방법을 제공합니다.
신경망에 대한 차원성과 저장은 자동으로 추론됩니다
(다만 매개변수가 초기화되기 전에 접근하려고 시도하지 않도록 주의하세요).
:end_tab:

## 연습문제

1. 미니배치에 대한 손실의 합계를 미니배치에 대한 손실의 평균으로 대체한다면
   학습률을 어떻게 바꿔야 할까요?
1. 어떤 손실 함수가 제공되는지 보기 위해 프레임워크 문서를 검토하세요. 특히,
   제곱 손실을 후버(Huber)의 강건 손실 함수로 대체하세요. 즉, 손실 함수
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$를 사용하세요.
1. 모델 가중치의 경사에는 어떻게 접근합니까?
1. 학습률과 에포크 수를 바꾸면 해에 어떤 영향이 있습니까? 계속해서 개선됩니까?
1. 생성되는 데이터의 양을 변화시킴에 따라 해가 어떻게 변합니까?
    1. $\hat{\mathbf{w}} - \mathbf{w}$와 $\hat{b} - b$의 추정 오차를 데이터의 양의 함수로 그려 보세요. 힌트: 데이터의 양을 선형적이 아니라 로그적으로, 즉 1000, 2000, ..., 10,000이 아니라 5, 10, 20, 50, ..., 10,000으로 늘리세요.
    2. 힌트의 제안이 적절한 이유는 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17977)
:end_tab:
