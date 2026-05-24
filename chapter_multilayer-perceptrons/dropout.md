```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 드롭아웃
:label:`sec_dropout`


좋은 예측 모델에서 저희가 무엇을 기대하는지
잠깐 생각해 봅시다.
저희는 모델이 보지 못한 데이터에서 잘 수행되기를 원합니다.
고전적 일반화 이론은
훈련 성능과 테스트 성능 사이의 격차를 좁히려면,
저희가 단순한 모델을 목표로 해야 한다고
시사합니다.
단순성은 적은 차원 수의 형태로 올 수 있습니다.
저희는 :numref:`sec_generalization_basics`에서 선형 모델의
단항 기저 함수를 논의할 때 이를 탐구했습니다.
또한 :numref:`sec_weight_decay`에서 가중치 감쇠
(($\ell_2$ 정규화))를 논의할 때 보았듯이,
파라미터의 ((역)) 노름도
단순성의 유용한 척도를 나타냅니다.
단순성의 또 다른 유용한 개념은 매끄러움입니다.
즉, 함수가 입력의 작은 변화에 민감하지 않아야 한다는 것입니다.
예를 들어, 이미지를 분류할 때,
저희는 픽셀에 약간의 무작위 잡음을 추가하는 것이
거의 무해해야 한다고 기대할 것입니다.

:citet:`Bishop.1995`는
입력 잡음으로 훈련하는 것이 티호노프 정규화와 동등하다는 것을 증명함으로써
이 아이디어를 형식화했습니다.
이 연구는 함수가 매끄러워야 한다는 ((그리고 따라서 단순해야 한다는)) 요건과
입력의 섭동에 견고해야 한다는 요건 사이에
명확한 수학적 연결을 그어주었습니다.

그 후, :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`는
Bishop의 아이디어를 네트워크의 내부 층에도
어떻게 적용할지에 대한 영리한 아이디어를 개발했습니다.
*드롭아웃*이라고 불리는 그들의 아이디어는
순전파 동안 각 내부 층을 계산하면서
잡음을 주입하는 것을 포함하며,
신경망 훈련의 표준 기법이 되었습니다.
이 방법이 *드롭아웃*이라 불리는 것은 저희가 훈련 동안
말 그대로 일부 뉴런을 *떨어뜨리기* 때문입니다.
훈련 내내, 각 반복마다,
표준 드롭아웃은 후속 층을 계산하기 전에
각 층의 노드 중 일부를 0으로 만드는 것으로
구성됩니다.

분명히 하자면, 저희는 Bishop과의 연결로
저희만의 서사를 강요하고 있는 것입니다.
드롭아웃에 관한 원 논문은
유성 생식과의 놀라운 비유를 통해
직관을 제공합니다.
저자들은 신경망 과적합이
각 층이 이전 층의 활성화의 특정 패턴에 의존하는
상태가 특징이라고 주장하며,
이 조건을 *공동 적응*이라고 부릅니다.
드롭아웃은, 그들이 주장하기를,
유성 생식이 공동 적응된 유전자를 깨뜨린다고 주장되는 것과 마찬가지로
공동 적응을 깨뜨립니다.
이 이론의 그러한 정당화가 확실히 논쟁의 여지가 있지만,
드롭아웃 기법 자체는 지속력이 있는 것으로 입증되었으며,
다양한 형태의 드롭아웃이 대부분의 딥러닝 라이브러리에
구현되어 있습니다.


핵심 과제는 이 잡음을 어떻게 주입할 것인가입니다.
한 가지 아이디어는 각 층의 기댓값이((다른 층들을 고정한 상태에서))
잡음이 없었을 때 가졌을 값과 같도록
*편향 없는* 방식으로 주입하는 것입니다.
Bishop의 연구에서, 그는 선형 모델의 입력에
가우시안 잡음을 추가했습니다.
각 훈련 반복마다, 그는 입력 $\mathbf{x}$에
평균이 0인 분포에서 샘플링한 잡음
$\epsilon \sim \mathcal{N}(0,\sigma^2)$을 추가하여,
섭동된 점 $\mathbf{x}' = \mathbf{x} + \epsilon$을 산출했습니다.
기댓값에서, $E[\mathbf{x}'] = \mathbf{x}$입니다.

표준 드롭아웃 정규화에서는,
각 층의 노드 중 일부를 0으로 만들고
유지된 노드의 비율 ((드롭되지 않은))로 정규화하여
각 층을 *편향 해소*합니다.
다시 말해,
*드롭아웃 확률* $p$로,
각 중간 활성화 $h$는 다음과 같이 무작위 변수 $h'$로
대체됩니다.

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ with probability } p \\
    \frac{h}{1-p} & \textrm{ otherwise}
\end{cases}
\end{aligned}
$$

설계상, 기댓값은 변하지 않습니다. 즉, $E[h'] = h$입니다.

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
import jax
from jax import numpy as jnp
import optax
```

## 실전에서의 드롭아웃

:numref:`fig_mlp`의 은닉층 하나와 다섯 개의 은닉 유닛을 가진
MLP를 떠올려 봅시다.
저희가 은닉층에 드롭아웃을 적용하여
각 은닉 유닛을 확률 $p$로 0으로 만들면,
결과는 원래 뉴런의 일부 부분 집합만을 포함하는
네트워크로 볼 수 있습니다.
:numref:`fig_dropout2`에서 $h_2$와 $h_5$가 제거됩니다.
결과적으로, 출력의 계산은
더 이상 $h_2$나 $h_5$에 의존하지 않으며
역전파를 수행할 때 그들의 각 기울기도
사라집니다.
이런 식으로, 출력층의 계산은
$h_1, \ldots, h_5$의 어느 한 원소에도
지나치게 의존할 수 없게 됩니다.

![드롭아웃 적용 전후의 MLP.](../img/dropout2.svg)
:label:`fig_dropout2`

일반적으로, 저희는 테스트 시 드롭아웃을 비활성화합니다.
훈련된 모델과 새로운 예제가 주어지면,
저희는 어떤 노드도 드롭하지 않으며
따라서 정규화할 필요가 없습니다.
하지만 몇 가지 예외가 있습니다.
일부 연구자들은 신경망 예측의 *불확실성*을 추정하기 위한
휴리스틱으로 테스트 시 드롭아웃을 사용합니다.
예측이 여러 다른 드롭아웃 출력에 걸쳐 일치한다면,
저희는 네트워크가 더 확신을 가지고 있다고 말할 수 있을 것입니다.

## 처음부터 구현하기

단일 층에 대해 드롭아웃 함수를 구현하려면,
저희는 층이 가진 차원만큼 많은 표본을
베르누이 ((이진)) 무작위 변수에서 뽑아야 합니다.
이 무작위 변수는 확률 $1-p$로 값 $1$ ((유지))을,
확률 $p$로 값 $0$ ((드롭))을 취합니다.
이를 구현하는 한 가지 쉬운 방법은 먼저 균등 분포
$U[0, 1]$에서 표본을 뽑는 것입니다.
그런 다음 저희는 해당 표본이 $p$보다 큰 노드를 유지하고,
나머지를 드롭할 수 있습니다.

다음 코드에서, 저희는 (**텐서 입력 `X`의 원소를 확률 `dropout`으로
드롭하고, 위에서 설명한 대로 나머지를 다시 스케일링하는,
즉 생존자를 `1.0-dropout`으로 나누는 `dropout_layer` 함수를
구현합니다**).

```{.python .input}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

저희는 [**몇 가지 예제에서 `dropout_layer` 함수를 시험해 볼 수 있습니다**].
다음 코드에서,
저희는 입력 `X`를 각각 확률 0, 0.5, 1로
드롭아웃 연산에 통과시킵니다.

```{.python .input}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### 모델 정의

아래 모델은 각 은닉층의 출력에 ((활성화 함수에 이어))
드롭아웃을 적용합니다.
저희는 각 층에 대해 별도로 드롭아웃 확률을 설정할 수 있습니다.
흔히 사용되는 선택은 입력층에 가까울수록
드롭아웃 확률을 더 낮게 설정하는 것입니다.
저희는 드롭아웃이 훈련 동안에만 활성화되도록 합니다.

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)

    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    def setup(self):
        self.lin1 = nn.Dense(self.num_hiddens_1)
        self.lin2 = nn.Dense(self.num_hiddens_2)
        self.lin3 = nn.Dense(self.num_outputs)
        self.relu = nn.relu

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**훈련**]

다음은 앞서 설명한 MLP의 훈련과 유사합니다.

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**간결한 구현**]

고수준 API를 사용하면, 저희가 할 일은 각 완전 연결 층 뒤에
`Dropout` 층을 추가하고,
드롭아웃 확률을 생성자의 유일한 인수로
전달하는 것뿐입니다.
훈련 동안, `Dropout` 층은 지정된 드롭아웃 확률에 따라
이전 층의 출력 ((또는 동등하게, 후속 층의 입력))을
무작위로 드롭합니다.
훈련 모드가 아닐 때는,
`Dropout` 층은 테스트 동안 데이터를 그대로 통과시킵니다.

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class DropoutMLP(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    @nn.compact
    def __call__(self, X):
        x = nn.relu(nn.Dense(self.num_hiddens_1)(X.reshape((X.shape[0], -1))))
        x = nn.Dropout(self.dropout_1, deterministic=not self.training)(x)
        x = nn.relu(nn.Dense(self.num_hiddens_2)(x))
        x = nn.Dropout(self.dropout_2, deterministic=not self.training)(x)
        return nn.Dense(self.num_outputs)(x)
```

:begin_tab:`jax`
드롭아웃 층이 있는 네트워크는 `Module.apply()`를 사용할 때 PRNGKey가 필요하므로
저희는 손실 함수를 다시 정의해야 한다는 점에 유의하세요.
그리고 이 RNG 시드는 명시적으로 `dropout`이라고 명명되어야 합니다. 이 키는
Flax의 `dropout` 층이 무작위 드롭아웃 마스크를
내부적으로 생성하는 데 사용됩니다. 훈련 루프의 모든 에포크마다
고유한 `dropout_rng` 키를 사용하는 것이 중요한데,
그렇지 않으면 생성된 드롭아웃 마스크가 확률적이지 않고
에포크 실행 사이에 동일해질 것입니다.
이 `dropout_rng`는
`TrainState` 객체 ((:numref:`oo-design-training`에 정의된 `d2l.Trainer` 클래스에 있음))에
속성으로 저장될 수 있으며, 매 에포크마다
새로운 `dropout_rng`로 교체됩니다. 저희는 이미 :numref:`sec_linear_scratch`에 정의된
`fit_epoch` 메서드로 이를 처리했습니다.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False,  # To be used later (e.g., batch norm)
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

다음으로, 저희는 [**모델을 훈련합니다**].

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## 요약

차원의 수와 가중치 벡터의 크기를 통제하는 것 외에도, 드롭아웃은 과적합을 피하기 위한 또 다른 도구입니다. 흔히 도구들은 함께 사용됩니다.
드롭아웃은 훈련 동안에만
사용된다는 점에 유의하세요.
이는 활성화 $h$를 기댓값이 $h$인 무작위 변수로 대체합니다.


## 연습문제

1. 첫 번째 층과 두 번째 층의 드롭아웃 확률을 바꾸면 어떤 일이 일어나나요? 특히, 두 층의 확률을 서로 바꾸면 어떤 일이 일어나나요? 이러한 질문에 답하기 위한 실험을 설계하고, 결과를 정량적으로 설명하며, 정성적 시사점을 요약하세요.
1. 에포크 수를 늘리고 드롭아웃을 사용했을 때와 사용하지 않았을 때의 결과를 비교하세요.
1. 드롭아웃을 적용했을 때와 적용하지 않았을 때 각 은닉층의 활성화의 분산은 무엇인가요? 두 모델에 대해 이 양이 시간에 따라 어떻게 진화하는지 보여주는 그래프를 그리세요.
1. 드롭아웃이 일반적으로 테스트 시 사용되지 않는 이유는 무엇인가요?
1. 이 절의 모델을 예로 들어, 드롭아웃과 가중치 감쇠를 사용하는 것의 효과를 비교하세요. 드롭아웃과 가중치 감쇠를 동시에 사용하면 어떤 일이 일어나나요? 결과는 가산적인가요? 수확 체감 ((또는 그보다 더 나쁜)) 현상이 있나요? 그들이 서로 상쇄되나요?
1. 활성화 대신 가중치 행렬의 개별 가중치에 드롭아웃을 적용하면 어떤 일이 일어나나요?
1. 표준 드롭아웃 기법과는 다른, 각 층에서 무작위 잡음을 주입하는 또 다른 기법을 발명하세요. ((고정된 구조에 대해)) Fashion-MNIST 데이터셋에서 드롭아웃을 능가하는 방법을 개발할 수 있나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17987)
:end_tab:
