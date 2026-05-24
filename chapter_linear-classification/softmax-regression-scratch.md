```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 소프트맥스 회귀 처음부터 구현하기
:label:`sec_softmax_scratch`

소프트맥스 회귀가 매우 기초적이기 때문에,
저희는 여러분이 직접 구현하는 방법을
알아야 한다고 생각합니다.
여기서는 모델의 소프트맥스에 특화된 측면을 정의하는 데 자신을 한정하고,
학습 루프를 비롯한
다른 구성 요소는 선형 회귀 절에서 그대로 재사용합니다.

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
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## 소프트맥스

가장 중요한 부분부터 시작합시다.
스칼라에서 확률로의 매핑입니다.
복습 차원에서, :numref:`subsec_lin-alg-reduction`과
:numref:`subsec_lin-alg-non-reduction`에서 논의한 바와 같이
텐서의 특정 차원을 따른 합 연산자의 동작을 떠올려 보시기 바랍니다.
[**행렬 `X`가 주어졌을 때 (기본적으로) 모든 원소에 대해 합을 구하거나
같은 축에 있는 원소들에 대해서만 합을 구할 수 있습니다.**]
`axis` 변수를 사용하면 행과 열의 합을 계산할 수 있습니다.

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

소프트맥스를 계산하려면 세 단계가 필요합니다.
(i) 각 항에 지수 취하기,
(ii) 각 예제에 대한 정규화 상수를 계산하기 위해 각 행에 대한 합,
(iii) 결과가 1이 되도록 각 행을 그 정규화 상수로 나누기.

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

분모(의 로그)는 (로그) *분배 함수(partition function)*라고 부릅니다.
이는 [통계물리학](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))에서
열역학적 앙상블의 모든 가능한 상태에 대해 합을 구하기 위해 도입되었습니다.
구현은 간단합니다.

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

어떤 입력 `X`에 대해서도 [**각 원소를
음이 아닌 수로 변환합니다.
각 행은 합이 1이 되며,**]
이는 확률에 요구되는 조건입니다. 주의: 위 코드는 매우 크거나 매우 작은 인수에 대해 견고하지 *않습니다*. 무슨 일이 일어나는지 보여주는 데는 충분하지만, 진지한 용도로는 이 코드를 글자 그대로 사용해서는 *안 됩니다*. 딥러닝 프레임워크에는 이러한 보호 장치가 내장되어 있으며, 앞으로는 내장 소프트맥스를 사용할 것입니다.

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## 모델

이제 [**소프트맥스 회귀 모델**]을 구현하는 데
필요한 모든 것이 있습니다.
선형 회귀 예제에서와 마찬가지로,
각 인스턴스는 고정 길이의 벡터로 표현될 것입니다.
여기서 원시 데이터는 $28 \times 28$ 픽셀 이미지로 구성되어 있으므로,
[**저희는 각 이미지를 평탄화(flatten)하여
길이 784의 벡터로 다룹니다.**]
이후 챕터에서 합성곱 신경망을 소개할 것인데,
이는 공간적 구조를 더 만족스러운 방식으로 활용합니다.


소프트맥스 회귀에서,
저희 신경망의 출력 수는
클래스의 수와 같아야 합니다.
(**데이터셋에 10개의 클래스가 있으므로,
저희 신경망은 출력 차원 10을 가집니다.**)
따라서 가중치는 $784 \times 10$ 행렬과
편향을 위한 $1 \times 10$ 행 벡터로 구성됩니다.
선형 회귀에서와 마찬가지로,
가중치 `W`를 가우시안 잡음으로 초기화합니다.
편향은 0으로 초기화됩니다.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

아래 코드는 신경망이 각 입력을
출력으로 어떻게 매핑하는지를 정의합니다.
데이터를 모델에 통과시키기 전에 `reshape`를 사용하여
배치 내 각 $28 \times 28$ 픽셀 이미지를
벡터로 평탄화한다는 점에 유의하시기 바랍니다.

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## 교차 엔트로피 손실

다음으로 교차 엔트로피 손실 함수
(:numref:`subsec_softmax-regression-loss-func`에서 소개됨)를 구현해야 합니다.
이는 모든 딥러닝을 통틀어 가장 흔한 손실 함수일 것입니다.
현재 분류 문제로 쉽게 다룰 수 있는 딥러닝 응용 분야는
회귀 문제로 더 잘 다뤄지는 응용 분야보다 훨씬 많습니다.

교차 엔트로피는 실제 레이블에 할당된 예측 확률의
음의 로그 가능도를 취한다는 것을 떠올려 보시기 바랍니다.
효율성을 위해 파이썬 for-루프를 피하고 대신 인덱싱을 사용합니다.
특히, $\mathbf{y}$의 원-핫 인코딩은
$\hat{\mathbf{y}}$에서 일치하는 항을 선택할 수 있게 해 줍니다.

이 동작을 보기 위해 3개 클래스에 대한 예측 확률 2개 예제와 그에 대응되는 레이블 `y`를 가진
[**예제 데이터 `y_hat`을 만들어 봅시다.**]
정답 레이블은 각각 $0$과 $2$입니다(즉, 첫 번째와 세 번째 클래스).
[**`y`를 `y_hat`의 확률에 대한 인덱스로 사용하면,**]
효율적으로 항을 선택할 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
이제 선택된 확률의 로그에 대해 평균을 내어 (**교차 엔트로피 손실 함수를 구현**)할 수 있습니다.
:end_tab:

:begin_tab:`jax`
이제 선택된 확률의 로그에 대해 평균을 내어 (**교차 엔트로피 손실 함수를 구현**)할 수 있습니다.

JAX 구현의 속도를 높이기 위해 `jax.jit`을 활용하고,
`loss`가 순수 함수임을 보장하기 위해 `cross_entropy` 함수는
`loss` 내부에 재정의되어 있습니다. 이는 `loss` 함수를 비순수하게 만들 수 있는
전역 변수나 함수의 사용을 피하기 위함입니다.
관심 있는 독자들은 `jax.jit`과 순수 함수에 대해 [JAX 문서](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)를 참고하시기 바랍니다.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return cross_entropy(y_hat, y), {}
```

## 학습

:numref:`sec_linear_scratch`에서 정의된 `fit` 메서드를 재사용하여 [**모델을 10 에폭 동안 학습합니다.**]
에폭 수(`max_epochs`),
미니배치 크기(`batch_size`),
그리고 학습률(`lr`)은 조정 가능한 하이퍼파라미터라는 점에 유의하시기 바랍니다.
이는 이 값들이 주 학습 루프 동안에는 학습되지 않지만,
모델의 학습 성능과 일반화 성능 모두에 영향을 미친다는 것을 의미합니다.
실제로는 데이터의 *검증(validation)* 분할을 기반으로 이 값들을 선택하고,
궁극적으로는 *테스트(test)* 분할에서 최종 모델을 평가하고자 할 것입니다.
:numref:`subsec_generalization-model-selection`에서 논의했듯이,
저희는 Fashion-MNIST의 테스트 데이터를 검증 셋으로 간주하므로,
이 분할에 대해 검증 손실과 검증 정확도를 보고합니다.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 예측

이제 학습이 완료되었으니,
저희 모델은 [**일부 이미지를 분류할**] 준비가 되었습니다.

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

저희는 *잘못된* 레이블이 붙은 이미지에 더 관심이 있습니다. 실제 레이블(텍스트 출력의 첫 번째 줄)을
모델의 예측(텍스트 출력의 두 번째 줄)과
비교하여 시각화합니다.

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## 요약

지금쯤이면 선형 회귀와
분류 문제를 푸는 데 어느 정도의
경험을 쌓기 시작했을 것입니다.
이로써 저희는 1960-1970년대 통계 모델링의
최첨단이라고 할 수 있는 수준에 도달했습니다.
다음 절에서는 딥러닝 프레임워크를 활용하여
이 모델을 훨씬 더 효율적으로
구현하는 방법을 보여드리겠습니다.

## 연습문제

1. 이 절에서 저희는 소프트맥스 연산의 수학적 정의에 기반하여 소프트맥스 함수를 직접 구현했습니다. :numref:`sec_softmax`에서 논의했듯이 이는 수치적 불안정성을 일으킬 수 있습니다.
    1. 입력이 $100$의 값을 갖는 경우에도 `softmax`가 여전히 정확하게 작동하는지 테스트하시오.
    1. 모든 입력 중 가장 큰 값이 $-100$보다 작은 경우에도 `softmax`가 여전히 정확하게 작동하는지 테스트하시오.
    1. 인수의 가장 큰 항목을 기준으로 상대적인 값을 보아 수정 방법을 구현하시오.
1. 교차 엔트로피 손실 함수 $\sum_i y_i \log \hat{y}_i$의 정의를 따르는 `cross_entropy` 함수를 구현하시오.
    1. 이 절의 코드 예제에서 시도해 보시기 바랍니다.
    1. 왜 더 느리게 실행된다고 생각하나요?
    1. 이를 사용해야 할까요? 언제 사용하는 것이 합리적일까요?
    1. 무엇을 주의해야 할까요? 힌트: 로그의 정의역을 고려해 보시기 바랍니다.
1. 항상 가장 가능성 높은 레이블을 반환하는 것이 좋은 생각일까요? 예를 들어, 의료 진단의 경우에 이렇게 할 것인가요? 이 문제를 어떻게 해결하려고 시도할 수 있을까요?
1. 일부 특성을 기반으로 다음 단어를 예측하기 위해 소프트맥스 회귀를 사용하고자 한다고 가정해 봅시다. 큰 어휘에서 발생할 수 있는 몇 가지 문제는 무엇인가요?
1. 이 절의 코드의 하이퍼파라미터로 실험해 보시기 바랍니다. 특히 다음을 시도해 보시기 바랍니다.
    1. 학습률을 변경함에 따라 검증 손실이 어떻게 변하는지 그려 보시기 바랍니다.
    1. 미니배치 크기를 변경함에 따라 검증 손실과 학습 손실이 변하는지요? 효과를 보려면 얼마나 크게 또는 작게 해야 하나요?


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17982)
:end_tab:
