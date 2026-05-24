```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 가중치 감쇠
:label:`sec_weight_decay`

이제 과적합 문제를 정의했으니, 첫 번째 *정규화* 기법을 소개할 수 있습니다.
저희는 언제든지 더 많은 훈련 데이터를 수집하여
과적합을 완화할 수 있다는 점을 떠올려 봅시다.
그러나 그 작업은 비용이 많이 들거나, 시간이 오래 걸리거나,
저희가 통제할 수 있는 범위를 완전히 벗어나기도 해서
단기적으로는 불가능한 경우가 있습니다.
지금은 저희가 가진 자원이 허락하는 범위 안에서
이미 가능한 한 많은 고품질 데이터를 확보했다고 가정하고,
데이터셋이 주어졌다고 했을 때
저희가 활용할 수 있는 도구에 집중하겠습니다.

다항 회귀 예제(:numref:`subsec_polynomial-curve-fitting`)에서
저희는 적합되는 다항식의 차수를 조정함으로써
모델의 표현 용량(capacity)을 제한할 수 있었음을 떠올려 봅시다.
실제로 특징(feature)의 수를 제한하는 것은
과적합을 완화하기 위한 인기 있는 기법입니다.
그러나 단순히 특징을 버리는 것은
지나치게 무딘 도구일 수 있습니다.
다항 회귀 예제를 계속 들자면,
고차원 입력이 주어졌을 때 어떤 일이 벌어질 수 있는지 생각해 봅시다.
다변량 데이터로의 다항식의 자연스러운 확장은
*단항식(monomial)*이라고 하며, 이는 단순히
변수 거듭제곱들의 곱입니다.
단항식의 차수는 거듭제곱들의 합입니다.
예를 들어, $x_1^2 x_2$와 $x_3 x_5^2$은
모두 차수 3의 단항식입니다.

$d$가 커짐에 따라 차수 $d$의 항의 개수가
급격하게 증가한다는 점에 유의하세요.
$k$개의 변수가 주어지면, 차수 $d$의 단항식의 개수는
${k - 1 + d} \choose {k - 1}$입니다.
차수의 작은 변화, 예를 들어 $2$에서 $3$으로의 변화조차도
저희 모델의 복잡도를 극적으로 증가시킵니다.
따라서 저희는 함수의 복잡도를 조정하기 위해
더 세밀한 도구가 필요한 경우가 많습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
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
import optax
```

## 노름과 가중치 감쇠

(**파라미터의 개수를 직접 조작하는 대신,
*가중치 감쇠*는 파라미터가 가질 수 있는 값을
제한하는 방식으로 동작합니다.**)
딥러닝 분야 밖에서는 미니배치 확률적 경사 하강법으로 최적화될 때
$\ell_2$ 정규화라고 더 흔히 부르는 가중치 감쇠는
파라미터 기반 머신러닝 모델을 정규화하기 위해
가장 널리 사용되는 기법일 것입니다.
이 기법은 모든 함수 $f$ 중에서
함수 $f = 0$(모든 입력에 값 $0$을 할당하는 함수)이
어떤 의미에서 *가장 단순한* 함수이며,
함수의 복잡도는 그 파라미터가 0에서 떨어진 거리로
측정할 수 있다는 기본적인 직관에서 동기를 얻었습니다.
그렇다면 함수와 0 사이의 거리는
정확히 어떻게 측정해야 할까요?
단 하나의 정답은 없습니다.
실제로 함수해석학의 일부와 바나흐 공간 이론 등
수학의 여러 분야 전체가
이러한 문제를 다루는 데 헌신하고 있습니다.

한 가지 단순한 해석은
선형 함수 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$의 복잡도를
그 가중치 벡터의 어떤 노름, 예를 들어 $\| \mathbf{w} \|^2$로 측정하는 것입니다.
저희는 :numref:`subsec_lin-algebra-norms`에서
더 일반적인 $\ell_p$ 노름의 특수한 경우인
$\ell_2$ 노름과 $\ell_1$ 노름을 소개했음을 떠올려 봅시다.
가중치 벡터를 작게 유지하기 위한 가장 흔한 방법은
손실을 최소화하는 문제에 그 노름을
페널티 항으로 추가하는 것입니다.
따라서 저희는 원래의 목적인
*훈련 레이블에 대한 예측 손실을 최소화하는 것*을
새로운 목적인
*예측 손실과 페널티 항의 합을 최소화하는 것*으로 대체합니다.
이제 저희의 가중치 벡터가 너무 커지면,
학습 알고리즘은 훈련 오류를 최소화하기보다
가중치 노름 $\| \mathbf{w} \|^2$을 최소화하는 데
집중하게 될 수 있습니다.
이것이 바로 저희가 원하는 바입니다.
코드로 설명하기 위해, 저희는
:numref:`sec_linear_regression`의 선형 회귀
이전 예제를 되살리겠습니다.
거기서 저희의 손실은 다음과 같이 주어졌습니다.

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$는 특징이고,
$y^{(i)}$는 임의의 데이터 예제 $i$의 레이블이며, $(\mathbf{w}, b)$는
각각 가중치와 편향 파라미터임을 떠올려 봅시다.
가중치 벡터의 크기에 페널티를 부과하려면,
어떻게든 $\| \mathbf{w} \|^2$을 손실 함수에 추가해야 하지만,
모델은 표준 손실과 이 새로운 가산적 페널티 사이에서
어떻게 절충해야 할까요?
실제로 저희는 이 절충을 *정규화 상수* $\lambda$를 통해
특징지으며, 이는 검증 데이터를 이용해 적합하는
음이 아닌 하이퍼파라미터입니다.

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


$\lambda = 0$이면 원래의 손실 함수를 회복합니다.
$\lambda > 0$이면 $\| \mathbf{w} \|$의 크기를 제한합니다.
관례적으로 $2$로 나누는데,
이차 함수의 도함수를 취할 때
$2$와 $1/2$가 상쇄되어 업데이트 식이
보기 좋고 단순해지도록 보장하기 위함입니다.
명민한 독자라면 저희가 왜 표준 노름(즉, 유클리드 거리)이 아니라
제곱 노름으로 작업하는지 궁금해할 수 있습니다.
저희가 이렇게 하는 것은 계산상의 편의를 위해서입니다.
$\ell_2$ 노름을 제곱함으로써 저희는 제곱근을 없애고,
가중치 벡터의 각 성분의 제곱의 합만 남겨둡니다.
이로 인해 페널티의 도함수를 계산하기 쉬워집니다.
도함수의 합은 합의 도함수와 같기 때문입니다.


또한 저희가 왜 애초에 $\ell_2$ 노름으로 작업하고,
예를 들어 $\ell_1$ 노름으로 하지 않는지 궁금할 수 있습니다.
사실 다른 선택지들도 유효하며
통계학 전반에 걸쳐 인기가 있습니다.
$\ell_2$ 정규화된 선형 모델이 고전적인
*릿지 회귀(ridge regression)* 알고리즘을 구성하는 반면,
$\ell_1$ 정규화된 선형 회귀는
통계학에서 마찬가지로 근본적인 방법이며
*라쏘 회귀(lasso regression)*로 널리 알려져 있습니다.
$\ell_2$ 노름으로 작업하는 한 가지 이유는
가중치 벡터의 큰 성분에
과도하게 큰 페널티를 부과하기 때문입니다.
이는 저희의 학습 알고리즘이 가중치를
더 많은 수의 특징에 고르게 분배하는 모델 쪽으로
편향되도록 만듭니다.
실제로 이는 모델을 단일 변수의 측정 오류에
더 견고하게 만들 수 있습니다.
반대로, $\ell_1$ 페널티는 다른 가중치들을 0으로
지워버림으로써 가중치를 소수의 특징에 집중시키는
모델로 이어집니다.
이는 *특징 선택(feature selection)*을 위한 효과적인 방법을 제공하며,
이는 다른 이유들로 바람직할 수도 있습니다.
예를 들어, 저희 모델이 소수의 특징에만 의존한다면,
다른(제거된) 특징들에 대한 데이터를
수집, 저장, 전송할 필요가 없을 수도 있습니다.

:eqref:`eq_linreg_batch_update`와 동일한 표기법을 사용하여,
$\ell_2$ 정규화된 회귀에 대한 미니배치 확률적 경사 하강법
업데이트는 다음과 같이 진행됩니다.

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

이전과 마찬가지로, 저희는 추정값이 관측값에서 벗어난 양만큼
$\mathbf{w}$를 업데이트합니다.
그러나 동시에 $\mathbf{w}$의 크기를 0 쪽으로 축소시키기도 합니다.
이것이 이 방법이 때때로 "가중치 감쇠"라고 불리는 이유입니다.
페널티 항만을 놓고 보면,
저희의 최적화 알고리즘은 훈련의 각 단계에서
가중치를 *감쇠*시킵니다.
특징 선택과 대조적으로,
가중치 감쇠는 저희에게 함수의 복잡도를 연속적으로 조정할 수 있는 메커니즘을 제공합니다.
$\lambda$ 값이 작을수록 $\mathbf{w}$가 덜 제약되는 데 해당하고,
$\lambda$ 값이 클수록
$\mathbf{w}$를 훨씬 더 많이 제약합니다.
이에 상응하는 편향 페널티 $b^2$를 포함할지 여부는
구현마다 다를 수 있고,
신경망의 층마다 다를 수도 있습니다.
종종 저희는 편향 항을 정규화하지 않습니다.
또한,
다른 최적화 알고리즘에서는 $\ell_2$ 정규화가 가중치 감쇠와 동등하지 않을 수 있지만,
가중치 크기를 축소하는 것을 통한 정규화의 아이디어는
여전히 유효합니다.

## 고차원 선형 회귀

저희는 간단한 합성 예제를 통해
가중치 감쇠의 이점을 설명할 수 있습니다.

먼저, [**이전과 같이 데이터를 생성합니다**].

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

이 합성 데이터셋에서 저희의 레이블은 입력에 대한 잠재적인 선형 함수로 주어지며,
평균이 0이고 표준편차가 0.01인 가우시안 잡음에 의해
오염되어 있습니다.
설명을 위해, 저희는 문제의 차원을 $d = 200$으로 늘리고
예제가 20개뿐인 작은 훈련 세트로 작업함으로써
과적합의 효과를 두드러지게 만들 수 있습니다.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## 처음부터 구현하기

이제 가중치 감쇠를 처음부터 구현해 봅시다.
저희의 옵티마이저가 미니배치 확률적 경사 하강법이므로,
원래의 손실 함수에 제곱 $\ell_2$ 페널티를
추가하기만 하면 됩니다.

### (**$\ell_2$ 노름 페널티 정의하기**)

아마도 이 페널티를 구현하는 가장 편리한 방법은
모든 항을 그 자리에서 제곱한 뒤 합산하는 것입니다.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### 모델 정의하기

최종 모델에서 선형 회귀와 제곱 손실은
:numref:`sec_linear_scratch`에서 변경되지 않았으므로,
저희는 `d2l.LinearRegressionScratch`의 서브클래스를 정의하기만 하면 됩니다. 여기서 유일한 변경 사항은 손실에 이제 페널티 항이 포함된다는 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

다음 코드는 예제 20개로 이루어진 훈련 세트에서 모델을 적합하고, 예제 100개로 이루어진 검증 세트에서 모델을 평가합니다.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**정규화 없이 훈련하기**]

이제 가중치 감쇠를 비활성화하여 `lambd = 0`으로
이 코드를 실행합니다.
저희가 심하게 과적합하여 훈련 오류는 감소하지만
검증 오류는 감소하지 않는다는 점에 주의하세요(교과서적 과적합 사례입니다).

```{.python .input}
%%tab all
train_scratch(0)
```

### [**가중치 감쇠 사용하기**]

아래에서 저희는 상당한 가중치 감쇠를 적용하여 실행합니다.
훈련 오류는 증가하지만
검증 오류는 감소한다는 점에 주의하세요.
이것이 바로 저희가 정규화로부터
기대하는 효과입니다.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**간결한 구현**]

가중치 감쇠는 신경망 최적화에서 어디에나 쓰이기 때문에,
딥러닝 프레임워크는 이를 특히 편리하게 만들며,
가중치 감쇠를 최적화 알고리즘 자체에 통합하여
임의의 손실 함수와 함께 쉽게 사용할 수 있도록 합니다.
나아가, 이 통합은 계산상의 이점도 제공하여,
추가적인 계산 부담 없이 가중치 감쇠를 알고리즘에 추가하는
구현 기법을 가능하게 합니다.
업데이트의 가중치 감쇠 부분은
각 파라미터의 현재 값에만 의존하기 때문에,
어차피 옵티마이저는 각 파라미터를 한 번씩 건드려야 합니다.

:begin_tab:`mxnet`
아래에서 저희는 `Trainer`를 인스턴스화할 때
`wd`를 통해 가중치 감쇠 하이퍼파라미터를
직접 지정합니다.
기본적으로 Gluon은 가중치와 편향을
동시에 감쇠시킵니다.
모델 파라미터를 업데이트할 때
하이퍼파라미터 `wd`가 `wd_mult`와
곱해진다는 점에 주의하세요.
따라서 `wd_mult`를 0으로 설정하면
편향 파라미터 $b$는 감쇠하지 않게 됩니다.
:end_tab:

:begin_tab:`pytorch`
아래에서 저희는 옵티마이저를 인스턴스화할 때
`weight_decay`를 통해 가중치 감쇠 하이퍼파라미터를
직접 지정합니다.
기본적으로 PyTorch는 가중치와 편향을 동시에 감쇠시키지만,
서로 다른 정책에 따라 서로 다른 파라미터를 다루도록
옵티마이저를 구성할 수 있습니다.
여기서 저희는 가중치(`net.weight` 파라미터)에만
`weight_decay`를 설정하며, 따라서
편향(`net.bias` 파라미터)은 감쇠하지 않게 됩니다.
:end_tab:

:begin_tab:`tensorflow`
아래에서 저희는 가중치 감쇠 하이퍼파라미터 `wd`를 가진
$\ell_2$ 정규화기를 만들고, 이를 `kernel_regularizer` 인수를 통해
층의 가중치에 적용합니다.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**이 그림은 저희가 가중치 감쇠를 처음부터 구현했을 때의 그림과
비슷해 보입니다**].
그러나 이 버전은 더 빠르게 실행되며
구현하기도 더 쉽습니다. 더 큰 문제를 다루고
이 작업이 더 일상화될수록 이러한 이점이 더욱 두드러질 것입니다.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

지금까지 저희는 단순한 선형 함수가 무엇인지에 대한
한 가지 개념을 다뤘습니다.
그러나 단순한 비선형 함수의 경우에도 상황은 훨씬 더 복잡해질 수 있습니다. 이를 알아보기 위해, [재생 커널 힐베르트 공간(reproducing kernel Hilbert space, RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)이라는 개념은
선형 함수에 대해 소개된 도구들을
비선형 맥락에서 적용할 수 있게 해 줍니다.
불행히도 RKHS 기반 알고리즘은
크고 고차원적인 데이터에는 잘 확장되지 않는 경향이 있습니다.
이 책에서 저희는 가중치 감쇠를
딥 네트워크의 모든 층에 적용하는 일반적인 휴리스틱을
종종 채택할 것입니다.

## 요약

정규화는 과적합을 다루기 위한 일반적인 방법입니다. 고전적인 정규화 기법은 학습된 모델의 복잡도를 줄이기 위해 (훈련 시) 손실 함수에 페널티 항을 추가합니다.
모델을 단순하게 유지하기 위한 한 가지 특별한 선택은 $\ell_2$ 페널티를 사용하는 것입니다. 이는 미니배치 확률적 경사 하강법 알고리즘의 업데이트 단계에서 가중치 감쇠로 이어집니다.
실제로 가중치 감쇠 기능은 딥러닝 프레임워크의 옵티마이저에서 제공됩니다.
서로 다른 파라미터 집합은 동일한 훈련 루프 안에서 서로 다른 업데이트 동작을 가질 수 있습니다.



## 연습문제

1. 이 절의 추정 문제에서 $\lambda$ 값을 가지고 실험해 보세요. 훈련과 검증 정확도를 $\lambda$의 함수로 그려 보세요. 무엇이 관찰되나요?
1. $\lambda$의 최적값을 찾기 위해 검증 세트를 사용해 보세요. 그것이 정말로 최적값인가요? 이것이 중요한가요?
1. 저희가 페널티 선택으로 $\|\mathbf{w}\|^2$ 대신 $\sum_i |w_i|$를 사용한다면($\ell_1$ 정규화) 업데이트 방정식이 어떻게 생겼을까요?
1. 저희는 $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$임을 알고 있습니다. 행렬에 대해 비슷한 식을 찾을 수 있나요(:numref:`subsec_lin-algebra-norms`의 프로베니우스 노름을 참조하세요)?
1. 훈련 오류와 일반화 오류 사이의 관계를 복습해 보세요. 가중치 감쇠, 훈련 증가, 적절한 복잡도의 모델 사용 외에, 과적합을 다루는 데 도움이 될 수 있는 다른 방법으로 무엇이 있을까요?
1. 베이즈 통계학에서 저희는 $P(w \mid x) \propto P(x \mid w) P(w)$를 통해 사후 분포에 도달하기 위해 사전과 가능도의 곱을 사용합니다. $P(w)$를 정규화와 어떻게 동일시할 수 있나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17979)
:end_tab:
