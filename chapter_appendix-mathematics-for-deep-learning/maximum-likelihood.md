# 최대 가능도
:label:`sec_maximum_likelihood`

머신러닝에서 가장 일반적으로 마주치는 사고방식 중 하나는 최대 가능도 관점입니다. 이는 알려지지 않은 매개변수를 가진 확률적 모델로 작업할 때, 데이터가 가장 높은 확률을 가지도록 만드는 매개변수가 가장 가능성 있는 것이라는 개념입니다.

## 최대 가능도 원리

이는 생각하는 데 도움이 될 수 있는 베이지안 해석을 가집니다. 매개변수 $\boldsymbol{\theta}$와 데이터 예제 모음 $X$를 가진 모델이 있다고 가정해 보십시오. 구체적으로 말하면, $\boldsymbol{\theta}$를 동전을 던졌을 때 앞면이 나올 확률을 나타내는 단일 값으로, $X$를 독립적인 동전 던지기의 시퀀스로 상상할 수 있습니다. 이 예제는 나중에 깊이 살펴볼 것입니다.

만약 저희 모델의 매개변수에 대한 가장 가능성 있는 값을 찾고 싶다면, 이는 다음을 찾고 싶다는 것을 의미합니다.

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

베이즈 규칙에 의해, 이는 다음과 같은 것입니다.

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

식 $P(X)$, 데이터를 생성하는 매개변수에 무관한 확률은, $\boldsymbol{\theta}$에 전혀 의존하지 않으므로, $\boldsymbol{\theta}$의 최선의 선택을 변경하지 않고 떨어뜨릴 수 있습니다. 유사하게, 어떤 매개변수 집합이 다른 어떤 것보다 더 낫다는 사전 가정이 없다고 이제 상정할 수 있으므로, $P(\boldsymbol{\theta})$도 theta에 의존하지 않는다고 선언할 수 있습니다! 이는 예를 들어 동전 던지기 예제에서 그것이 공정한지 아닌지에 대한 어떤 사전 믿음 없이 앞면이 나올 확률이 $[0,1]$의 어떤 값이든 될 수 있는(종종 *무정보 사전*이라고 함) 경우 의미가 있습니다. 따라서 베이즈 규칙의 저희의 적용은 $\boldsymbol{\theta}$의 최선의 선택이 $\boldsymbol{\theta}$에 대한 최대 가능도 추정임을 보여줍니다.

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

일반적인 용어 문제로, 매개변수가 주어졌을 때의 데이터의 확률($P(X \mid \boldsymbol{\theta})$)은 *가능도*라고 합니다.

### 구체적인 예제

구체적인 예제에서 이것이 어떻게 작동하는지 봅시다. 동전 던지기가 앞면일 확률을 나타내는 단일 매개변수 $\theta$가 있다고 가정해 봅시다. 그러면 뒷면이 나올 확률은 $1-\theta$이며, 만약 저희의 관찰된 데이터 $X$가 $n_H$개의 앞면과 $n_T$개의 뒷면을 가진 시퀀스라면, 저희는 독립 확률이 곱한다는 사실을 사용하여 다음을 볼 수 있습니다.

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

만약 $13$개의 동전을 던져 $n_H = 9$이고 $n_T = 4$인 시퀀스 "HHHTHTTHHHHHT"를 얻는다면, 이는 다음과 같음을 봅니다.

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

이 예제의 좋은 점 중 하나는 저희가 들어가는 답을 안다는 것입니다. 사실, 만약 저희가 구두로 "저는 13개의 동전을 던졌고, 9개가 앞면이 나왔습니다. 동전이 앞면이 나올 확률에 대한 저희의 최선의 추측은 무엇입니까?"라고 말한다면, 모두가 올바르게 $9/13$을 추측할 것입니다. 이 최대 가능도 방법이 저희에게 제공하는 것은 훨씬 더 복잡한 상황으로 일반화될 수 있는 방식으로 제1원리에서 그 숫자를 얻는 방법입니다.

저희의 예제의 경우, $P(X \mid \theta)$의 플롯은 다음과 같습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

이는 저희가 기대한 $9/13 \approx 0.7\ldots$ 근처에서 최댓값을 가집니다. 그것이 정확히 거기에 있는지 보기 위해, 미적분으로 향할 수 있습니다. 최댓값에서, 함수의 그래디언트가 평평하다는 점에 유의하십시오. 따라서, 도함수가 0인 $\theta$의 값을 찾고, 가장 높은 확률을 주는 것을 찾음으로써 최대 가능도 추정 :eqref:`eq_max_like`를 찾을 수 있습니다. 저희는 다음을 계산합니다.

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

이는 세 가지 해를 가집니다. $0$, $1$ 그리고 $9/13$. 처음 두 개는 분명히 최댓값이 아닌 최솟값입니다. 왜냐하면 그들은 저희의 시퀀스에 확률 $0$을 부여하기 때문입니다. 마지막 값은 저희의 시퀀스에 0인 확률을 부여하지 *않으며*, 따라서 최대 가능도 추정 $\hat \theta = 9/13$이어야 합니다.

## 수치적 최적화와 음의 로그 가능도

이전 예제는 좋지만, 만약 수십억 개의 매개변수와 데이터 예제가 있다면 어떨까요?

먼저, 모든 데이터 예제가 독립이라는 가정을 한다면, 가능도 자체는 많은 확률의 곱이기 때문에 더 이상 실용적으로 고려할 수 없다는 점에 유의하십시오. 사실, 각 확률은 $[0,1]$에 있으며, 전형적으로 약 $1/2$의 값이라고 합시다. 그리고 $(1/2)^{1000000000}$의 곱은 기계 정밀도보다 훨씬 아래입니다. 저희는 그것을 직접 다룰 수 없습니다.

그러나, 로그가 곱을 합으로 바꾼다는 것을 떠올리십시오. 이 경우

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

이 숫자는 단일 정밀도 $32$비트 float에도 완벽하게 들어맞습니다. 따라서, 저희는 *로그 가능도*를 고려해야 합니다. 이는 다음과 같습니다.

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

함수 $x \mapsto \log(x)$가 증가하므로, 가능도를 최대화하는 것은 로그 가능도를 최대화하는 것과 같은 것입니다. 사실 :numref:`sec_naive_bayes`에서 나이브 베이즈 분류기의 특정 예제로 작업할 때 이 추론이 적용되는 것을 볼 것입니다.

저희는 종종 손실 함수로 작업하는데, 여기서 손실을 최소화하고 싶습니다. 저희는 $-\log(P(X \mid \boldsymbol{\theta}))$를 취함으로써 최대 가능도를 손실의 최소화로 바꿀 수 있는데, 이는 *음의 로그 가능도*입니다.

이를 설명하기 위해, 이전의 동전 던지기 문제를 고려하고, 닫힌 형식의 해를 모른다고 가장해 봅시다. 저희는 다음을 계산할 수 있습니다.

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

이는 코드로 작성될 수 있고, 수십억 개의 동전 던지기에 대해서도 자유롭게 최적화될 수 있습니다.

```{.python .input}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

수치적 편의성이 사람들이 음의 로그 가능도를 사용하는 것을 좋아하는 유일한 이유는 아닙니다. 그것이 선호되는 몇 가지 다른 이유가 있습니다.



로그 가능도를 고려하는 두 번째 이유는 미적분 규칙의 단순화된 적용입니다. 위에서 논의된 것처럼, 독립성 가정으로 인해, 머신러닝에서 마주치는 대부분의 확률은 개별 확률의 곱입니다.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

이는 도함수를 계산하기 위해 곱 규칙을 직접 적용하면 다음을 얻는다는 것을 의미합니다.

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

이는 $(n-1)$ 덧셈과 함께 $n(n-1)$ 곱셈을 필요로 하므로, 입력에서 이차 시간에 비례합니다! 항을 그룹화하는 데 충분한 영리함이 있으면 이를 선형 시간으로 줄일 수 있지만, 약간의 사고가 필요합니다. 음의 로그 가능도의 경우 대신 다음을 가집니다.

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

이는 그러면 다음을 제공합니다.

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

이는 단지 $n$개의 나눗셈과 $n-1$개의 합만을 필요로 하므로, 입력에서 선형 시간입니다.

음의 로그 가능도를 고려하는 세 번째이자 마지막 이유는 정보 이론과의 관계이며, :numref:`sec_information_theory`에서 자세히 논의할 것입니다. 이는 확률 변수에서 정보 또는 무작위성의 정도를 측정하는 방법을 제공하는 엄격한 수학 이론입니다. 그 분야에서 연구의 핵심 객체는 엔트로피이며, 이는 다음과 같습니다.

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

이는 소스의 무작위성을 측정합니다. 이는 평균 $-\log$ 확률에 불과하다는 점에 유의하십시오. 따라서 만약 저희가 음의 로그 가능도를 취하고 데이터 예제의 수로 나누면, 교차 엔트로피로 알려진 엔트로피의 친척을 얻습니다. 이 이론적 해석만으로도 모델 성능을 측정하는 방법으로 데이터셋에 대한 평균 음의 로그 가능도를 보고하도록 동기를 부여하기에 충분히 설득력 있을 것입니다.

## 연속 변수에 대한 최대 가능도

지금까지 저희가 한 모든 것은 이산 확률 변수로 작업하는 것을 가정하지만, 만약 연속적인 것으로 작업하고 싶다면 어떨까요?

짧은 요약은 확률을 확률 밀도로 대체하는 것을 제외하고는 전혀 아무것도 변경되지 않는다는 것입니다. 밀도를 소문자 $p$로 작성한다는 것을 떠올리면, 이는 예를 들어 이제 다음과 같이 말한다는 것을 의미합니다.

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

질문은 "왜 이것이 괜찮은가?"가 됩니다. 결국, 저희가 밀도를 도입한 이유는 특정 결과를 얻을 확률 자체가 0이었기 때문이며, 따라서 어떤 매개변수 집합에 대해서도 저희 데이터를 생성할 확률이 0이지 않습니까?

사실, 그것이 그렇고, 왜 저희가 밀도로 이동할 수 있는지 이해하는 것은 엡실론에 무슨 일이 일어나는지 추적하는 연습입니다.

먼저 저희의 목표를 다시 정의합시다. 연속 확률 변수의 경우, 정확히 올바른 값을 얻을 확률을 더 이상 계산하고 싶지 않고, 대신 어떤 범위 $\epsilon$ 내에서 일치하기를 원한다고 가정해 봅시다. 단순함을 위해, 저희의 데이터가 동일하게 분포된 확률 변수 $X_1, \ldots, X_N$의 반복된 관찰 $x_1, \ldots, x_N$이라고 가정합니다. 이전에 본 것처럼, 이는 다음과 같이 작성될 수 있습니다.

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

따라서, 만약 이것의 음의 로그를 취하면 다음을 얻습니다.

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

이 식을 검토하면, $\epsilon$이 발생하는 유일한 곳은 덧셈 상수 $-N\log(\epsilon)$입니다. 이는 매개변수 $\boldsymbol{\theta}$에 전혀 의존하지 않으므로, $\boldsymbol{\theta}$의 최적 선택은 $\epsilon$의 저희 선택에 의존하지 않습니다! 만약 저희가 4자리 또는 400자리를 요구하더라도, $\boldsymbol{\theta}$의 최선의 선택은 동일하게 유지되므로, 자유롭게 엡실론을 떨어뜨려 저희가 최적화하고 싶은 것이 다음임을 봅니다.

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

따라서, 저희는 최대 가능도 관점이 확률을 확률 밀도로 대체함으로써 이산 확률 변수와 마찬가지로 쉽게 연속 확률 변수로도 작동할 수 있음을 봅니다.

## 요약
* 최대 가능도 원리는 주어진 데이터셋에 대한 가장 잘 맞는 모델이 데이터를 가장 높은 확률로 생성하는 모델이라고 알려줍니다.
* 종종 사람들은 다양한 이유로 대신 음의 로그 가능도로 작업합니다. 수치적 안정성, 곱을 합으로 변환(그리고 결과로 그래디언트 계산의 단순화), 그리고 정보 이론과의 이론적 연결.
* 이산 환경에서 가장 단순하게 동기 부여되지만, 데이터 포인트에 할당된 확률 밀도를 최대화함으로써 연속 환경으로도 자유롭게 일반화될 수 있습니다.

## 연습문제
1. 음이 아닌 확률 변수가 어떤 값 $\alpha>0$에 대해 밀도 $\alpha e^{-\alpha x}$를 가진다는 것을 안다고 가정해 보십시오. 확률 변수에서 숫자 $3$인 단일 관찰을 얻습니다. $\alpha$에 대한 최대 가능도 추정은 무엇입니까?
2. 알려지지 않은 평균이지만 분산 $1$을 가진 가우시안에서 추출된 샘플 데이터셋 $\{x_i\}_{i=1}^N$이 있다고 가정해 보십시오. 평균에 대한 최대 가능도 추정은 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab:
