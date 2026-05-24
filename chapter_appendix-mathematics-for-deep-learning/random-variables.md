# 확률 변수
:label:`sec_random_variables`

:numref:`sec_prob`에서 저희는 이산 확률 변수로 작업하는 방법의 기초를 보았는데, 저희의 경우 그것들은 유한한 가능한 값 집합 또는 정수를 취하는 확률 변수를 가리킵니다. 이 절에서, 저희는 *연속 확률 변수*의 이론을 발전시키는데, 이는 어떤 실숫값이든 취할 수 있는 확률 변수입니다.

## 연속 확률 변수

연속 확률 변수는 이산 확률 변수보다 훨씬 더 미묘한 주제입니다. 적절한 비유는 기술적 점프가 숫자 리스트를 더하는 것과 함수를 적분하는 것 사이의 점프에 비교할 수 있다는 것입니다. 그러므로, 저희는 이론을 발전시키는 데 약간의 시간을 들여야 할 것입니다.

### 이산에서 연속으로

연속 확률 변수로 작업할 때 마주치는 추가 기술적 도전을 이해하기 위해, 사고 실험을 수행해 봅시다. 다트판에 다트를 던지고 있다고 가정하고, 정확히 보드의 중심에서 $2 \textrm{cm}$ 떨어진 곳을 맞출 확률을 알고 싶다고 가정해 보십시오.

시작하기 위해, 한 자릿수 정확도를 측정하는 것을 상상해 봅시다. 즉, $0 \textrm{cm}$, $1 \textrm{cm}$, $2 \textrm{cm}$ 등에 대한 빈을 가지고 말입니다. 다트판에 $100$개의 다트를 던졌다고 하고, 그 중 $20$개가 $2\textrm{cm}$ 빈에 들어가면, 저희가 던진 다트의 $20\%$가 중심에서 $2 \textrm{cm}$ 떨어진 곳을 맞췄다고 결론 내립니다.

그러나, 더 자세히 살펴보면, 이는 저희의 질문과 일치하지 않습니다! 저희는 정확한 동등성을 원했지만, 이 빈은 $1.5\textrm{cm}$와 $2.5\textrm{cm}$ 사이에 떨어진 모든 것을 담고 있습니다.

좌절하지 않고, 저희는 더 진행합니다. 저희는 더 정확하게 측정합니다. 예를 들어 $1.9\textrm{cm}$, $2.0\textrm{cm}$, $2.1\textrm{cm}$ 등으로 말입니다. 이제 $100$개의 다트 중 아마도 $3$개가 $2.0\textrm{cm}$ 버킷에 들어갔다고 봅니다. 따라서 저희는 확률이 $3\%$라고 결론 내립니다.

그러나, 이것은 아무것도 해결하지 않습니다! 저희는 단지 문제를 한 자릿수 더 아래로 밀어냈을 뿐입니다. 약간 추상화합시다. 첫 $k$자리가 $2.00000\ldots$와 일치할 확률을 알고 있고, 첫 $k+1$자리에 대해 일치할 확률을 알고 싶다고 상상해 보십시오. ${k+1}^{\textrm{th}}$ 자리가 본질적으로 집합 $\{0, 1, 2, \ldots, 9\}$에서의 무작위 선택이라고 가정하는 것은 꽤 합리적입니다. 적어도, 저희는 중심에서 떨어진 마이크로미터 수가 $3$보다 $7$로 끝나는 것을 선호하도록 강요할 물리적으로 의미 있는 과정을 상상할 수 없습니다.

이것이 의미하는 것은 본질적으로 저희가 요구하는 정확도의 각 추가 자릿수가 일치 확률을 $10$의 인자로 감소시켜야 한다는 것입니다. 또는 다르게 말하면, 저희는 다음을 기대할 것입니다.

$$
P(\textrm{distance is}\; 2.00\ldots, \;\textrm{to}\; k \;\textrm{digits} ) \approx p\cdot10^{-k}.
$$

값 $p$는 본질적으로 처음 몇 자릿수에서 무슨 일이 일어나는지를 인코딩하고, $10^{-k}$는 나머지를 처리합니다.

소수점 이후 $k=4$자리까지 정확한 위치를 안다면, 그것은 값이 예를 들어 길이 $2.00005-1.99995 = 10^{-4}$인 구간 $[1.99995,2.00005]$에 떨어진다는 것을 안다는 것을 의미합니다. 따라서 이 구간의 길이를 $\epsilon$이라고 부르면, 저희는 다음과 같이 말할 수 있습니다.

$$
P(\textrm{distance is in an}\; \epsilon\textrm{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

이를 한 단계 마지막으로 더 진행해 봅시다. 저희는 내내 점 $2$에 대해 생각해 왔지만, 다른 점에 대해서는 결코 생각하지 않았습니다. 거기에는 근본적으로 다른 것이 없지만, 값 $p$는 다를 가능성이 높습니다. 저희는 적어도 다트 던지는 사람이 $20\textrm{cm}$보다 $2\textrm{cm}$와 같이 중심에 가까운 점을 맞출 가능성이 더 높기를 바랄 것입니다. 따라서, 값 $p$는 고정되어 있지 않고, 오히려 점 $x$에 의존해야 합니다. 이는 저희가 다음을 기대해야 한다고 알려줍니다.

$$P(\textrm{distance is in an}\; \epsilon \textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

사실, :eqref:`eq_pdf_deriv`는 정확히 *확률 밀도 함수*를 정의합니다. 이는 한 점 대 다른 점 근처를 맞추는 상대적 확률을 인코딩하는 함수 $p(x)$입니다. 그러한 함수가 어떻게 보일 수 있는지 시각화해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

함수 값이 큰 위치는 저희가 무작위 값을 찾을 가능성이 더 높은 영역을 나타냅니다. 낮은 부분은 저희가 무작위 값을 찾을 가능성이 낮은 영역입니다.

### 확률 밀도 함수

이제 이를 더 조사해 봅시다. 저희는 이미 확률 변수 $X$에 대한 확률 밀도 함수가 무엇인지 직관적으로 보았습니다. 즉, 밀도 함수는 다음과 같은 함수 $p(x)$입니다.

$$P(X \; \textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

그러나 이것이 $p(x)$의 속성에 대해 무엇을 의미합니까?

첫째, 확률은 결코 음수가 아니므로, 저희는 $p(x) \ge 0$도 기대해야 합니다.

둘째, 저희가 $\mathbb{R}$을 $\epsilon$ 너비의 무한한 수의 슬라이스로 잘랐다고 상상해 봅시다. 예를 들어 슬라이스 $(\epsilon\cdot i, \epsilon \cdot (i+1)]$과 함께 말입니다. 이들 각각에 대해, :eqref:`eq_pdf_def`로부터 저희는 확률이 대략 다음과 같다는 것을 압니다.

$$
P(X \; \textrm{is in an}\; \epsilon\textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

따라서 그것들 모두에 대해 합산하면 다음과 같아야 합니다.

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

이는 :numref:`sec_integral_calculus`에서 논의된 적분의 근사에 불과하므로, 저희는 다음과 같이 말할 수 있습니다.

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

확률 변수는 *어떤* 숫자를 취해야 하므로 $P(X\in\mathbb{R}) = 1$임을 알고 있고, 어떤 밀도에 대해서도 다음과 같이 결론 내릴 수 있습니다.

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

사실, 이를 더 파헤치면 어떤 $a$와 $b$에 대해서도 다음을 볼 수 있습니다.

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

저희는 이전과 같이 같은 이산 근사 방법을 사용하여 코드에서 이를 근사할 수 있습니다. 이 경우 저희는 파란색 영역에 떨어질 확률을 근사할 수 있습니다.

```{.python .input}
#@tab mxnet
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

이 두 속성이 가능한 확률 밀도 함수(또는 일반적으로 마주치는 약자로 *p.d.f.*)의 공간을 정확히 설명한다는 것이 밝혀집니다. 그것들은 다음과 같은 음이 아닌 함수 $p(x) \ge 0$입니다.

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

저희는 적분을 사용하여 저희의 확률 변수가 특정 구간에 있을 확률을 얻음으로써 이 함수를 해석합니다.

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

:numref:`sec_distributions`에서 저희는 많은 일반적인 분포를 볼 것이지만, 추상에서 계속 작업해 봅시다.

### 누적 분포 함수

이전 절에서, 저희는 p.d.f.의 개념을 보았습니다. 실제로, 이는 연속 확률 변수를 논의하는 일반적으로 마주치는 방법이지만, 한 가지 중요한 함정이 있습니다. p.d.f.의 값 자체가 확률이 아니라, 확률을 산출하기 위해 적분해야 하는 함수라는 것입니다. 밀도가 길이 $1/10$의 구간보다 더 많이 $10$보다 크지 않는 한, 밀도가 $10$보다 큰 것에는 잘못된 것이 없습니다. 이는 반직관적일 수 있으므로, 사람들은 종종 *누적 분포 함수* 또는 c.d.f.의 관점에서도 생각하는데, 이는 확률*입니다*.

특히, :eqref:`eq_pdf_int_int`를 사용하여, 저희는 밀도 $p(x)$를 가진 확률 변수 $X$에 대한 c.d.f.를 다음과 같이 정의합니다.

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

몇 가지 속성을 관찰해 봅시다.

* $x\rightarrow -\infty$일 때 $F(x) \rightarrow 0$.
* $x\rightarrow \infty$일 때 $F(x) \rightarrow 1$.
* $F(x)$는 비감소($y > x \implies F(y) \ge F(x)$)입니다.
* $X$가 연속 확률 변수라면 $F(x)$는 연속(점프가 없음)입니다.

네 번째 글머리표와 함께, 만약 $X$가 이산이라면, 예를 들어 모두 확률 $1/2$로 값 $0$과 $1$을 취한다면 이것이 참이 아닐 것이라는 점에 유의하십시오. 그 경우

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

이 예제에서, 저희는 c.d.f.로 작업하는 이점 중 하나, 즉 같은 프레임워크에서 연속 또는 이산 확률 변수를 다룰 수 있는 능력, 또는 실제로 둘의 혼합(동전 던지기: 앞면이면 주사위 굴림을 반환, 뒷면이면 다트판 중심에서 다트 던지기 거리를 반환)을 봅니다.

### 평균

확률 변수 $X$를 다루고 있다고 가정해 보십시오. 분포 자체는 해석하기 어려울 수 있습니다. 종종 확률 변수의 동작을 간결하게 요약할 수 있는 것이 유용합니다. 확률 변수의 동작을 포착하는 데 도움이 되는 숫자를 *요약 통계*라고 합니다. 가장 일반적으로 마주치는 것들은 *평균*, *분산*, 그리고 *표준 편차*입니다.

*평균*은 확률 변수의 평균값을 인코딩합니다. 만약 확률 $p_i$로 값 $x_i$를 취하는 이산 확률 변수 $X$가 있다면, 평균은 가중 평균에 의해 주어집니다. 값에 그 값을 취할 확률 변수의 확률을 곱한 것을 합산합니다.

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

평균을 해석해야 하는 방법은(주의가 필요하지만) 본질적으로 확률 변수가 어디에 위치하는 경향이 있는지 알려준다는 것입니다.

이 절 전반에 걸쳐 검토할 최소한의 예로, $X$를 확률 $p$로 값 $a-2$, 확률 $p$로 $a+2$, 그리고 확률 $1-2p$로 $a$를 취하는 확률 변수라고 합시다. $a$와 $p$의 어떤 가능한 선택에 대해서도, :eqref:`eq_exp_def`를 사용하여 평균이 다음과 같음을 계산할 수 있습니다.

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

따라서 평균이 $a$임을 봅니다. 이는 $a$가 저희가 확률 변수를 중심에 둔 위치이기 때문에 직관과 일치합니다.

도움이 되므로, 몇 가지 속성을 요약합시다.

* 어떤 확률 변수 $X$와 숫자 $a$와 $b$에 대해, 저희는 $\mu_{aX+b} = a\mu_X + b$를 가집니다.
* 만약 두 확률 변수 $X$와 $Y$가 있다면, 저희는 $\mu_{X+Y} = \mu_X+\mu_Y$를 가집니다.

평균은 확률 변수의 평균 동작을 이해하는 데 유용하지만, 평균만으로는 완전한 직관적 이해를 갖기에 충분하지 않습니다. 판매당 $\$10 \pm \$1$의 이익을 내는 것은 같은 평균값을 가지지만 판매당 $\$10 \pm \$15$를 내는 것과 매우 다릅니다. 두 번째 것은 변동의 정도가 훨씬 더 크며, 따라서 훨씬 더 큰 위험을 나타냅니다. 따라서, 확률 변수의 동작을 이해하기 위해, 저희는 최소한 하나의 추가 측정값, 즉 확률 변수가 얼마나 광범위하게 변동하는지에 대한 어떤 측정값이 필요할 것입니다.

### 분산

이는 저희를 확률 변수의 *분산*을 고려하도록 이끕니다. 이는 확률 변수가 평균에서 얼마나 멀리 벗어나는지에 대한 정량적 측정값입니다. 식 $X - \mu_X$를 고려해 보십시오. 이는 확률 변수의 평균에서의 편차입니다. 이 값은 양수 또는 음수일 수 있으므로, 편차의 크기를 측정하기 위해 양수로 만들기 위해 무언가를 해야 합니다.

시도할 합리적인 것은 $\left|X-\mu_X\right|$를 보는 것이며, 실제로 이는 *평균 절대 편차*라고 불리는 유용한 양으로 이어집니다. 그러나 수학과 통계의 다른 영역과의 연결로 인해, 사람들은 종종 다른 해결책을 사용합니다.

특히, 그들은 $(X-\mu_X)^2$을 봅니다. 평균을 취함으로써 이 양의 일반적인 크기를 보면, 저희는 분산에 도달합니다.

$$\sigma_X^2 = \textrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

:eqref:`eq_var_def`의 마지막 등식은 중간의 정의를 확장하고 기댓값의 속성을 적용함으로써 성립합니다.

$X$가 확률 $p$로 값 $a-2$, 확률 $p$로 $a+2$, 그리고 확률 $1-2p$로 $a$를 취하는 확률 변수인 저희의 예제를 봅시다. 이 경우 $\mu_X = a$이므로, 계산해야 할 모든 것은 $E\left[X^2\right]$입니다. 이는 쉽게 수행될 수 있습니다.

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

따라서, 저희는 :eqref:`eq_var_def`에 의해 저희의 분산이 다음과 같음을 봅니다.

$$
\sigma_X^2 = \textrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

이 결과도 의미가 있습니다. $p$가 될 수 있는 가장 큰 것은 $1/2$이며, 이는 동전 던지기로 $a-2$ 또는 $a+2$를 선택하는 것에 해당합니다. 이것이 $4$인 분산은 $a-2$와 $a+2$ 둘 다 평균에서 $2$ 단위 떨어져 있고 $2^2 = 4$라는 사실에 해당합니다. 스펙트럼의 다른 쪽 끝에서, 만약 $p=0$이라면, 이 확률 변수는 항상 값 $0$을 취하므로 전혀 분산이 없습니다.

분산의 몇 가지 속성을 아래에 나열하겠습니다.

* 어떤 확률 변수 $X$에 대해, $\textrm{Var}(X) \ge 0$이고, $X$가 상수인 경우에 그리고 오직 그 경우에만 $\textrm{Var}(X) = 0$.
* 어떤 확률 변수 $X$와 숫자 $a$와 $b$에 대해, 저희는 $\textrm{Var}(aX+b) = a^2\textrm{Var}(X)$를 가집니다.
* 만약 두 *독립* 확률 변수 $X$와 $Y$가 있다면, 저희는 $\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y)$를 가집니다.

이러한 값을 해석할 때, 약간의 딸꾹질이 있을 수 있습니다. 특히, 이 계산을 통해 단위를 추적한다면 무슨 일이 일어나는지 상상해 봅시다. 저희가 웹페이지의 제품에 할당된 별점으로 작업하고 있다고 가정해 봅시다. 그러면 $a$, $a-2$, 그리고 $a+2$는 모두 별 단위로 측정됩니다. 유사하게, 평균 $\mu_X$도 별 단위로 측정됩니다(가중 평균이므로). 그러나, 분산에 도달하면, 즉시 문제에 직면하는데, 그것은 저희가 *제곱된 별* 단위인 $(X-\mu_X)^2$를 보고 싶다는 것입니다. 이는 분산 자체가 원래 측정값과 비교할 수 없음을 의미합니다. 그것을 해석 가능하게 만들기 위해서는, 저희는 원래 단위로 돌아가야 할 것입니다.

### 표준 편차

이 요약 통계는 제곱근을 취함으로써 항상 분산에서 추론될 수 있습니다! 따라서 저희는 *표준 편차*를 다음과 같이 정의합니다.

$$
\sigma_X = \sqrt{\textrm{Var}(X)}.
$$

저희의 예제에서, 이는 이제 표준 편차가 $\sigma_X = 2\sqrt{2p}$임을 의미합니다. 만약 저희가 리뷰 예제에 대해 별 단위로 다루고 있다면, $\sigma_X$도 별 단위입니다.

저희가 분산에 대해 가졌던 속성은 표준 편차에 대해 다시 진술될 수 있습니다.

* 어떤 확률 변수 $X$에 대해, $\sigma_{X} \ge 0$.
* 어떤 확률 변수 $X$와 숫자 $a$와 $b$에 대해, 저희는 $\sigma_{aX+b} = |a|\sigma_{X}$를 가집니다.
* 만약 두 *독립* 확률 변수 $X$와 $Y$가 있다면, 저희는 $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$를 가집니다.

이 시점에서 "만약 표준 편차가 우리 원래 확률 변수의 단위라면, 그것이 그 확률 변수와 관련해서 우리가 그릴 수 있는 무언가를 나타냅니까?"라고 묻는 것은 자연스럽습니다. 답은 단호한 그렇다입니다! 사실 평균이 저희의 확률 변수의 일반적인 위치를 알려준 것처럼, 표준 편차는 그 확률 변수의 일반적인 변동 범위를 제공합니다. 저희는 체비셰프의 부등식으로 알려진 것으로 이를 엄격하게 만들 수 있습니다.

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

또는 $\alpha=10$의 경우 구두로 말하자면, 어떤 확률 변수의 샘플 중 $99\%$가 평균의 $10$ 표준 편차 내에 떨어집니다. 이는 저희의 표준 요약 통계에 즉각적인 해석을 제공합니다.

이 진술이 얼마나 미묘한지 보기 위해, $X$가 확률 $p$로 값 $a-2$, 확률 $p$로 $a+2$, 그리고 확률 $1-2p$로 $a$를 취하는 확률 변수인 저희의 진행 중인 예제를 다시 살펴봅시다. 저희는 평균이 $a$이고 표준 편차가 $2\sqrt{2p}$임을 보았습니다. 이는 $\alpha = 2$로 체비셰프의 부등식 :eqref:`eq_chebyshev`를 취하면, 식이 다음과 같음을 봄을 의미합니다.

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

이는 $p$의 어떤 값에 대해서도 시간의 $75\%$, 이 확률 변수가 이 구간 내에 떨어질 것임을 의미합니다. 이제, $p \rightarrow 0$일 때, 이 구간도 단일 점 $a$로 수렴한다는 점에 유의하십시오. 그러나 저희는 저희의 확률 변수가 $a-2, a$, 그리고 $a+2$만의 값을 취한다는 것을 알고 있으므로, 결국 $a-2$와 $a+2$가 구간 밖에 떨어질 것이라고 확신할 수 있습니다! 질문은 어떤 $p$에서 그런 일이 일어나는가입니다. 그래서 저희는 풀고 싶습니다. 어떤 $p$에 대해 $a+4\sqrt{2p} = a+2$인가, 이는 $p=1/8$일 때 풀리며, 이는 분포의 샘플의 $1/4$ 이상이 구간 밖에 떨어지지 않을 것이라는 저희의 주장을 위반하지 않고 가능하게 일어날 수 있는 *정확히* 첫 번째 $p$입니다(왼쪽으로 $1/8$, 오른쪽으로 $1/8$).

이를 시각화해 봅시다. 저희는 세 값을 얻을 확률을 확률에 비례하는 높이의 세 개의 수직 막대로 표시할 것입니다. 구간은 중간에 수평선으로 그려질 것입니다. 첫 번째 플롯은 $p > 1/8$일 때 구간이 모든 점을 안전하게 포함하는 것을 보여줍니다.

```{.python .input}
#@tab mxnet
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

두 번째는 $p = 1/8$에서 구간이 정확히 두 점을 만진다는 것을 보여줍니다. 이는 부등식이 *날카로운* 것임을 보여주는데, 왜냐하면 부등식이 참인 채로 유지하면서 더 작은 구간을 취할 수 없기 때문입니다.

```{.python .input}
#@tab mxnet
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

세 번째는 $p < 1/8$의 경우 구간이 중심만 포함한다는 것을 보여줍니다. 이는 부등식을 무효화하지 않습니다. 저희는 단지 확률의 $1/4$ 이하가 구간 밖에 떨어지도록 보장해야 했기 때문이며, 이는 일단 $p < 1/8$이면, $a-2$와 $a+2$의 두 점이 버려질 수 있음을 의미합니다.

```{.python .input}
#@tab mxnet
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### 연속에서의 평균과 분산

이 모든 것은 이산 확률 변수의 관점에서였지만, 연속 확률 변수의 경우도 유사합니다. 이것이 어떻게 작동하는지 직관적으로 이해하기 위해, 저희가 실수 직선을 $(\epsilon i, \epsilon (i+1)]$로 주어진 길이 $\epsilon$의 구간으로 나눈다고 상상해 보십시오. 일단 이렇게 하면, 저희의 연속 확률 변수가 이산화되었고 :eqref:`eq_exp_def`를 사용하여 다음과 같이 말할 수 있습니다.

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

여기서 $p_X$는 $X$의 밀도입니다. 이는 $xp_X(x)$의 적분에 대한 근사이므로, 저희는 다음과 같이 결론 내릴 수 있습니다.

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

유사하게, :eqref:`eq_var_def`를 사용하여 분산은 다음과 같이 작성될 수 있습니다.

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

평균, 분산, 그리고 표준 편차에 대해 위에서 진술된 모든 것은 이 경우에도 여전히 적용됩니다. 예를 들어, 만약 다음 밀도를 가진 확률 변수를 고려한다면

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \textrm{otherwise}.
\end{cases}
$$

저희는 다음을 계산할 수 있습니다.

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

그리고

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

경고로, *코시 분포*로 알려진 한 가지 더 예를 검토해 봅시다. 이는 다음과 같이 주어진 p.d.f.를 가진 분포입니다.

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
#@tab mxnet
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

이 함수는 무해해 보이며, 실제로 적분 테이블을 참조하면 그 아래에 면적 1을 가진다는 것을 보여주므로, 연속 확률 변수를 정의합니다.

무엇이 잘못되는지 보기 위해, 이것의 분산을 계산해 봅시다. 이는 :eqref:`eq_var_def`를 사용하여 다음을 계산하는 것을 포함할 것입니다.

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

내부의 함수는 다음과 같이 보입니다.

```{.python .input}
#@tab mxnet
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

이 함수는 본질적으로 0 근처에 작은 함몰이 있는 상수 1이므로 그 아래에 분명히 무한한 면적을 가지며, 실제로 저희는 다음을 보일 수 있습니다.

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

이는 그것이 잘 정의된 유한 분산을 가지지 않음을 의미합니다.

그러나, 더 깊이 보면 훨씬 더 불안한 결과를 보여줍니다. :eqref:`eq_exp_def`를 사용하여 평균을 계산해 봅시다. 변수 변환 공식을 사용하면, 저희는 다음을 봅니다.

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

내부의 적분은 로그의 정의이므로, 이는 본질적으로 $\log(\infty) = \infty$이므로, 잘 정의된 평균값도 없습니다!

머신러닝 과학자들은 그들의 모델을 정의하므로 저희가 가장 자주 이러한 문제를 다룰 필요가 없고, 대다수의 경우 잘 정의된 평균과 분산을 가진 확률 변수를 다룰 것입니다. 그러나, 가끔씩 *두꺼운 꼬리*를 가진 확률 변수(즉, 큰 값을 얻을 확률이 평균이나 분산과 같은 것을 정의되지 않게 만들 만큼 충분히 큰 확률 변수)가 물리적 시스템을 모델링하는 데 도움이 되므로, 그것들이 존재한다는 것을 아는 것은 가치가 있습니다.

### 결합 밀도 함수

위 작업 모두는 저희가 단일 실숫값 확률 변수로 작업하고 있다고 가정합니다. 그러나 만약 잠재적으로 매우 상관된 두 개 이상의 확률 변수를 다루고 있다면 어떨까요? 이 상황은 머신러닝의 표준입니다. 이미지에서 $(i, j)$ 좌표의 픽셀의 빨간색 값을 인코딩하는 $R_{i, j}$ 또는 시간 $t$에 주가에 의해 주어진 확률 변수인 $P_t$와 같은 확률 변수를 상상해 보십시오. 가까운 픽셀은 비슷한 색을 가지는 경향이 있고, 가까운 시간은 비슷한 가격을 가지는 경향이 있습니다. 저희는 그것들을 별도의 확률 변수로 취급할 수 없고 성공적인 모델을 만들 것이라고 기대할 수 없습니다(:numref:`sec_naive_bayes`에서 그러한 가정으로 인해 성능이 떨어지는 모델을 볼 것입니다). 저희는 이러한 상관된 연속 확률 변수를 다루기 위한 수학적 언어를 발전시켜야 합니다.

다행스럽게도, :numref:`sec_integral_calculus`의 다중 적분으로 저희는 그러한 언어를 발전시킬 수 있습니다. 단순함을 위해, 상관될 수 있는 두 확률 변수 $X, Y$가 있다고 가정해 보십시오. 그러면, 단일 변수의 경우와 유사하게, 저희는 질문을 할 수 있습니다.

$$
P(X \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x \; \textrm{and} \;Y \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; y ).
$$

단일 변수 경우와 유사한 추론은 이것이 대략 다음과 같아야 한다는 것을 보여줍니다.

$$
P(X \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x \; \textrm{and} \;Y \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

어떤 함수 $p(x, y)$에 대해서 말입니다. 이는 $X$와 $Y$의 결합 밀도라고 합니다. 저희가 단일 변수 경우에 본 것처럼 이에 대해서도 유사한 속성이 참입니다. 즉:

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

이런 식으로, 저희는 여러 개의 잠재적으로 상관된 확률 변수를 다룰 수 있습니다. 만약 두 개 이상의 확률 변수로 작업하고 싶다면, 저희는 $p(\mathbf{x}) = p(x_1, \ldots, x_n)$를 고려함으로써 다변량 밀도를 원하는 만큼의 좌표로 확장할 수 있습니다. 음이 아니고 총 적분이 1인 같은 속성이 여전히 성립합니다.

### 주변 분포
여러 변수를 다룰 때, 저희는 종종 관계를 무시하고 "이 한 변수는 어떻게 분포되어 있는가?"라고 묻고 싶습니다. 그러한 분포는 *주변 분포*라고 합니다.

구체적으로 말하면, $p _ {X, Y}(x, y)$로 주어진 결합 밀도를 가진 두 확률 변수 $X, Y$가 있다고 가정해 봅시다. 저희는 밀도가 어떤 확률 변수에 대한 것인지를 나타내기 위해 첨자를 사용할 것입니다. 주변 분포를 찾는 질문은 이 함수를 가져다가 $p _ X(x)$를 찾는 데 사용하는 것입니다.

대부분의 것과 마찬가지로, 무엇이 참이어야 하는지 알아내기 위해 직관적인 그림으로 돌아가는 것이 가장 좋습니다. 밀도가 다음과 같은 함수 $p _ X$라는 것을 떠올리십시오.

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

$Y$에 대한 언급이 없지만, 만약 저희에게 주어진 모든 것이 $p _{X, Y}$라면, 저희는 어떻게든 $Y$를 포함해야 합니다. 저희는 먼저 이것이 다음과 같다는 것을 관찰할 수 있습니다.

$$
P(X \in [x, x+\epsilon] \textrm{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

저희의 밀도는 이 경우 무슨 일이 일어나는지 직접 알려주지 않으므로, 저희는 또한 $y$에서 작은 구간으로 분할해야 합니다. 그래서 저희는 이를 다음과 같이 쓸 수 있습니다.

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \textrm{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![저희의 확률 배열의 열을 따라 합산함으로써, 저희는 $\mathit{x}$축을 따라 표현된 확률 변수만에 대한 주변 분포를 얻을 수 있습니다.](../img/marginal.svg)
:label:`fig_marginal`

이는 :numref:`fig_marginal`에서 보여지는 것처럼 일렬로 있는 일련의 정사각형을 따라 밀도의 값을 더하라고 알려줍니다. 사실, 양변에서 엡실론의 한 인자를 소거하고, 오른쪽의 합이 $y$에 대한 적분임을 인식한 후, 저희는 다음과 같이 결론 내릴 수 있습니다.

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

따라서 저희는 다음을 봅니다.

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

이는 저희에게 주변 분포를 얻기 위해서는 신경 쓰지 않는 변수에 대해 적분한다고 알려줍니다. 이 과정은 종종 불필요한 변수를 *적분으로 제거하기* 또는 *주변화하기*라고 합니다.

### 공분산

여러 확률 변수를 다룰 때, 알아두면 유용한 추가 요약 통계가 하나 있습니다. *공분산*입니다. 이는 두 확률 변수가 함께 변동하는 정도를 측정합니다.

두 확률 변수 $X$와 $Y$가 있다고 가정하고, 시작하기 위해 그들이 이산이고 확률 $p_{ij}$로 값 $(x_i, y_j)$를 취한다고 가정해 봅시다. 이 경우, 공분산은 다음과 같이 정의됩니다.

$$\sigma_{XY} = \textrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

이에 대해 직관적으로 생각하기 위해, 다음 쌍의 확률 변수를 고려해 보십시오. $X$가 값 $1$과 $3$을 취하고, $Y$가 값 $-1$과 $3$을 취한다고 가정해 봅시다. 다음 확률을 가진다고 가정해 봅시다.

$$
\begin{aligned}
P(X = 1 \; \textrm{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \textrm{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

여기서 $p$는 저희가 선택할 수 있는 $[0,1]$의 매개변수입니다. 만약 $p=1$이면 그들은 둘 다 항상 동시에 최솟값 또는 최댓값이고, 만약 $p=0$이면 그들은 보장된 채로 동시에 뒤집힌 값을 취하는 것임에 유의하십시오(하나는 다른 것이 작을 때 크고 그 반대도). 만약 $p=1/2$이면, 네 가지 가능성이 모두 동등하게 가능성이 있고, 어느 쪽도 관련되지 않아야 합니다. 공분산을 계산해 봅시다. 먼저, $\mu_X = 2$이고 $\mu_Y = 1$임에 유의하십시오. 따라서 저희는 :eqref:`eq_cov_def`를 사용하여 계산할 수 있습니다.

$$
\begin{aligned}
\textrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

$p=1$일 때(둘 다 동시에 최대로 양수 또는 음수인 경우) 공분산은 $2$입니다. $p=0$일 때(그들이 뒤집힌 경우) 공분산은 $-2$입니다. 마지막으로, $p=1/2$일 때(그들이 관련되지 않은 경우), 공분산은 $0$입니다. 따라서 저희는 공분산이 이 두 확률 변수가 어떻게 관련되어 있는지를 측정한다는 것을 봅니다.

공분산에 대한 빠른 메모는 그것이 이러한 선형 관계만 측정한다는 것입니다. $Y$가 동등한 확률로 $\{-2, -1, 0, 1, 2\}$에서 무작위로 선택될 때 $X = Y^2$와 같은 더 복잡한 관계는 놓칠 수 있습니다. 사실 빠른 계산은 이러한 확률 변수가 하나가 다른 것의 결정적 함수임에도 불구하고 공분산이 0임을 보여줍니다.

연속 확률 변수의 경우, 거의 같은 이야기가 성립합니다. 이 시점에서, 저희는 이산과 연속 사이의 전환을 하는 데 꽤 편안하므로, 어떤 유도 없이 :eqref:`eq_cov_def`의 연속 유사물을 제공할 것입니다.

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

시각화를 위해, 조정 가능한 공분산을 가진 확률 변수의 모음을 살펴봅시다.

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

공분산의 몇 가지 속성을 봅시다.

* 어떤 확률 변수 $X$에 대해, $\textrm{Cov}(X, X) = \textrm{Var}(X)$.
* 어떤 확률 변수 $X, Y$와 숫자 $a$와 $b$에 대해, $\textrm{Cov}(aX+b, Y) = \textrm{Cov}(X, aY+b) = a\textrm{Cov}(X, Y)$.
* 만약 $X$와 $Y$가 독립이라면 $\textrm{Cov}(X, Y) = 0$.

추가로, 저희는 공분산을 사용하여 이전에 본 관계를 확장할 수 있습니다. $X$와 $Y$가 두 독립 확률 변수라면 다음을 떠올리십시오.

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y).
$$

공분산의 지식으로, 저희는 이 관계를 확장할 수 있습니다. 사실, 약간의 대수는 일반적으로 다음을 보일 수 있습니다.

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\textrm{Cov}(X, Y).
$$

이는 상관된 확률 변수에 대한 분산 합산 규칙을 일반화할 수 있게 해줍니다.

### 상관

저희가 평균과 분산의 경우 했던 것처럼, 이제 단위를 고려해 봅시다. 만약 $X$가 한 단위(예: 인치)로 측정되고 $Y$가 다른 단위(예: 달러)로 측정된다면, 공분산은 이 두 단위의 곱 $\textrm{inches} \times \textrm{dollars}$로 측정됩니다. 이러한 단위는 해석하기 어려울 수 있습니다. 이 경우 저희가 종종 원하는 것은 관련성의 단위 없는 측정입니다. 사실, 종종 저희는 정확한 정량적 상관에 대해 신경 쓰지 않고, 오히려 상관이 같은 방향인지, 그리고 관계가 얼마나 강한지 묻습니다.

무엇이 의미가 있는지 보기 위해, 사고 실험을 수행해 봅시다. 인치와 달러로 된 저희의 확률 변수를 인치와 센트로 변환한다고 가정해 봅시다. 이 경우 확률 변수 $Y$는 $100$이 곱해집니다. 만약 정의를 통해 작업하면, 이는 $\textrm{Cov}(X, Y)$가 $100$이 곱해질 것임을 의미합니다. 따라서 저희는 이 경우 단위의 변경이 공분산을 $100$의 인자로 변경한다는 것을 봅니다. 따라서, 상관의 단위 불변 측정값을 찾기 위해서는, 저희는 $100$으로도 스케일링되는 다른 것으로 나눠야 할 것입니다. 사실 저희는 명확한 후보, 즉 표준 편차를 가지고 있습니다! 사실 저희가 *상관 계수*를 다음과 같이 정의한다면

$$\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

이것이 단위 없는 값임을 봅니다. 약간의 수학은 이 숫자가 $-1$과 $1$ 사이임을 보일 수 있는데, $1$은 최대로 양의 상관, 반면 $-1$은 최대로 음의 상관을 의미합니다.

위의 저희의 명시적인 이산 예제로 돌아가서, $\sigma_X = 1$이고 $\sigma_Y = 2$임을 볼 수 있으므로, :eqref:`eq_cor_def`를 사용하여 두 확률 변수 사이의 상관을 계산하여 다음을 볼 수 있습니다.

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

이는 이제 가장 상관됨을 의미하는 $1$과 최소로 상관됨을 의미하는 $-1$의 예상되는 동작과 함께 $-1$과 $1$ 사이의 범위입니다.

또 다른 예로, $X$를 어떤 확률 변수로, $Y=aX+b$를 $X$의 어떤 선형 결정적 함수로 고려해 보십시오. 그러면, 다음을 계산할 수 있습니다.

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\textrm{Cov}(X, Y) = \textrm{Cov}(X, aX+b) = a\textrm{Cov}(X, X) = a\textrm{Var}(X),$$

따라서 :eqref:`eq_cor_def`에 의해 다음을 얻습니다.

$$
\rho(X, Y) = \frac{a\textrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \textrm{sign}(a).
$$

따라서 저희는 상관이 어떤 $a > 0$에 대해서도 $+1$이고, 어떤 $a < 0$에 대해서도 $-1$임을 보는데, 이는 상관이 변동이 취하는 스케일이 아니라 두 확률 변수가 관련되어 있는 정도와 방향성을 측정함을 보여줍니다.

조정 가능한 상관을 가진 확률 변수의 모음을 다시 플롯해 봅시다.

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

상관의 몇 가지 속성을 아래에 나열해 봅시다.

* 어떤 확률 변수 $X$에 대해, $\rho(X, X) = 1$.
* 어떤 확률 변수 $X, Y$와 숫자 $a$와 $b$에 대해, $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$.
* 만약 $X$와 $Y$가 0이 아닌 분산을 가지고 독립이라면 $\rho(X, Y) = 0$.

마지막 메모로, 이러한 공식들 중 일부가 친숙하게 느껴질 수 있습니다. 사실, $\mu_X = \mu_Y = 0$이라고 가정하고 모든 것을 확장하면, 저희는 이것이 다음과 같음을 봅니다.

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

이는 항의 곱의 합을 항의 합의 제곱근으로 나눈 것처럼 보입니다. 이는 다른 좌표가 $p_{ij}$에 의해 가중된 두 벡터 $\mathbf{v}, \mathbf{w}$ 사이의 각도의 코사인에 대한 정확한 공식입니다.

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

사실 만약 저희가 노름을 표준 편차와 관련된 것으로, 그리고 상관을 각도의 코사인으로 생각한다면, 기하학에서 저희가 가진 직관의 많은 부분이 확률 변수에 대해 생각하는 데 적용될 수 있습니다.

## 요약
* 연속 확률 변수는 값의 연속체를 취할 수 있는 확률 변수입니다. 그들은 이산 확률 변수와 비교하여 작업하기 더 도전적으로 만드는 몇 가지 기술적 어려움을 가지고 있습니다.
* 확률 밀도 함수는 곡선 아래의 면적이 어떤 구간에서 그 구간에서 샘플 점을 찾을 확률을 제공하는 함수를 제공함으로써 연속 확률 변수로 작업할 수 있게 해줍니다.
* 누적 분포 함수는 확률 변수가 주어진 임계값보다 작음을 관찰할 확률입니다. 이는 이산과 연속 변수를 통합하는 유용한 대안 관점을 제공할 수 있습니다.
* 평균은 확률 변수의 평균값입니다.
* 분산은 확률 변수와 그 평균의 차이의 기대 제곱입니다.
* 표준 편차는 분산의 제곱근입니다. 이는 확률 변수가 취할 수 있는 값의 범위를 측정하는 것으로 생각될 수 있습니다.
* 체비셰프의 부등식은 대부분의 시간 동안 확률 변수를 포함하는 명시적 구간을 제공함으로써 이 직관을 엄격하게 만들 수 있게 해줍니다.
* 결합 밀도는 상관된 확률 변수로 작업할 수 있게 해줍니다. 저희는 원하는 확률 변수의 분포를 얻기 위해 원치 않는 확률 변수에 대해 적분함으로써 결합 밀도를 주변화할 수 있습니다.
* 공분산과 상관 계수는 두 상관된 확률 변수 사이의 어떤 선형 관계를 측정하는 방법을 제공합니다.

## 연습문제
1. $x \ge 1$에 대해 $p(x) = \frac{1}{x^2}$로 주어진 밀도와 그렇지 않으면 $p(x) = 0$을 가진 확률 변수가 있다고 가정해 보십시오. $P(X > 2)$는 무엇입니까?
2. 라플라스 분포는 밀도가 $p(x = \frac{1}{2}e^{-|x|}$로 주어진 확률 변수입니다. 이 함수의 평균과 표준 편차는 무엇입니까? 힌트로, $\int_0^\infty xe^{-x} \; dx = 1$이고 $\int_0^\infty x^2e^{-x} \; dx = 2$.
3. 저는 거리에서 당신에게 다가가 "저는 평균 $1$, 표준 편차 $2$인 확률 변수가 있고, 저는 제 샘플의 $25\%$가 $9$보다 큰 값을 취하는 것을 관찰했습니다."라고 말합니다. 저를 믿습니까? 왜 또는 왜 아닙니까?
4. $x, y \in [0,1]$에 대해 $p_{XY}(x, y) = 4xy$로 주어진 결합 밀도와 그렇지 않으면 $p_{XY}(x, y) = 0$을 가진 두 확률 변수 $X, Y$가 있다고 가정해 보십시오. $X$와 $Y$의 공분산은 무엇입니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1094)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1095)
:end_tab:
