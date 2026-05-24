# 통계
:label:`sec_statistics`

의심할 여지 없이, 최고 수준의 딥러닝 실무자가 되기 위해서는, 최첨단의 고정확도 모델을 훈련시키는 능력이 중요합니다. 그러나, 개선이 유의미한 것인지, 아니면 단지 훈련 과정의 무작위 변동의 결과인지가 종종 불분명합니다. 추정된 값의 불확실성을 논의할 수 있으려면, 저희는 약간의 통계를 배워야 합니다.


*통계*에 대한 가장 초기의 언급은 $9^{\textrm{th}}$ 세기의 아랍 학자 알 킨디에게 거슬러 올라갈 수 있는데, 그는 통계와 빈도 분석을 사용하여 암호화된 메시지를 해독하는 방법에 대한 자세한 설명을 제공했습니다. 800년 후, 현대 통계는 1700년대 독일에서 일어났는데, 연구자들이 인구통계학적 및 경제 데이터 수집과 분석에 초점을 맞췄을 때였습니다. 오늘날, 통계는 데이터의 수집, 처리, 분석, 해석, 그리고 시각화와 관련된 과학 과목입니다. 더욱이, 통계의 핵심 이론은 학계, 산업계, 그리고 정부 내의 연구에서 널리 사용되어 왔습니다.


더 구체적으로, 통계는 *기술 통계*와 *통계적 추론*으로 나눌 수 있습니다. 전자는 *샘플*이라고 불리는, 관찰된 데이터 모음의 특징을 요약하고 설명하는 데 초점을 맞춥니다. 샘플은 *모집단*에서 추출되며, 이는 저희의 실험적 관심의 유사한 개체, 항목 또는 사건의 전체 집합을 나타냅니다. 기술 통계와 반대로, *통계적 추론*은 샘플 분포가 어느 정도 모집단 분포를 재현할 수 있다는 가정에 기반하여, 주어진 *샘플*에서 모집단의 특성을 추가로 추론합니다.


다음과 같이 궁금해 할 수 있습니다. "머신러닝과 통계 사이의 본질적인 차이는 무엇인가?" 근본적으로 말하면, 통계는 추론 문제에 초점을 맞춥니다. 이 유형의 문제는 인과 추론과 같이 변수 사이의 관계를 모델링하는 것과, A/B 테스트와 같이 모델 매개변수의 통계적 유의성을 테스트하는 것을 포함합니다. 대조적으로, 머신러닝은 각 매개변수의 기능을 명시적으로 프로그래밍하고 이해하지 않고도 정확한 예측을 하는 것을 강조합니다.


이 절에서, 저희는 세 가지 유형의 통계 추론 방법, 즉 추정량을 평가하고 비교하는 것, 가설 검정을 수행하는 것, 그리고 신뢰 구간을 구성하는 것을 소개할 것입니다. 이러한 방법은 저희가 주어진 모집단의 특성, 즉 참 매개변수 $\theta$를 추론하는 데 도움이 될 수 있습니다. 간결함을 위해, 저희는 주어진 모집단의 참 매개변수 $\theta$가 스칼라 값이라고 가정합니다. $\theta$가 벡터 또는 텐서인 경우로 확장하는 것은 간단하므로, 저희는 그것을 논의에서 생략합니다.



## 추정량 평가 및 비교

통계에서, *추정량*은 참 매개변수 $\theta$를 추정하는 데 사용되는 주어진 샘플의 함수입니다. 저희는 샘플 {$x_1, x_2, \ldots, x_n$}을 관찰한 후 $\theta$의 추정에 대해 $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$이라고 쓸 것입니다.

저희는 이전에 :numref:`sec_maximum_likelihood` 절에서 추정량의 간단한 예제를 보았습니다. 만약 베르누이 확률 변수로부터의 여러 샘플이 있다면, 확률 변수가 1일 확률에 대한 최대 가능도 추정은 관찰된 1의 수를 세고 총 샘플 수로 나눔으로써 얻을 수 있습니다. 유사하게, 한 연습문제는 여러 샘플이 주어졌을 때 가우시안의 평균에 대한 최대 가능도 추정이 모든 샘플의 평균값에 의해 주어진다는 것을 보여달라고 요청했습니다. 이러한 추정량은 거의 매개변수의 참값을 결코 주지 않을 것이지만, 이상적으로는 많은 샘플에 대해 추정이 가까울 것입니다.

예로, 아래에서 저희는 평균 0과 분산 1을 가진 가우시안 확률 변수의 참 밀도를 그 가우시안으로부터의 샘플 모음과 함께 보여줍니다. 저희는 모든 점이 보이고 원래 밀도와의 관계가 더 명확하도록 $y$ 좌표를 구성했습니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

매개변수 $\hat{\theta}_n$의 추정량을 계산하는 많은 방법이 있을 수 있습니다. 이 절에서, 저희는 추정량을 평가하고 비교하는 세 가지 일반적인 방법, 즉 평균 제곱 오차, 표준 편차, 그리고 통계적 편향을 소개합니다.

### 평균 제곱 오차

추정량을 평가하는 데 사용되는 가장 단순한 척도는 아마도 *평균 제곱 오차(MSE)*(또는 $l_2$ 손실) 추정량일 것이며, 이는 다음과 같이 정의될 수 있습니다.

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

이는 참값에서의 평균 제곱 편차를 정량화할 수 있게 해줍니다. MSE는 항상 음이 아닙니다. 만약 :numref:`sec_linear_regression`을 읽었다면, 가장 일반적으로 사용되는 회귀 손실 함수로 그것을 인식할 것입니다. 추정량을 평가하는 척도로서, 그 값이 0에 가까울수록, 추정량은 참 매개변수 $\theta$에 더 가깝습니다.


### 통계적 편향

MSE는 자연스러운 척도를 제공하지만, 저희는 그것을 크게 만들 수 있는 여러 다른 현상을 쉽게 상상할 수 있습니다. 두 가지 근본적으로 중요한 것은 데이터셋의 무작위성으로 인한 추정량의 변동과 추정 절차로 인한 추정량의 체계적 오차입니다.


먼저, 체계적 오차를 측정해 봅시다. 추정량 $\hat{\theta}_n$의 경우, *통계적 편향*의 수학적 설명은 다음과 같이 정의될 수 있습니다.

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\textrm{bias}(\hat{\theta}_n) = 0$일 때, 추정량 $\hat{\theta}_n$의 기댓값이 매개변수의 참값과 같다는 점에 유의하십시오. 이 경우, 저희는 $\hat{\theta}_n$이 불편 추정량이라고 말합니다. 일반적으로, 불편 추정량은 그 기댓값이 참 매개변수와 같기 때문에 편향 추정량보다 낫습니다.


그러나, 편향 추정량이 실제로 자주 사용된다는 것을 알아둘 가치가 있습니다. 추가 가정 없이는 불편 추정량이 존재하지 않거나, 계산하기 어려운 경우가 있습니다. 이것은 추정량의 중대한 결함처럼 보일 수 있지만, 실제로 마주치는 대부분의 추정량은 사용 가능한 샘플 수가 무한대로 가는 경향이 있을 때 편향이 0으로 가는 경향이 있다는 의미에서 적어도 점근적으로 불편입니다. 즉, $\lim_{n \rightarrow \infty} \textrm{bias}(\hat{\theta}_n) = 0$.


### 분산과 표준 편차

둘째로, 추정량의 무작위성을 측정해 봅시다. :numref:`sec_random_variables`에서, *표준 편차*(또는 *표준 오차*)는 분산의 제곱근으로 정의됨을 떠올리십시오. 저희는 그 추정량의 표준 편차 또는 분산을 측정함으로써 추정량의 변동 정도를 측정할 수 있습니다.

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

:eqref:`eq_var_est`을 :eqref:`eq_mse_est`와 비교하는 것이 중요합니다. 이 방정식에서 저희는 참 모집단 값 $\theta$와 비교하는 것이 아니라, 기대 샘플 평균 $E(\hat{\theta}_n)$과 비교합니다. 따라서 저희는 추정량이 참값에서 얼마나 멀어지는 경향이 있는지 측정하는 것이 아니라, 추정량 자체의 변동을 측정합니다.


### 편향-분산 트레이드오프

이 두 가지 주요 구성 요소가 평균 제곱 오차에 기여한다는 것은 직관적으로 명확합니다. 약간 충격적인 것은 저희가 실제로 이것이 평균 제곱 오차의 이러한 두 가지 기여와 세 번째 기여로의 *분해*임을 보일 수 있다는 것입니다. 즉, 평균 제곱 오차를 편향의 제곱, 분산, 그리고 환원할 수 없는 오차의 합으로 쓸 수 있습니다.

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\\
\end{aligned}
$$

저희는 위의 공식을 *편향-분산 트레이드오프*라고 부릅니다. 평균 제곱 오차는 세 가지 오차의 출처, 즉 높은 편향으로부터의 오차, 높은 분산으로부터의 오차, 그리고 환원할 수 없는 오차로 나눌 수 있습니다. 편향 오차는 특징과 출력 사이의 고차원 관계를 추출할 수 없는 단순한 모델(예: 선형 회귀 모델)에서 일반적으로 보입니다. 만약 모델이 높은 편향 오차로 고통받는다면, 저희는 종종 그것이 (:numref:`sec_generalization_basics`)에서 소개된 *과소적합* 또는 *유연성* 부족이라고 말합니다. 높은 분산은 보통 너무 복잡한 모델에서 비롯되는데, 이는 훈련 데이터를 과적합합니다. 결과적으로, *과적합* 모델은 데이터의 작은 변동에 민감합니다. 만약 모델이 높은 분산으로 고통받는다면, 저희는 종종 그것이 (:numref:`sec_generalization_basics`)에서 소개된 *과적합*이고 *일반화* 부족이라고 말합니다. 환원할 수 없는 오차는 $\theta$ 자체의 잡음으로부터의 결과입니다.


### 코드에서 추정량 평가

추정량의 표준 편차는 텐서 `a`에 대해 단순히 `a.std()`를 호출함으로써 구현되어 왔으므로, 저희는 그것을 건너뛰지만 통계적 편향과 평균 제곱 오차를 구현할 것입니다.

```{.python .input}
#@tab mxnet
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

편향-분산 트레이드오프의 방정식을 설명하기 위해, $10,000$개의 샘플로 정규 분포 $\mathcal{N}(\theta, \sigma^2)$를 시뮬레이션해 봅시다. 여기서, 저희는 $\theta = 1$과 $\sigma = 4$를 사용합니다. 추정량이 주어진 샘플의 함수이므로, 여기서 저희는 이 정규 분포 $\mathcal{N}(\theta, \sigma^2)$에서 참 $\theta$에 대한 추정량으로 샘플의 평균을 사용합니다.

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

저희 추정량의 제곱 편향과 분산의 합산을 계산하여 트레이드오프 방정식을 검증해 봅시다. 먼저, 저희 추정량의 MSE를 계산합니다.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

다음으로, 아래와 같이 $\textrm{Var} (\hat{\theta}_n) + [\textrm{bias} (\hat{\theta}_n)]^2$를 계산합니다. 보시다시피, 두 값은 수치 정밀도에서 일치합니다.

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 가설 검정 수행


통계적 추론에서 가장 일반적으로 마주치는 주제는 가설 검정입니다. 가설 검정은 $20^{th}$ 세기 초에 대중화되었지만, 첫 번째 사용은 1700년대 존 아버스넛에게 거슬러 올라갈 수 있습니다. 존은 런던에서 80년 출생 기록을 추적했고 매년 여성보다 더 많은 남성이 태어났다고 결론지었습니다. 이어, 현대의 유의성 검정은 $p$-값과 피어슨의 카이제곱 검정을 발명한 칼 피어슨, 스튜던트 t-분포의 아버지인 윌리엄 고셋, 그리고 영가설과 유의성 검정을 시작한 로널드 피셔에 의한 지적 유산입니다.

*가설 검정*은 모집단에 대한 기본 진술에 대해 어떤 증거를 평가하는 방법입니다. 저희는 관찰된 데이터를 사용하여 기각하려고 시도하는 기본 진술을 *영가설* $H_0$이라고 부릅니다. 여기서, 저희는 $H_0$를 통계적 유의성 검정을 위한 시작점으로 사용합니다. *대립가설* $H_A$(또는 $H_1$)은 영가설에 반대되는 진술입니다. 영가설은 종종 변수 사이의 관계를 상정하는 선언적 형태로 진술됩니다. 그것은 가능한 한 명시적으로 개요를 반영해야 하며, 통계 이론에 의해 테스트 가능해야 합니다.

당신이 화학자라고 상상해 보십시오. 실험실에서 수천 시간을 보낸 후, 당신은 수학을 이해하는 능력을 극적으로 향상시킬 수 있는 새로운 약을 개발합니다. 그 마법의 힘을 보여주기 위해, 당신은 그것을 테스트해야 합니다. 자연스럽게, 약을 복용하고 그것이 수학을 더 잘 배우는 데 도움이 되는지 보기 위해 몇몇 자원봉사자가 필요할 수 있습니다. 어떻게 시작합니까?

먼저, 당신은 어떤 척도에 의해 측정된 수학적 이해 능력 사이에 차이가 없도록 무작위로 신중하게 선택된 두 그룹의 자원봉사자가 필요할 것입니다. 두 그룹은 일반적으로 테스트 그룹과 대조군이라고 합니다. *테스트 그룹*(또는 *처리 그룹*)은 약을 경험할 개체의 그룹인 반면, *대조군*은 벤치마크로 따로 설정된 사용자 그룹, 즉 이 약을 복용하는 것을 제외하고 동일한 환경 설정을 나타냅니다. 이런 식으로, 처리에서 독립 변수의 영향을 제외한 모든 변수의 영향이 최소화됩니다.

둘째로, 약을 복용하는 기간 후에, 당신은 동일한 척도, 예를 들어 새로운 수학 공식을 배운 후 자원봉사자들에게 동일한 테스트를 보게 함으로써 두 그룹의 수학적 이해를 측정해야 할 것입니다. 그런 다음, 그들의 성과를 수집하고 결과를 비교할 수 있습니다. 이 경우, 저희의 영가설은 두 그룹 사이에 차이가 없다는 것이 될 것이고, 저희의 대안은 차이가 있다는 것이 될 것입니다.

이것은 여전히 완전히 형식적이지 않습니다. 신중하게 생각해야 할 많은 세부사항이 있습니다. 예를 들어, 그들의 수학적 이해 능력을 테스트하기 위한 적합한 척도는 무엇입니까? 당신의 약의 효과를 자신 있게 주장할 수 있도록 얼마나 많은 자원봉사자가 테스트에 필요합니까? 테스트를 얼마나 오래 실행해야 합니까? 두 그룹 사이에 차이가 있는지 어떻게 결정합니까? 평균 성과에 대해서만 신경 씁니까, 아니면 점수의 변동 범위에 대해서도 신경 씁니까? 등등.

이런 식으로, 가설 검정은 실험 설계와 관찰된 결과의 확실성에 대한 추론을 위한 프레임워크를 제공합니다. 만약 이제 영가설이 참일 가능성이 매우 낮다는 것을 보일 수 있다면, 저희는 자신 있게 그것을 기각할 수 있습니다.

가설 검정으로 작업하는 방법에 대한 이야기를 완성하기 위해, 이제 추가 용어를 소개하고 위의 저희 개념 중 일부를 형식화해야 합니다.


### 통계적 유의성

*통계적 유의성*은 영가설 $H_0$이 기각되지 않아야 할 때 잘못 기각될 확률을 측정합니다. 즉,

$$ \textrm{statistical significance }= 1 - \alpha = 1 - P(\textrm{reject } H_0 \mid H_0 \textrm{ is true} ).$$

이는 또한 *제1종 오류* 또는 *거짓 양성*이라고도 합니다. $\alpha$는 *유의수준*이라고 불리며 일반적으로 사용되는 값은 $5\%$, 즉 $1-\alpha = 95\%$입니다. 유의수준은 저희가 참 영가설을 기각할 때 기꺼이 감수할 위험의 수준으로 설명될 수 있습니다.

:numref:`fig_statistical_significance`은 두 샘플 가설 검정에서 주어진 정규 분포의 관찰 값과 확률을 보여줍니다. 만약 관찰 데이터 예제가 $95\%$ 임계값 외부에 위치한다면, 영가설 가정 하에서 매우 가능성이 낮은 관찰이 될 것입니다. 따라서, 영가설에 문제가 있을 수 있고 저희는 그것을 기각할 것입니다.

![통계적 유의성.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


### 통계적 검정력

*통계적 검정력*(또는 *민감도*)은 영가설 $H_0$이 기각되어야 할 때 기각될 확률을 측정합니다. 즉,

$$ \textrm{statistical power }= 1 - \beta = 1 - P(\textrm{ fail to reject } H_0  \mid H_0 \textrm{ is false} ).$$

*제1종 오류*는 영가설이 참일 때 그것을 기각함으로써 야기된 오류인 반면, *제2종 오류*는 영가설이 거짓일 때 그것을 기각하지 못함으로써 발생합니다. 제2종 오류는 보통 $\beta$로 표기되며, 따라서 해당 통계적 검정력은 $1-\beta$입니다.


직관적으로, 통계적 검정력은 저희의 테스트가 원하는 통계적 유의수준에서 어떤 최소 크기의 실제 불일치를 얼마나 가능성 있게 감지할 수 있는지로 해석될 수 있습니다. $80\%$는 일반적으로 사용되는 통계적 검정력 임계값입니다. 통계적 검정력이 높을수록, 저희는 참 차이를 감지할 가능성이 더 높습니다.

통계적 검정력의 가장 일반적인 사용 중 하나는 필요한 샘플 수를 결정하는 것입니다. 영가설이 거짓일 때 그것을 기각할 확률은 그것이 거짓인 정도(*효과 크기*로 알려진)와 가지고 있는 샘플 수에 의존합니다. 예상할 수 있듯이, 작은 효과 크기는 높은 확률로 감지될 수 있도록 매우 많은 수의 샘플을 필요로 할 것입니다. 이 짧은 부록에서 자세히 유도하는 것은 범위를 벗어나지만, 예로, 저희의 샘플이 평균 0 분산 1 가우시안에서 나왔다는 영가설을 기각할 수 있기를 원하고, 저희의 샘플의 평균이 실제로 1에 가깝다고 믿는다면, 단지 $8$개의 샘플 크기로 허용 가능한 오류율로 그렇게 할 수 있습니다. 그러나, 만약 저희의 샘플 모집단 참 평균이 $0.01$에 가깝다고 생각한다면, 차이를 감지하기 위해 거의 $80000$개의 샘플 크기가 필요할 것입니다.

저희는 검정력을 정수기로 상상할 수 있습니다. 이 비유에서, 높은 검정력의 가설 검정은 물에서 유해 물질을 가능한 한 많이 줄일 고품질 정수 시스템과 같습니다. 반면에, 더 작은 불일치는 저품질 정수기와 같은데, 여기서 일부 상대적으로 작은 물질은 쉽게 틈에서 빠져나갈 수 있습니다. 유사하게, 만약 통계적 검정력이 충분히 높지 않다면, 테스트는 더 작은 불일치를 잡지 못할 수 있습니다.


### 검정 통계량

*검정 통계량* $T(x)$는 샘플 데이터의 어떤 특성을 요약하는 스칼라입니다. 그러한 통계량을 정의하는 목표는 그것이 다른 분포를 구별하고 저희의 가설 검정을 수행할 수 있게 해야 한다는 것입니다. 저희의 화학자 예제로 돌아가 생각해 보면, 만약 한 모집단이 다른 것보다 더 잘 수행한다는 것을 보이고 싶다면, 검정 통계량으로 평균을 취하는 것이 합리적일 수 있습니다. 검정 통계량의 다른 선택은 극단적으로 다른 통계적 검정력을 가진 통계적 검정으로 이어질 수 있습니다.

종종, $T(X)$(영가설 하에서 검정 통계량의 분포)는 영가설 하에서 고려될 때 적어도 대략적으로 정규 분포와 같은 일반적인 확률 분포를 따를 것입니다. 만약 그러한 분포를 명시적으로 유도하고, 그런 다음 저희의 데이터셋에서 저희의 검정 통계량을 측정할 수 있다면, 저희의 통계량이 기대할 범위 밖에 멀리 있다면 안전하게 영가설을 기각할 수 있습니다. 이것을 정량적으로 만들면 $p$-값의 개념으로 이어집니다.


### $p$-값

$p$-값(또는 *확률값*)은 영가설이 *참*이라고 가정할 때 $T(X)$가 관찰된 검정 통계량 $T(x)$만큼 극단적일 확률입니다. 즉,

$$ p\textrm{-value} = P_{H_0}(T(X) \geq T(x)).$$

만약 $p$-값이 사전 정의되고 고정된 통계적 유의수준 $\alpha$보다 작거나 같다면, 영가설을 기각할 수 있습니다. 그렇지 않으면, 저희는 영가설을 기각할 증거가 부족하다고 결론지을 것입니다. 주어진 모집단 분포의 경우, *기각 영역*은 통계적 유의수준 $\alpha$보다 작은 $p$-값을 가진 모든 점이 포함된 구간이 될 것입니다.


### 일측 검정과 양측 검정

보통 두 가지 종류의 유의성 검정이 있습니다. 일측 검정과 양측 검정. *일측 검정*(또는 *한쪽 꼬리 검정*)은 영가설과 대립가설이 한 방향만 가질 때 적용할 수 있습니다. 예를 들어, 영가설은 참 매개변수 $\theta$가 값 $c$보다 작거나 같다고 진술할 수 있습니다. 대립가설은 $\theta$가 $c$보다 크다는 것이 될 것입니다. 즉, 기각 영역은 샘플링 분포의 한쪽에만 있습니다. 일측 검정과 반대로, *양측 검정*(또는 *양쪽 꼬리 검정*)은 기각 영역이 샘플링 분포의 양쪽에 있을 때 적용할 수 있습니다. 이 경우의 예는 참 매개변수 $\theta$가 값 $c$와 같다는 영가설 상태를 가질 수 있습니다. 대립가설은 $\theta$가 $c$와 같지 않다는 것이 될 것입니다.


### 가설 검정의 일반적인 단계

위의 개념에 친숙해진 후, 가설 검정의 일반적인 단계를 살펴봅시다.

1. 질문을 진술하고 영가설 $H_0$을 수립합니다.
2. 통계적 유의수준 $\alpha$와 통계적 검정력 ($1 - \beta$)을 설정합니다.
3. 실험을 통해 샘플을 얻습니다. 필요한 샘플 수는 통계적 검정력과 예상 효과 크기에 따라 달라질 것입니다.
4. 검정 통계량과 $p$-값을 계산합니다.
5. $p$-값과 통계적 유의수준 $\alpha$에 기반하여 영가설을 유지하거나 기각할 결정을 내립니다.

가설 검정을 수행하기 위해, 저희는 영가설과 기꺼이 감수할 위험 수준을 정의함으로써 시작합니다. 그런 다음 저희는 샘플의 검정 통계량을 계산하며, 검정 통계량의 극단값을 영가설에 대한 증거로 취합니다. 만약 검정 통계량이 기각 영역에 떨어진다면, 저희는 대안에 찬성하여 영가설을 기각할 수 있습니다.

가설 검정은 임상 시험과 A/B 테스트와 같은 다양한 시나리오에서 적용 가능합니다.


## 신뢰 구간 구성


매개변수 $\theta$의 값을 추정할 때, $\hat \theta$와 같은 점 추정량은 불확실성의 개념을 포함하지 않기 때문에 유용성이 제한적입니다. 오히려, 만약 저희가 높은 확률로 참 매개변수 $\theta$를 포함할 구간을 생성할 수 있다면 훨씬 더 좋을 것입니다. 만약 한 세기 전에 그러한 아이디어에 관심이 있었다면, 1937년에 처음으로 신뢰 구간의 개념을 도입한 예지 네이만의 "고전적 확률 이론에 기반한 통계적 추정 이론의 개요" :cite:`Neyman.1937`를 읽는 것에 흥분했을 것입니다.

유용하기 위해서는, 신뢰 구간은 주어진 확실성 정도에 대해 가능한 한 작아야 합니다. 그것을 어떻게 유도하는지 봅시다.


### 정의

수학적으로, 참 매개변수 $\theta$에 대한 *신뢰 구간*은 다음과 같이 샘플 데이터로부터 계산된 구간 $C_n$입니다.

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

여기서 $\alpha \in (0, 1)$이고, $1 - \alpha$는 구간의 *신뢰수준* 또는 *커버리지*라고 합니다. 이것은 저희가 위에서 논의한 유의수준과 같은 $\alpha$입니다.

:eqref:`eq_confidence`는 고정된 $\theta$가 아니라 변수 $C_n$에 관한 것임에 유의하십시오. 이를 강조하기 위해, 저희는 $P_{\theta} (\theta \in C_n)$ 대신 $P_{\theta} (C_n \ni \theta)$를 씁니다.

### 해석

$95\%$ 신뢰 구간을 참 매개변수가 $95\%$ 확실히 있는 구간으로 해석하는 것은 매우 매력적이지만, 슬프게도 이것은 참이 아닙니다. 참 매개변수는 고정되어 있고, 무작위인 것은 구간입니다. 따라서 더 나은 해석은 이 절차에 의해 많은 수의 신뢰 구간을 생성한다면, 생성된 구간의 $95\%$가 참 매개변수를 포함할 것이라고 말하는 것입니다.

이는 현학적으로 보일 수 있지만, 결과 해석에 실제 영향을 미칠 수 있습니다. 특히, 저희는 충분히 드물게만 그렇게 하는 한, *거의 확실히* 참값을 포함하지 않는 구간을 구성함으로써 :eqref:`eq_confidence`를 만족시킬 수 있습니다. 저희는 세 가지 매력적이지만 거짓된 진술을 제공하면서 이 절을 마칩니다. 이러한 점에 대한 심층 논의는 :citet:`Morey.Hoekstra.Rouder.ea.2016`에서 찾을 수 있습니다.

* **오류 1**. 좁은 신뢰 구간은 저희가 매개변수를 정밀하게 추정할 수 있다는 것을 의미합니다.
* **오류 2**. 신뢰 구간 내의 값은 구간 외부의 값보다 참값일 가능성이 더 높습니다.
* **오류 3**. 특정 관찰된 $95\%$ 신뢰 구간이 참값을 포함할 확률은 $95\%$입니다.

신뢰 구간은 미묘한 객체라고 말하기에 충분합니다. 그러나, 해석을 명확하게 유지한다면, 그것들은 강력한 도구가 될 수 있습니다.

### 가우시안 예제

가장 고전적인 예제, 알려지지 않은 평균과 분산을 가진 가우시안의 평균에 대한 신뢰 구간을 논의해 봅시다. 저희의 가우시안 $\mathcal{N}(\mu, \sigma^2)$로부터 $n$개의 샘플 $\{x_i\}_{i=1}^n$을 수집한다고 가정해 봅시다. 저희는 다음을 취함으로써 평균과 분산에 대한 추정량을 계산할 수 있습니다.

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\textrm{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

이제 다음 확률 변수를 고려한다면

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

저희는 *$n-1$ 자유도의 스튜던트 t-분포*로 알려진 잘 알려진 분포를 따르는 확률 변수를 얻습니다.

이 분포는 매우 잘 연구되어 있으며, 예를 들어, $n\rightarrow \infty$일 때 그것이 대략 표준 가우시안임을 알고 있으며, 따라서 표에서 가우시안 c.d.f.의 값을 찾아봄으로써, $T$의 값이 적어도 시간의 $95\%$가 구간 $[-1.96, 1.96]$에 있다고 결론 내릴 수 있습니다. $n$의 유한한 값의 경우, 구간은 다소 더 커야 하지만, 잘 알려져 있고 표에서 미리 계산되어 있습니다.

따라서, 큰 $n$에 대해, 저희는 다음과 같이 결론 내릴 수 있습니다.

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

양변에 $\hat\sigma_n/\sqrt{n}$을 곱한 다음 $\hat\mu_n$을 더함으로써 이를 재배열하면, 저희는 다음을 얻습니다.

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

따라서 저희는 $95\%$ 신뢰 구간을 찾았음을 압니다.
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

:eqref:`eq_gauss_confidence`이 통계에서 가장 많이 사용되는 공식 중 하나라고 말하는 것은 안전합니다. 그것을 구현함으로써 통계 논의를 마치겠습니다. 단순함을 위해, 저희는 점근적 영역에 있다고 가정합니다. $N$의 작은 값은 프로그래밍적으로 또는 $t$-표에서 얻은 `t_star`의 올바른 값을 포함해야 합니다.

```{.python .input}
#@tab mxnet
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000

# Sample dataset
samples = tf.random.normal((N,), 0, 1)

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## 요약

* 통계는 추론 문제에 초점을 맞추는 반면, 딥러닝은 명시적으로 프로그래밍하고 이해하지 않고 정확한 예측을 하는 것을 강조합니다.
* 세 가지 일반적인 통계 추론 방법이 있습니다. 추정량을 평가하고 비교하는 것, 가설 검정을 수행하는 것, 그리고 신뢰 구간을 구성하는 것.
* 세 가지 가장 일반적인 추정량이 있습니다. 통계적 편향, 표준 편차, 그리고 평균 제곱 오차.
* 신뢰 구간은 샘플이 주어졌을 때 구성할 수 있는 참 모집단 매개변수의 추정 범위입니다.
* 가설 검정은 모집단에 대한 기본 진술에 대해 어떤 증거를 평가하는 방법입니다.


## 연습문제

1. $X_1, X_2, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0, \theta)$라고 하며, 여기서 "iid"는 *독립 동일 분포*를 의미합니다. $\theta$의 다음 추정량을 고려해 보십시오.
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * $\hat{\theta}$의 통계적 편향, 표준 편차, 그리고 평균 제곱 오차를 찾으십시오.
    * $\tilde{\theta}$의 통계적 편향, 표준 편차, 그리고 평균 제곱 오차를 찾으십시오.
    * 어떤 추정량이 더 낫습니까?
1. 도입부의 저희 화학자 예제의 경우, 양측 가설 검정을 수행하기 위한 5단계를 유도할 수 있습니까? 통계적 유의수준 $\alpha = 0.05$와 통계적 검정력 $1 - \beta = 0.8$이 주어졌을 때.
1. $100$개의 독립적으로 생성된 데이터셋에 대해 $N=2$와 $\alpha = 0.5$로 신뢰 구간 코드를 실행하고, 결과 구간을 플롯하십시오(이 경우 `t_star = 1.0`). 참 평균 $0$을 포함하는 것에서 매우 먼 매우 짧은 구간을 여러 개 보게 될 것입니다. 이것이 신뢰 구간의 해석과 모순됩니까? 높은 정밀도 추정을 나타내기 위해 짧은 구간을 사용하는 것이 편안하다고 느낍니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1103)
:end_tab:
