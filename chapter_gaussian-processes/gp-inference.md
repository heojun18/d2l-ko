```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# 가우스 과정 추론

이 절에서는 지난 절에서 소개한 GP 사전 분포를 사용하여 사후 추론을 수행하고 예측하는 방법을 보여드리겠습니다. _닫힌 형식(closed form)_으로 추론을 수행할 수 있는 회귀부터 시작하겠습니다. 이는 가우스 과정을 실제로 빠르게 시작할 수 있도록 하는 "한눈에 보는 GP" 절입니다. 모든 기본 연산을 처음부터(from scratch) 코딩하는 것부터 시작하고, 그런 다음 [GPyTorch](https://gpytorch.ai/)를 소개할 것입니다. 이는 최첨단 가우스 과정을 다루는 작업과 딥 신경망과의 통합을 훨씬 더 편리하게 만들어 줄 것입니다. 다음 절에서 이러한 더 고급 주제들을 깊이 있게 다룰 것입니다. 그 절에서는 또한 분류, 점 과정(point processes), 또는 어떤 비가우스 우도(likelihood)와 같이 근사 추론이 필요한 설정들도 고려할 것입니다.

## 회귀를 위한 사후 추론

_관측_ 모델은 저희가 학습하고자 하는 함수 $f(x)$를 관측치 $y(x)$에 연결하며, 둘 다 어떤 입력 $x$로 색인됩니다. 분류에서, $x$는 이미지의 픽셀일 수 있고, $y$는 관련된 클래스 라벨일 수 있습니다. 회귀에서, $y$는 일반적으로 지표면 온도, 해수면, $CO_2$ 농도 등과 같은 연속적인 출력을 나타냅니다.

회귀에서, 저희는 종종 출력이 잠재된 노이즈 없는 함수 $f(x)$에 i.i.d. 가우스 노이즈 $\epsilon(x)$를 더한 것에 의해 주어진다고 가정합니다.

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

여기서 $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$입니다. $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$을 저희 훈련 관측치의 벡터로 두고, $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$을 훈련 입력 $X = {x_1, \dots, x_n}$에서 질의된 잠재된 노이즈 없는 함숫값의 벡터로 둡시다.

저희는 $f(x) \sim \mathcal{GP}(m,k)$를 가정할 것이며, 이는 어떤 함숫값들의 모음 $\textbf{f}$도 평균 벡터 $\mu_i = m(x_i)$와 공분산 행렬 $K_{ij} = k(x_i,x_j)$를 갖는 결합 다변량 가우스 분포를 가짐을 의미합니다. RBF 커널 $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$은 공분산 함수의 표준적인 선택일 것입니다. 표기의 단순함을 위해, 평균 함수가 $m(x)=0$이라고 가정하겠습니다. 저희의 유도는 나중에 쉽게 일반화될 수 있습니다.

입력의 집합 $$X_* = x_{*1},x_{*2},\dots,x_{*m}$$에서 예측을 하고자 한다고 가정합시다. 그러면 $x^2$와 $p(\mathbf{f}_* | \mathbf{y}, X)$를 찾고자 합니다. 회귀 설정에서, $\mathbf{f}_* = f(X_*)$와 $\mathbf{y}$에 대한 결합 분포를 찾은 후 가우스 항등식(Gaussian identities)을 사용하여 이 분포를 편리하게 찾을 수 있습니다.

훈련 입력 $X$에서 식 :eqref:`eq_gp-regression`을 평가하면, $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$입니다. 가우스 과정의 정의(지난 절 참조)에 의해, $\mathbf{f} \sim \mathcal{N}(0,K(X,X))$이며, 여기서 $K(X,X)$는 모든 가능한 입력 쌍 $x_i, x_j \in X$에서 저희의 공분산 함수(즉, _커널_)를 평가하여 형성한 $n \times n$ 행렬입니다. $\mathbf{\epsilon}$은 단순히 $\mathcal{N}(0,\sigma^2)$로부터의 iid 표본으로 구성된 벡터이고 따라서 $\mathcal{N}(0,\sigma^2I)$ 분포를 갖습니다. 따라서 $\mathbf{y}$는 두 독립 다변량 가우스 변수의 합이며 따라서 $\mathcal{N}(0, K(X,X) + \sigma^2I)$ 분포를 갖습니다. 또한 $\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$임을 보일 수 있는데, 여기서 $K(X_*,X)$는 모든 테스트 및 훈련 입력 쌍에서 커널을 평가하여 형성한 $m \times n$ 행렬입니다.

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*) \\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}
\right)
$$

그런 다음 표준 가우스 항등식을 사용하여 결합 분포로부터 조건부 분포를 찾을 수 있습니다 (예: Bishop 2장 참조).
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$이며, 여기서 $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$이고 $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$입니다.

일반적으로, 저희는 전체 예측 공분산 행렬 $S$를 사용할 필요가 없으며, 대신 각 예측에 대한 불확실성을 위해 $S$의 대각 성분을 사용합니다. 이러한 이유로 종종 저희는 테스트 점들의 모음에 대해서가 아니라, 단일 테스트 점 $x_*$에 대한 예측 분포를 씁니다.

커널 행렬은 또한 추정하고자 하는 매개변수 $\theta$를 가지고 있습니다. 예를 들어 위 RBF 커널의 진폭 $a$와 길이 스케일 $\ell$이 그러합니다. 이러한 목적을 위해 저희는 _주변 우도(marginal likelihood)_ $p(\textbf{y} | \theta, X)$를 사용합니다. 이는 저희가 이미 $\textbf{y},\textbf{f}_*$에 대한 결합 분포를 찾으면서 주변 분포를 구하는 과정에서 유도했습니다. 살펴보겠지만, 주변 우도는 모델 적합과 모델 복잡도 항으로 분해되며, 하이퍼파라미터 학습을 위한 오컴의 면도날(Occam's razor) 개념을 자동으로 인코딩합니다. 전체 논의는 MacKay의 28장 :cite:`mackay2003information`과 Rasmussen and Williams의 5장 :cite:`rasmussen2006gaussian`을 참조하세요.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## GP 회귀에서 예측 수행 및 커널 하이퍼파라미터 학습을 위한 방정식들

여기서는 가우스 과정 회귀에서 하이퍼파라미터를 학습하고 예측을 수행하기 위해 사용할 방정식을 나열합니다. 다시, 입력 $X = \{x_1,\dots,x_n\}$으로 색인된 회귀 타깃의 벡터 $\textbf{y}$를 가정하고, 테스트 입력 $x_*$에서 예측을 하고자 합니다. 저희는 평균이 0이고 분산이 $\sigma^2$인 i.i.d. 가법 가우스 노이즈를 가정합니다. 평균 함수 $m$과 커널 함수 $k$를 갖는 가우스 과정 사전 분포 $f(x) \sim \mathcal{GP}(m,k)$를 잠재된 노이즈 없는 함수에 사용합니다. 커널 자체에는 저희가 학습하고자 하는 매개변수 $\theta$가 있습니다. 예를 들어, RBF 커널 $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$을 사용한다면, $\theta = \{a^2, \ell^2\}$을 학습하고자 합니다. $K(X,X)$를 $n$개의 훈련 입력의 모든 가능한 쌍에 대해 커널을 평가하는 것에 해당하는 $n \times n$ 행렬로 표기합니다. $K(x_*,X)$를 $k(x_*, x_i)$, $i=1,\dots,n$을 평가하여 형성한 $1 \times n$ 벡터로 표기합니다. $\mu$를 모든 훈련 점 $x$에서 평균 함수 $m(x)$를 평가하여 형성한 평균 벡터라 합시다.

일반적으로 가우스 과정으로 작업할 때, 저희는 두 단계 절차를 따릅니다.
1. 이러한 하이퍼파라미터에 대해 주변 우도를 최대화하여 커널 하이퍼파라미터 $\hat{\theta}$를 학습합니다.
2. 점 예측기로서 예측 평균을 사용하고, 학습된 하이퍼파라미터 $\hat{\theta}$에 조건을 부여하여 예측 표준편차의 2배를 95% 신용 집합을 형성하는 데 사용합니다.

로그 주변 우도는 단순히 로그 가우스 밀도이며, 다음과 같은 형태를 가집니다.
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

예측 분포는 다음과 같은 형태를 가집니다.
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## 학습 및 예측 방정식 해석

가우스 과정의 예측 분포에 대해 주목해야 할 몇 가지 핵심 사항이 있습니다.

* 이 모델 클래스의 유연성에도 불구하고, GP 회귀에 대해 _닫힌 형식_으로 _정확한(exact)_ 베이지안 추론을 수행하는 것이 가능합니다. 커널 하이퍼파라미터를 학습하는 것 외에는 _훈련_이 없습니다. 저희가 예측을 하기 위해 사용하고자 하는 방정식을 정확히 적을 수 있습니다. 가우스 과정은 이 점에서 비교적 예외적이며, 이는 그것의 편리성, 다재다능함, 그리고 지속적인 인기에 크게 기여했습니다.

* 예측 평균 $a_*$는 커널 $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$에 의해 가중된, 훈련 타깃 $\textbf{y}$의 선형 조합입니다. 살펴보겠지만, 커널(과 그 하이퍼파라미터)은 따라서 모델의 일반화 속성에 결정적인 역할을 합니다.

* 예측 평균은 타깃 값 $\textbf{y}$에 명시적으로 의존하지만 예측 분산은 그렇지 않습니다. 예측 불확실성은 대신 테스트 입력 $x_*$가 타깃 위치 $X$에서 멀어질수록, 커널 함수에 의해 지배되는 방식으로 증가합니다. 그러나 데이터로부터 학습된 커널 하이퍼파라미터 $\theta$를 통해, 불확실성은 암묵적으로 타깃 $\textbf{y}$의 값에 의존할 것입니다.

* 주변 우도는 모델 적합과 모델 복잡도(로그 행렬식) 항으로 분해됩니다. 주변 우도는 데이터와 여전히 일치하는 가장 단순한 적합을 제공하는 하이퍼파라미터를 선택하는 경향이 있습니다.

* 주요 계산 병목 현상은 $n$개의 훈련 점에 대한 $n \times n$ 대칭 양의 정부호 행렬 $K(X,X)$에 대한 선형 시스템을 푸는 것과 로그 행렬식을 계산하는 것에서 옵니다. 단순하게는, 이러한 연산들은 각각 $\mathcal{O}(n^3)$의 계산과, 커널(공분산) 행렬의 각 항목에 대한 $\mathcal{O}(n^2)$의 저장 공간을 발생시키며, 종종 콜레스키 분해(Cholesky decomposition)부터 시작합니다. 역사적으로, 이러한 병목 현상은 GP를 약 10,000개 미만의 훈련 점을 가진 문제로 제한해 왔으며, 이제는 거의 10년 동안 부정확한, "느리다(being slow)"는 평판을 GP에 부여해 왔습니다. 고급 주제에서, GP를 수백만 개의 점을 가진 문제로 어떻게 확장할 수 있는지 논의할 것입니다.

* 인기 있는 커널 함수 선택에 대해, $K(X,X)$는 종종 특이행렬에 가까운데, 이는 콜레스키 분해나 선형 시스템을 풀기 위한 다른 연산을 수행할 때 수치적 문제를 일으킬 수 있습니다. 다행히도, 회귀에서 저희는 종종 $K_{\theta}(X,X)+\sigma^2I$로 작업하며, 이는 노이즈 분산 $\sigma^2$이 $K(X,X)$의 대각에 더해져 조건수(conditioning)를 크게 개선합니다. 노이즈 분산이 작거나 노이즈 없는 회귀를 하는 경우, 조건수를 개선하기 위해 $10^{-6}$ 정도의 작은 양의 "지터(jitter)"를 대각에 더하는 것이 일반적인 관행입니다.


## 처음부터 작성하는 워크드 예제

회귀 데이터를 만들고, 모든 단계를 처음부터(from scratch) 구현하여 데이터에 GP를 적합시켜 봅시다.
저희는 $\epsilon \sim \mathcal{N}(0,\sigma^2)$일 때 $$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon,$$ 로부터 데이터를 표본 추출할 것입니다. 저희가 찾고자 하는 노이즈 없는 함수는 $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$입니다. 노이즈 표준편차 $\sigma = 0.25$를 사용하여 시작하겠습니다.

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

여기서는 노이즈가 있는 관측치를 원으로, 그리고 저희가 찾고자 하는 노이즈 없는 함수를 파란색으로 볼 수 있습니다.

이제, 잠재된 노이즈 없는 함수에 대해 GP 사전 분포 $f(x)\sim \mathcal{GP}(m,k)$를 명시하겠습니다. 평균 함수 $m(x) = 0$과 RBF 공분산 함수(커널)를 사용하겠습니다.
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

저희는 길이 스케일을 0.2로 시작했습니다. 데이터를 적합하기 전에, 합리적인 사전 분포를 명시했는지 고려하는 것이 중요합니다. 이 사전 분포로부터의 몇몇 표본 함수와 함께, 95% 신용 집합(저희는 참 함수가 이 영역 내에 있을 확률이 95%라고 믿습니다)을 시각화해 봅시다.

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

이러한 표본들이 합리적으로 보이나요? 함수의 고수준 속성이 저희가 모델링하고자 하는 데이터의 유형과 일치하나요?

이제 임의의 테스트 점 $x_*$에서 사후 예측 분포의 평균과 분산을 형성해 봅시다.

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

예측을 하기 전에, 커널 하이퍼파라미터 $\theta$와 노이즈 분산 $\sigma^2$을 학습해야 합니다. 저희가 적합시키고 있는 데이터에 비해 사전 함수가 너무 빠르게 변화하는 것처럼 보였으므로, 길이 스케일을 0.75로 초기화하겠습니다. 또한 노이즈 표준편차 $\sigma$를 0.75로 추측하겠습니다.

이러한 매개변수를 학습하기 위해, 이러한 매개변수에 대해 주변 우도를 최대화할 것입니다.

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$


아마도 저희 사전 함수는 너무 빠르게 변화했을 것입니다. 길이 스케일을 0.4로 추측해 봅시다. 또한 노이즈 표준편차를 0.75로 추측하겠습니다. 이들은 단순히 하이퍼파라미터 초기화일 뿐이며 (저희는 주변 우도로부터 이러한 매개변수를 학습할 것입니다).

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

이 경우, 길이 스케일 0.299와 노이즈 표준편차 0.24를 학습합니다. 학습된 노이즈는 참 노이즈와 매우 가깝다는 점에 주목하세요. 이는 저희 GP가 이 문제에 매우 잘 명시되었음을 나타내는 데 도움이 됩니다.

일반적으로, 커널을 선택하고 하이퍼파라미터를 초기화하는 데 신중한 생각을 기울이는 것이 중요합니다. 주변 우도 최적화가 초기화에 대해 비교적 견고할 수 있지만, 나쁜 초기화에 대해 면역되어 있지는 않습니다. 다양한 초기화로 위 스크립트를 실행해 보고 어떤 결과가 나오는지 확인해 보세요.

이제, 이러한 학습된 하이퍼파라미터로 예측을 해 봅시다.

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

주황색의 사후 평균이 참 노이즈 없는 함수와 거의 완벽하게 일치하는 것을 볼 수 있습니다! 저희가 보여주고 있는 95% 신용 집합은 잠재된 _노이즈 없는_ (참) 함수에 대한 것이지, 데이터 포인트에 대한 것이 아니라는 점에 유의하세요. 이 신용 집합이 참 함수를 완전히 포함하고 있으며, 지나치게 넓거나 좁아 보이지 않는다는 것을 볼 수 있습니다. 데이터 포인트를 포함하기를 원하지도, 기대하지도 않을 것입니다. 관측치에 대한 신용 집합을 갖고 싶다면, 다음을 계산해야 합니다.

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

불확실성에는 두 가지 원천이 있습니다. _인식적(epistemic)_ 불확실성은 _감소 가능한(reducible)_ 불확실성을 나타내고, _우연적(aleatoric)_ 또는 _감소 불가능한(irreducible)_ 불확실성이 있습니다. 여기서 _인식적_ 불확실성은 노이즈 없는 함수의 참값에 대한 불확실성을 나타냅니다. 이 불확실성은 데이터 포인트에서 멀어질수록 커져야 하는데, 데이터에서 떨어진 곳에서는 저희 데이터와 일치하는 함숫값들이 더 다양해지기 때문입니다. 더 많은 데이터를 관측함에 따라, 참 함수에 대한 저희의 믿음이 더 확신을 가지게 되고, 인식적 불확실성은 사라집니다. 이 경우의 _우연적_ 불확실성은 관측 노이즈이며, 데이터가 이 노이즈와 함께 저희에게 주어졌으므로, 이를 줄일 수는 없습니다.

데이터의 _인식적_ 불확실성은 잠재된 노이즈 없는 함수의 분산 np.diag(post\_cov)에 의해 포착됩니다. _우연적_ 불확실성은 노이즈 분산 post_sig_est**2에 의해 포착됩니다.

불행히도, 사람들은 종종 불확실성을 어떻게 표현하는지에 대해 부주의합니다. 많은 논문이 완전히 정의되지 않은 오차 막대를 보여주고, 저희가 인식적 또는 우연적 불확실성 중 어느 것을 또는 둘 모두를 시각화하고 있는지에 대한 명확한 감각이 없으며, 노이즈 분산과 노이즈 표준편차, 표준편차와 표준오차, 신뢰 구간과 신용 집합 등을 혼동합니다. 불확실성이 무엇을 나타내는지에 대해 정확하지 않으면, 그것은 본질적으로 무의미합니다.

저희의 불확실성이 무엇을 나타내는지에 세심한 주의를 기울이는 정신에 따라, 노이즈 없는 함수에 대한 분산 추정치의 _제곱근_의 _2배_를 취하고 있다는 점을 주목하는 것이 중요합니다. 저희 예측 분포가 가우스이므로, 이 양은 95% 신용 집합을 형성할 수 있게 해주며, 이는 기준 진리 함수를 포함할 확률이 95%인 구간에 대한 저희의 믿음을 나타냅니다. 노이즈 _분산_은 완전히 다른 스케일에 있으며, 훨씬 덜 해석 가능합니다.

마지막으로, 20개의 사후 표본을 살펴봅시다. 이 표본들은 저희가 사후적으로 데이터에 적합할 수 있다고 믿는 함수의 유형에 대해 알려줍니다.

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

기본적인 회귀 응용에서, 사후 예측 평균과 표준편차를 각각 점 예측기와 불확실성에 대한 척도로 사용하는 것이 가장 일반적입니다. 몬테카를로 획득 함수(Monte Carlo acquisition functions)를 사용한 베이지안 최적화나 모델 기반 강화 학습(model-based RL)을 위한 가우스 과정과 같은 더 고급 응용에서는, 종종 사후 표본을 취하는 것이 필요합니다. 그러나 기본 응용에서 엄격히 요구되지 않더라도, 이러한 표본들은 데이터에 대한 적합에 대해 더 많은 직관을 제공하며, 종종 시각화에 포함하는 것이 유용합니다.

## GPyTorch로 인생을 편하게

저희가 보았듯이, 기본 가우스 과정 회귀를 완전히 처음부터 구현하는 것은 사실 매우 쉽습니다. 그러나 다양한 커널 선택을 탐색하고, 근사 추론을 고려하고(분류에서도 필요), GP를 신경망과 결합하거나, 약 10,000개 이상의 데이터셋을 갖고자 하면, 처음부터 구현하는 것이 다루기 힘들고 번거로워집니다. SKI(KISS-GP라고도 알려진)와 같이 확장 가능한 GP 추론을 위한 가장 효과적인 방법 중 일부는, 고급 수치 선형 대수 루틴을 구현하는 수백 줄의 코드를 요구할 수 있습니다.

이러한 경우, _GPyTorch_ 라이브러리가 저희 인생을 훨씬 더 쉽게 만들어 줄 것입니다. 가우스 과정 수치 해석과 고급 방법에 대한 향후 노트북에서 GPyTorch에 대해 더 논의할 것입니다. GPyTorch 라이브러리에는 [많은 예제](https://github.com/cornellius-gp/gpytorch/tree/master/examples)가 있습니다. 패키지에 대한 감을 잡기 위해, [간단한 회귀 예제](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb)를 살펴볼 것이며, 이것이 어떻게 GPyTorch를 사용하여 저희 위 결과를 재현하도록 조정될 수 있는지 보여드릴 것입니다. 이는 단순히 위의 기본 회귀를 재현하기 위한 많은 양의 코드처럼 보일 수 있고, 어떤 의미에서는 그렇습니다. 하지만 잠재적으로 수천 줄의 새로운 코드를 작성하는 대신, 아래에서 몇 줄의 코드만 변경함으로써 즉시 다양한 커널, 확장 가능한 추론 기법, 그리고 근사 추론을 사용할 수 있습니다.

```{.python .input}
# First let's convert our data into tensors for use with PyTorch
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# We are using exact GP inference with a zero mean and RBF kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

이 코드 블록은 데이터를 GPyTorch에 맞는 형식으로 두고, 정확한 추론을 사용한다는 것과 사용하고자 하는
평균 함수(0)와 커널 함수(RBF)를 명시합니다. 예를 들어 gpytorch.kernels.matern_kernel() 또는 gpyotrch.kernels.spectral_mixture_kernel()을
호출함으로써, 다른 어떤 커널도 매우 쉽게 사용할 수 있습니다. 지금까지는 어떤 근사 없이
예측 분포를 추론하는 것이 가능한 정확한 추론만 논의했습니다.
가우스 과정의 경우, 가우스 우도를 가질 때만 정확한 추론을 수행할 수 있습니다. 더 구체적으로,
저희가 관측치가 가우스 과정으로 표현되는 노이즈 없는 함수에 가우스 노이즈가 더해진 것으로 생성된다고
가정할 때입니다. 향후 노트북에서는 이러한 가정을 할 수 없는, 분류와 같은 다른 설정들을 고려할 것입니다.

```{.python .input}
# Initialize Gaussian likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use the adam optimizer, includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# Set our loss as the negative log GP marginal likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

여기서는 사용하고자 하는 우도(가우스), 커널 하이퍼파라미터를 훈련하는 데 사용할 목적함수(여기서는 주변 우도), 그리고 그 목적함수를 최적화하는 데 사용하고자 하는 절차(이 경우 Adam)를 명시적으로 지정합니다. 저희는 Adam(이는 "확률적(stochastic)" 옵티마이저)을 사용하고 있지만, 이 경우에는 풀배치 Adam이라는 점에 주목합니다. 주변 우도는 데이터 인스턴스에 대해 인수분해되지 않으므로, 데이터의 "미니배치"에 대한 옵티마이저를 사용하면서 수렴이 보장될 수 없습니다. L-BFGS와 같은 다른 옵티마이저도 GPyTorch에서 지원됩니다. 표준 딥러닝과 달리, 주변 우도를 잘 최적화하는 것은 좋은 일반화와 강하게 상관관계가 있으며, 이는 종종 L-BFGS와 같은 강력한 옵티마이저(터무니없이 비싸지 않다고 가정할 때)로 저희를 기울게 합니다.

```{.python .input}
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

여기서는 실제로 최적화 절차를 실행하며, 10 반복마다 손실 값을 출력합니다.

```{.python .input}
# Get into evaluation (predictive posterior) mode
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

위 코드 블록은 저희가 테스트 입력에 대해 예측할 수 있게 해줍니다.

```{.python .input}
with torch.no_grad():
    # Initialize plot
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower bounds for 95\% credible set (in this case, in
    # observation space)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

마지막으로, 적합을 플롯합니다.

적합이 사실상 동일하다는 것을 볼 수 있습니다. 몇 가지 주목할 점이 있습니다. GPyTorch는 _제곱_ 길이 스케일과 관측 노이즈로 작업하고 있습니다. 예를 들어, 처음부터 작성한 코드에서 학습된 노이즈 표준편차는 약 0.283입니다. GPyTorch가 찾은 노이즈 분산은 $0.81 \approx 0.283^2$입니다. GPyTorch 플롯에서는 잠재 함수 공간이 아니라 _관측 공간(observation space)_에서의 신용 집합도 보여주는데, 이는 실제로 관측된 데이터 포인트들을 덮고 있음을 보이기 위함입니다.

## 요약

가우스 과정 사전 분포와 데이터를 결합하여 예측을 하는 데 사용하는 사후 분포를 형성할 수 있습니다. 또한 주변 우도를 형성할 수 있으며, 이는 가우스 과정의 변동률과 같은 속성을 제어하는 커널 하이퍼파라미터의 자동 학습에 유용합니다. 회귀에서 사후 분포를 형성하고 커널 하이퍼파라미터를 학습하는 메커니즘은 단순하며, 약 12줄의 코드를 포함합니다. 이 노트북은 가우스 과정을 빠르게 "실행"하고자 하는 어떤 독자에게도 좋은 참고가 됩니다. 또한 GPyTorch 라이브러리를 소개했습니다. 기본 회귀를 위한 GPyTorch 코드는 비교적 길지만, 다른 커널 함수나, 확장 가능한 추론, 분류를 위한 비가우스 우도와 같이 향후 노트북에서 논의할 더 고급 기능을 위해 사소하게 수정될 수 있습니다.


## 연습 문제

1. 저희는 커널 하이퍼파라미터를 _학습_하는 것의 중요성과, 하이퍼파라미터와 커널이 가우스 과정의 일반화 속성에 미치는 영향을 강조해 왔습니다. 하이퍼파라미터를 학습하는 단계를 건너뛰고, 대신 다양한 길이 스케일과 노이즈 분산을 추측해보고, 그것들이 예측에 미치는 영향을 확인해 보세요. 큰 길이 스케일을 사용하면 어떻게 되나요? 작은 길이 스케일은요? 큰 노이즈 분산은요? 작은 노이즈 분산은요?
2. 저희는 주변 우도가 볼록한 목적함수가 아니지만, 길이 스케일과 노이즈 분산과 같은 하이퍼파라미터는 GP 회귀에서 신뢰성 있게 추정될 수 있다고 말해왔습니다. 이는 일반적으로 사실입니다 (실제로 주변 우도는 경험적 자기상관 함수("코바리오그램(covariograms)")를 적합시키는 것을 포함하는 공간 통계학에서의 전통적인 접근법보다 길이 스케일 하이퍼파라미터를 학습하는 데 _훨씬_ 더 우수합니다). 머신러닝이 가우스 과정 연구에 기여한 가장 큰 공헌은, 적어도 확장 가능한 추론에 대한 최근 연구 이전에는, 아마도 하이퍼파라미터 학습을 위한 주변 우도의 도입이었을 것입니다.

*그러나*, 심지어 이러한 매개변수들의 서로 다른 짝짓기조차도 많은 데이터셋에 대해 해석 가능한 그럴듯한 설명을 제공하여, 저희의 목적함수에서 지역 최적점(local optima)으로 이어집니다. 큰 길이 스케일을 사용하면, 참 기저 함수가 천천히 변한다고 가정합니다. 관측 데이터가 상당히 변한다면, 그럴듯하게 큰 길이 스케일을 가질 수 있는 유일한 방법은 큰 노이즈 분산을 갖는 것입니다. 반면, 작은 길이 스케일을 사용하면, 저희 적합은 데이터의 변동에 매우 민감해져서, 노이즈(우연적 불확실성)로 변동을 설명할 여지를 거의 남기지 않습니다.

이러한 지역 최적점들을 찾을 수 있는지 시도해 보세요. 매우 큰 길이 스케일과 큰 노이즈, 그리고 작은 길이 스케일과 작은 노이즈로 초기화해 보세요. 다른 해로 수렴하나요?
  
3. 저희는 베이지안 방법의 근본적인 이점은 자연스럽게 _인식적_ 불확실성을 표현하는 것이라고 말해왔습니다. 위 예제에서는, 인식적 불확실성의 효과를 완전히 볼 수 없습니다. 대신 `test_x = np.linspace(0, 10, 1000)`으로 예측해 보세요. 예측이 데이터를 넘어 이동함에 따라 95% 신용 집합에 어떤 일이 일어나나요? 그 구간에서 참 함수를 덮나요? 그 영역에서 우연적 불확실성만 시각화하면 어떻게 되나요?

4. 위 예제를 10,000개, 20,000개, 40,000개의 훈련 점으로 대신 실행해 보고, 실행 시간을 측정해 보세요. 훈련 시간은 어떻게 스케일링되나요? 또는, 테스트 점의 수에 따라 실행 시간은 어떻게 스케일링되나요? 예측 평균과 예측 분산에 대해 다른가요? 이 질문에 대해 훈련 및 테스트 시간 복잡도를 이론적으로 계산해서, 그리고 다른 점의 수로 위 코드를 실행해서 답해 보세요.

5. 다른 공분산 함수, 예를 들어 Matern 커널로 GPyTorch 예제를 실행해 보세요. 결과는 어떻게 변하나요? GPyTorch 라이브러리에서 찾을 수 있는 스펙트럼 혼합 커널(spectral mixture kernel)은 어떤가요? 어떤 것은 다른 것보다 주변 우도를 훈련하기 더 쉬운가요? 장거리 대 단거리 예측에 더 가치 있는 것이 있나요?

6. 저희 GPyTorch 예제에서는, 관측 노이즈를 포함한 예측 분포를 플롯했고, 반면 "처음부터" 예제에서는 인식적 불확실성만 포함했습니다. GPyTorch 예제를 다시 수행하되, 이번에는 인식적 불확실성만 플롯하고, 처음부터 작성한 결과와 비교해 보세요. 예측 분포가 이제 동일하게 보이나요? (그래야 합니다.)

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12117)
:end_tab:
