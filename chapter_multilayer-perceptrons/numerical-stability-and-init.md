```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 수치 안정성과 초기화
:label:`sec_numerical_stability`


지금까지 저희가 구현한 모든 모델은
사전에 지정된 어떤 분포에 따라 그 파라미터를
초기화해야 했습니다.
지금까지 저희는 초기화 방식을 당연한 것으로 여기고,
이러한 선택이 어떻게 이루어지는지에 대한 세부 사항을 그저 넘겨 왔습니다.
어쩌면 여러분은 이러한 선택이
별로 중요하지 않다는 인상을 받았을지도 모릅니다.
오히려, 초기화 방식의 선택은
신경망 학습에서 상당한 역할을 하며,
수치 안정성을 유지하는 데 결정적일 수 있습니다.
게다가, 이러한 선택은 비선형 활성화 함수의 선택과
흥미로운 방식으로 얽힐 수 있습니다.
어떤 함수를 선택하고 파라미터를 어떻게 초기화하는지에 따라
저희 최적화 알고리즘이 얼마나 빨리 수렴하는지가 결정될 수 있습니다.
여기서 잘못된 선택은 훈련 중에 기울기 폭주나
기울기 소실을 겪게 만들 수 있습니다.
이 절에서, 저희는 이러한 주제를 더 자세히 파헤치고
딥러닝 경력 전반에 걸쳐 유용할
유용한 휴리스틱 몇 가지를
논의합니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
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
from jax import grad, vmap
```

## 기울기 소실과 폭주

$L$개의 층, 입력 $\mathbf{x}$, 그리고 출력 $\mathbf{o}$를 갖는 심층 네트워크를 고려합시다.
각 층 $l$은 가중치 $\mathbf{W}^{(l)}$로 매개변수화되는 변환 $f_l$로 정의되고,
그 은닉층 출력이 $\mathbf{h}^{(l)}$이라고 합시다 (($\mathbf{h}^{(0)} = \mathbf{x}$라고 둡니다)).
저희 네트워크는 다음과 같이 표현될 수 있습니다.

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

모든 은닉층 출력과 입력이 벡터라면,
저희는 임의의 파라미터 집합 $\mathbf{W}^{(l)}$에 대한
$\mathbf{o}$의 기울기를 다음과 같이 쓸 수 있습니다.

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

다시 말해, 이 기울기는
$L-l$개의 행렬
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$과
기울기 벡터 $\mathbf{v}^{(l)}$의 곱입니다.
따라서 저희는 너무 많은 확률을 함께 곱할 때
자주 발생하는 수치적 언더플로의 동일한
문제에 취약합니다.
확률을 다룰 때, 흔히 사용되는 요령은
로그 공간으로 전환하는 것입니다. 즉, 수치 표현의 가수에서
지수로 압력을 옮기는 것입니다.
유감스럽게도, 위의 저희 문제는 더 심각합니다.
처음에 행렬 $\mathbf{M}^{(l)}$은 매우 다양한 고윳값을 가질 수 있습니다.
이들은 작을 수도 클 수도 있고,
그 곱은 *매우 클* 수도 *매우 작을* 수도 있습니다.

불안정한 기울기가 초래하는 위험은
수치 표현을 넘어섭니다.
예측할 수 없는 크기의 기울기는
저희 최적화 알고리즘의 안정성도 위협합니다.
저희는 (i) 지나치게 커서 모델을 파괴하는
((*기울기 폭주* 문제)),
또는 (ii) 지나치게 작아서
((*기울기 소실* 문제))
파라미터가 각 업데이트마다 거의 움직이지 않아 학습을 불가능하게 만드는
파라미터 업데이트에 직면할 수 있습니다.


### (**기울기 소실**)

기울기 소실 문제를 일으키는 흔한 원인 중 하나는
각 층의 선형 연산 뒤에 따라붙는
활성화 함수 $\sigma$의 선택입니다.
역사적으로, 시그모이드 함수
$1/(1 + \exp(-x))$ (:numref:`sec_mlp`에서 소개))는
임계값 함수와 닮았기 때문에 인기가 있었습니다.
초기 인공 신경망은 생물학적 신경망에서 영감을 받았으므로,
((생물학적 뉴런처럼)) *완전히* 발화하거나 *전혀* 발화하지 않는 뉴런의
아이디어가 매력적으로 보였습니다.
시그모이드가 왜 기울기 소실을 일으킬 수 있는지 보기 위해
시그모이드를 더 자세히 살펴봅시다.

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

보시다시피, (**시그모이드의 기울기는
입력이 크거나 작을 때 모두 사라집니다**).
게다가, 여러 층에 걸쳐 역전파할 때,
많은 시그모이드의 입력이 0에 가까운
Goldilocks 영역에 있지 않은 한,
전체 곱의 기울기는 사라질 수 있습니다.
저희 네트워크가 많은 층을 가지고 있을 때,
주의하지 않으면 기울기는
어떤 층에서 끊어질 가능성이 큽니다.
실제로 이 문제는 심층 네트워크 훈련을 괴롭히곤 했습니다.
결과적으로, 더 안정적인 ((그러나 생물학적으로는 덜 그럴듯한)) ReLU가
실무자들의 기본 선택지로 부상했습니다.


### [**기울기 폭주**]

기울기가 폭주하는 반대의 문제도
마찬가지로 까다로울 수 있습니다.
이를 좀 더 잘 설명하기 위해,
저희는 100개의 가우시안 무작위 행렬을 뽑아
어떤 초기 행렬과 곱합니다.
저희가 선택한 크기((분산 $\sigma^2=1$의 선택))에서,
행렬 곱은 폭주합니다.
이것이 심층 네트워크의 초기화로 인해 발생할 때,
저희는 경사 하강법 최적화기가 수렴하도록 할 가능성이
전혀 없습니다.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # Generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### 대칭성 깨기

신경망 설계의 또 다른 문제는
그 매개변수화에 내재된 대칭성입니다.
하나의 은닉층과 두 개의 유닛을 가진 단순한 MLP가
있다고 가정합시다.
이 경우, 저희는 첫 번째 층의 가중치 $\mathbf{W}^{(1)}$을
순열하고 마찬가지로 출력층의 가중치를
순열하여 동일한 함수를 얻을 수 있습니다.
첫 번째와 두 번째 은닉 유닛을 구별하는
특별한 것이 전혀 없습니다.
다시 말해, 저희는 각 층의 은닉 유닛들 사이에
순열 대칭성을 가지고 있습니다.

이것은 단순한 이론적 골칫거리 이상의 것입니다.
앞서 언급한 두 개의 은닉 유닛을 가진
단일 은닉층 MLP를 고려해 봅시다.
설명을 위해,
출력층이 두 개의 은닉 유닛을 단 하나의 출력 유닛으로 변환한다고 가정합시다.
은닉층의 모든 파라미터를 어떤 상수 $c$에 대해
$\mathbf{W}^{(1)} = c$로 초기화하면 어떤 일이 일어날지
상상해 봅시다.
이 경우, 순전파 동안
어느 은닉 유닛이든 동일한 입력과 파라미터를 받아
동일한 활성화를 생성하고
이것이 출력 유닛에 전달됩니다.
역전파 동안,
파라미터 $\mathbf{W}^{(1)}$에 대해 출력 유닛을 미분하면 모든 원소가 동일한 값을 갖는 기울기가 나옵니다.
따라서, 기울기 기반 반복 ((예: 미니배치 확률적 경사 하강법)) 이후에도,
$\mathbf{W}^{(1)}$의 모든 원소는 여전히 동일한 값을 갖습니다.
이러한 반복은
스스로 *대칭성을 깰* 수 없을 것이며
저희는 네트워크의 표현력을
결코 실현하지 못할 수도 있습니다.
은닉층은 마치 단 하나의 유닛만 가진 것처럼
동작할 것입니다.
미니배치 확률적 경사 하강법은 이 대칭성을 깨지 못하지만,
드롭아웃 정규화 ((나중에 소개))는 깰 수 있다는 점에 유의하세요!


## 파라미터 초기화

위에서 제기된 문제들을 해결하는 (또는 적어도 완화하는) 한 가지 방법은
신중한 초기화를 통해서입니다.
나중에 보시겠지만,
최적화 동안의 추가적인 주의와
적절한 정규화는 안정성을 더욱 향상시킬 수 있습니다.


### 기본 초기화

이전 절들, 예를 들어 :numref:`sec_linear_concise`에서,
저희는 가중치 값을 초기화하기 위해
정규분포를 사용했습니다.
초기화 방법을 지정하지 않으면, 프레임워크는
기본 무작위 초기화 방법을 사용하며, 이는 보통의 문제 크기에서는 실제로 잘 작동하는 경우가 많습니다.




### Xavier 초기화
:label:`subsec_xavier`

*비선형성이 없는* 어떤 완전 연결 층에 대한 출력 $o_{i}$의
크기 분포를 살펴봅시다.
이 층에 대한 $n_\textrm{in}$개의 입력 $x_j$와
관련된 가중치 $w_{ij}$를 가지고,
출력은 다음과 같이 주어집니다.

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

가중치 $w_{ij}$는 모두 동일한 분포에서
독립적으로 뽑힙니다.
또한, 이 분포가 평균이 0이고 분산이 $\sigma^2$이라고 가정합시다.
이는 분포가 가우시안이어야 한다는 의미가 아니라,
단지 평균과 분산이 존재해야 한다는 의미라는 점에 유의하세요.
지금은 층에 대한 입력 $x_j$도 평균이 0이고 분산이 $\gamma^2$이며
$w_{ij}$와 독립적이고 서로 독립적이라고
가정합시다.
이 경우, 저희는 $o_i$의 평균을 계산할 수 있습니다.

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

그리고 분산은 다음과 같습니다.

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

분산을 고정으로 유지하는 한 가지 방법은
$n_\textrm{in} \sigma^2 = 1$로 설정하는 것입니다.
이제 역전파를 고려해 봅시다.
거기서 저희는 비슷한 문제에 직면하는데,
다만 기울기가 출력에 가까운 층들에서 전파됩니다.
순전파에 대한 것과 동일한 추론을 사용하면,
저희는 이 층의 출력 수인 $n_\textrm{out}$에 대해
$n_\textrm{out} \sigma^2 = 1$이 아닌 한
기울기의 분산이 폭주할 수 있음을 알 수 있습니다.
이는 저희를 딜레마에 빠뜨립니다.
두 조건을 동시에 만족시키는 것은 불가능합니다.
대신, 저희는 단순히 다음을 만족시키려 시도합니다.

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

이것이 이제 표준이 되고 실용적으로 유익한
*Xavier 초기화*의 기저에 있는 추론으로,
그 창안자의 제1저자 이름을 따서 명명되었습니다 :cite:`Glorot.Bengio.2010`.
일반적으로, Xavier 초기화는
평균이 0이고 분산이
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$인 가우시안 분포에서
가중치를 샘플링합니다.
저희는 또한 이를 균등 분포에서 가중치를 샘플링할 때
분산을 선택하도록
적용할 수도 있습니다.
균등 분포 $U(-a, a)$는 분산이 $\frac{a^2}{3}$이라는 점에 유의하세요.
$\frac{a^2}{3}$을 $\sigma^2$에 대한 저희의 조건에 대입하면
저희는 다음에 따라 초기화하게 됩니다.

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

위의 수학적 추론에서 비선형성이 존재하지 않는다는 가정은
신경망에서 쉽게 위배될 수 있지만,
Xavier 초기화 방법은
실제로 잘 작동하는 것으로 밝혀졌습니다.


### 그 너머

위의 추론은 파라미터 초기화에 대한 현대적 접근법의
표면만을 살짝 긁었을 뿐입니다.
딥러닝 프레임워크는 종종 12가지가 넘는 서로 다른 휴리스틱을 구현합니다.
게다가, 파라미터 초기화는
딥러닝의 근본적 연구에서 여전히 뜨거운 영역으로
남아 있습니다.
이 중에는 묶인 ((공유된)) 파라미터, 초고해상도,
시퀀스 모델, 그리고 기타 상황에 특화된 휴리스틱이 있습니다.
예를 들어,
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018`은 신중하게 설계된 초기화 방법을 사용하여
구조적 요령 없이 10,000층 신경망을 훈련할 수 있는
가능성을 보여주었습니다.

이 주제가 흥미롭다면 저희는
이 모듈이 제공하는 것들을 깊이 살펴보고,
각 휴리스틱을 제안하고 분석한 논문들을 읽고,
그런 다음 그 주제에 대한 최신 출판물들을 탐색해 볼 것을 권합니다.
어쩌면 여러분은 영리한 아이디어를 우연히 발견하거나 심지어 발명하여
딥러닝 프레임워크에 구현을 기여하게 될지도 모릅니다.


## 요약

기울기 소실과 폭주는 심층 네트워크에서 흔한 문제입니다. 기울기와 파라미터가 잘 통제된 상태로 유지되도록 하려면 파라미터 초기화에 큰 주의가 필요합니다.
초기 기울기가 너무 크지도 작지도 않도록 보장하기 위해 초기화 휴리스틱이 필요합니다.
최적화 전에 대칭성이 깨지도록 보장하는 데에는 무작위 초기화가 핵심입니다.
Xavier 초기화는, 각 층에 대해, 어떤 출력의 분산도 입력의 수에 영향을 받지 않고, 어떤 기울기의 분산도 출력의 수에 영향을 받지 않도록 제안합니다.
ReLU 활성화 함수는 기울기 소실 문제를 완화합니다. 이는 수렴을 가속할 수 있습니다.

## 연습문제

1. MLP 층의 순열 대칭성 외에, 신경망이 깨야 할 대칭성을 보일 수 있는 다른 경우를 설계할 수 있나요?
1. 선형 회귀나 소프트맥스 회귀에서 모든 가중치 파라미터를 동일한 값으로 초기화할 수 있나요?
1. 두 행렬의 곱의 고윳값에 대한 분석적 경계를 찾아보세요. 이것이 기울기가 잘 조건화되도록 보장하는 것에 대해 무엇을 시사하나요?
1. 어떤 항이 발산한다는 것을 안다면, 사후에 이를 고칠 수 있나요? 영감을 얻기 위해 층별 적응적 비율 스케일링에 관한 논문을 살펴보세요 :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17986)
:end_tab:
