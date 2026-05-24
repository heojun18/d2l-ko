# 순전파, 역전파, 그리고 계산 그래프
:label:`sec_backprop`

지금까지 저희는 미니배치 확률적 경사 하강법으로
모델을 훈련해 왔습니다.
하지만 알고리즘을 구현할 때,
저희는 모델을 통한 *순전파*와 관련된 계산만
신경 썼습니다.
기울기를 계산할 때가 되면,
저희는 단순히 딥러닝 프레임워크가 제공하는 역전파 함수를 호출했습니다.

기울기의 자동 계산은
딥러닝 알고리즘의 구현을
크게 단순화합니다.
자동 미분 이전에는,
복잡한 모델에 작은 변경만 가해도
복잡한 도함수를 손으로 다시 계산해야 했습니다.
놀랍게도 자주, 학술 논문은
업데이트 규칙을 유도하는 데에만 수많은 페이지를
할애해야 했습니다.
저희가 흥미로운 부분에 집중할 수 있도록
계속해서 자동 미분에 의존해야겠지만,
딥러닝에 대한 얕은 이해를
넘어서고 싶다면, 이러한 기울기가
내부에서 어떻게 계산되는지 알아야 합니다.

이 절에서, 저희는 *역전파*((더 흔히는 *backpropagation*이라고 불립니다))의
세부 사항을 깊이 살펴봅니다.
기법과 그 구현에 대한 어느 정도의 통찰을 전달하기 위해,
저희는 몇 가지 기본 수학과 계산 그래프에 의존합니다.
시작을 위해, 저희는 가중치 감쇠(($\ell_2$ 정규화, 이후 장에서 설명)))를 가진
단일 은닉층 MLP에 설명을 집중합니다.

## 순전파

*순전파* ((또는 *forward pass*))는 입력층에서 출력층까지
순서대로 신경망의 중간 변수((출력 포함))를
계산하고 저장하는 것을 가리킵니다.
이제 저희는 하나의 은닉층을 가진 신경망의 메커니즘을
단계별로 살펴봅니다.
지루해 보일 수도 있지만, 펑크의 거장
James Brown의 영원한 말씀처럼,
여러분은 "보스가 되려면 그 대가를 치러야" 합니다.


단순함을 위해, 입력 예제가 $\mathbf{x}\in \mathbb{R}^d$이고
저희 은닉층이 편향 항을 포함하지 않는다고 가정합시다.
여기서 중간 변수는 다음과 같습니다.

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

여기서 $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$는
은닉층의 가중치 파라미터입니다.
중간 변수 $\mathbf{z}\in \mathbb{R}^h$를
활성화 함수 $\phi$에 통과시킨 후
저희는 길이 $h$의 은닉 활성화 벡터를 얻습니다.

$$\mathbf{h}= \phi (\mathbf{z}).$$

은닉층 출력 $\mathbf{h}$도
중간 변수입니다.
출력층의 파라미터가
오직 $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$의 가중치만
가진다고 가정하면,
저희는 길이 $q$의 벡터를 갖는 출력층 변수를
얻을 수 있습니다.

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

손실 함수가 $l$이고
예제 레이블이 $y$라고 가정하면,
저희는 단일 데이터 예제에 대한
손실 항을 다음과 같이 계산할 수 있습니다.

$$L = l(\mathbf{o}, y).$$

나중에 도입될 $\ell_2$ 정규화의 정의에서 보게 되듯이,
하이퍼파라미터 $\lambda$가 주어지면,
정규화 항은 다음과 같습니다.

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_\textrm{F}^2 + \|\mathbf{W}^{(2)}\|_\textrm{F}^2\right),$$
:eqlabel:`eq_forward-s`

여기서 행렬의 프로베니우스 노름은
단순히 행렬을 벡터로 평탄화한 후 적용된
$\ell_2$ 노름입니다.
마지막으로, 주어진 데이터 예제에 대한 모델의 정규화된 손실은
다음과 같습니다.

$$J = L + s.$$

저희는 다음 논의에서 $J$를
*목적 함수*라고 부릅니다.


## 순전파의 계산 그래프

*계산 그래프*를 그리면 계산 내의 연산자와 변수의
의존성을 시각화하는 데 도움이 됩니다.
:numref:`fig_forward`는 위에서 설명한 단순한 네트워크에 해당하는
그래프를 담고 있으며,
여기서 사각형은 변수를, 원은 연산자를 나타냅니다.
왼쪽 아래 모서리는 입력을 나타내고
오른쪽 위 모서리는 출력입니다.
((데이터 흐름을 보여주는)) 화살표의 방향이
주로 오른쪽과 위쪽을 향한다는 점에 유의하세요.

![순전파의 계산 그래프.](../img/forward.svg)
:label:`fig_forward`

## 역전파

*역전파*는 신경망 파라미터의 기울기를 계산하는
방법을 가리킵니다.
간단히 말해, 이 방법은 미적분의 *연쇄 법칙*에 따라
출력에서 입력층으로 역순으로
네트워크를 순회합니다.
알고리즘은 어떤 파라미터에 대한 기울기를
계산하면서 필요한 중간 변수
((편도함수))를 저장합니다.
함수
$\mathsf{Y}=f(\mathsf{X})$와 $\mathsf{Z}=g(\mathsf{Y})$가
있다고 가정하고, 여기서 입력과 출력
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$는
임의 형상의 텐서라고 합시다.
연쇄 법칙을 사용함으로써,
저희는 다음을 통해 $\mathsf{X}$에 대한 $\mathsf{Z}$의
도함수를 계산할 수 있습니다.

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \textrm{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

여기서 저희는 $\textrm{prod}$ 연산자를 사용하여
전치 및 입력 위치 교환과 같은 필요한 연산이
수행된 후 인수들을 곱합니다.
벡터의 경우, 이는 직관적입니다.
단순히 행렬(-)행렬 곱셈입니다.
더 높은 차원의 텐서의 경우,
저희는 적절한 대응물을 사용합니다.
$\textrm{prod}$ 연산자는 모든 표기적 부담을 숨깁니다.

:numref:`fig_forward`에 계산 그래프가 있는, 하나의 은닉층을 가진 단순한 네트워크의
파라미터는 $\mathbf{W}^{(1)}$과 $\mathbf{W}^{(2)}$임을
떠올려 봅시다.
역전파의 목표는
기울기 $\partial J/\partial \mathbf{W}^{(1)}$과
$\partial J/\partial \mathbf{W}^{(2)}$를 계산하는 것입니다.
이를 달성하기 위해, 저희는 연쇄 법칙을 적용하고
각 중간 변수와 파라미터의 기울기를
차례로 계산합니다.
순전파에서 수행되는 계산과 비교하여 계산 순서가 뒤집힙니다.
왜냐하면 저희는 계산 그래프의 결과에서 시작하여
파라미터를 향해 거꾸로 진행해야 하기 때문입니다.
첫 단계는 손실 항 $L$과 정규화 항 $s$에 대한
목적 함수 $J=L+s$의 기울기를
계산하는 것입니다.

$$\frac{\partial J}{\partial L} = 1 \; \textrm{and} \; \frac{\partial J}{\partial s} = 1.$$

다음으로, 저희는 연쇄 법칙에 따라
출력층 변수 $\mathbf{o}$에 대한 목적 함수의 기울기를
계산합니다.

$$
\frac{\partial J}{\partial \mathbf{o}}
= \textrm{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

다음으로, 저희는 두 파라미터 모두에 대한
정규화 항의 기울기를 계산합니다.

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \textrm{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

이제 저희는 출력층에 가장 가까운 모델 파라미터의
기울기 $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$를
계산할 수 있습니다.
연쇄 법칙을 사용하면 다음을 얻습니다.

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

$\mathbf{W}^{(1)}$에 대한 기울기를 얻으려면,
저희는 출력층을 따라 은닉층까지 역전파를
계속해야 합니다.
은닉층 출력에 대한 기울기
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$는 다음과 같이 주어집니다.


$$
\frac{\partial J}{\partial \mathbf{h}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

활성화 함수 $\phi$가 원소별로 적용되므로,
중간 변수 $\mathbf{z}$의 기울기
$\partial J/\partial \mathbf{z} \in \mathbb{R}^h$를 계산하려면
저희는 $\odot$로 표기하는 원소별 곱셈 연산자를
사용해야 합니다.

$$
\frac{\partial J}{\partial \mathbf{z}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

마지막으로, 저희는 입력층에 가장 가까운 모델 파라미터의
기울기 $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$를
얻을 수 있습니다.
연쇄 법칙에 따라 저희는 다음을 얻습니다.

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## 신경망 훈련

신경망을 훈련할 때,
순전파와 역전파는 서로 의존합니다.
특히, 순전파의 경우,
저희는 의존성 방향으로 계산 그래프를 순회하며
그 경로상의 모든 변수를 계산합니다.
이들은 이후 그래프의 계산 순서가 뒤집히는
역전파에서 사용됩니다.

앞서 언급한 단순한 네트워크를 예시로 들어 봅시다.
한편으로,
순전파 동안 정규화 항 :eqref:`eq_forward-s`을
계산하는 것은
모델 파라미터 $\mathbf{W}^{(1)}$과 $\mathbf{W}^{(2)}$의
현재 값에 의존합니다.
이들은 가장 최근 반복에서 역전파에 따라 최적화 알고리즘이 제공합니다.
다른 한편으로,
역전파 동안 파라미터에 대한
기울기 계산 :eqref:`eq_backprop-J-h`은
은닉층 출력 $\mathbf{h}$의 현재 값에 의존하며,
이는 순전파에 의해 주어집니다.


따라서 신경망을 훈련할 때, 모델 파라미터가 초기화된 후에는,
저희는 순전파와 역전파를 번갈아 수행하면서,
역전파가 제공하는 기울기를 사용해 모델 파라미터를 업데이트합니다.
역전파는 중복 계산을 피하기 위해 순전파에서 저장된 중간 값을 재사용한다는 점에 유의하세요.
그 결과 중 하나로, 저희는
역전파가 완료될 때까지 중간 값을 유지해야 합니다.
이것이 또한 훈련이
단순한 예측보다 훨씬 더 많은 메모리를 필요로 하는 이유 중 하나입니다.
게다가, 이러한 중간 값의 크기는 대략
네트워크 층의 수와 배치 크기에 비례합니다.
따라서,
더 큰 배치 크기를 사용해 더 깊은 네트워크를 훈련하는 것은
*메모리 부족* 오류로 더 쉽게 이어집니다.


## 요약

순전파는 신경망에 의해 정의된 계산 그래프 내의 중간 변수를 순차적으로 계산하고 저장합니다. 이는 입력층에서 출력층으로 진행됩니다.
역전파는 신경망 내의 중간 변수와 파라미터의 기울기를 역순으로 순차적으로 계산하고 저장합니다.
딥러닝 모델을 훈련할 때, 순전파와 역전파는 상호 의존적이며,
훈련은 예측보다 훨씬 더 많은 메모리를 필요로 합니다.


## 연습문제

1. 어떤 스칼라 함수 $f$에 대한 입력 $\mathbf{X}$가 $n \times m$ 행렬이라고 가정합시다. $\mathbf{X}$에 대한 $f$의 기울기의 차원은 무엇인가요?
1. 이 절에서 설명한 모델의 은닉층에 편향을 추가하세요 ((정규화 항에 편향을 포함할 필요는 없습니다)).
    1. 해당 계산 그래프를 그리세요.
    1. 순전파와 역전파 방정식을 유도하세요.
1. 이 절에서 설명한 모델에서 훈련과 예측에 대한 메모리 사용량을 계산하세요.
1. 2계 도함수를 계산하고 싶다고 가정합시다. 계산 그래프에 어떤 일이 일어나나요? 계산이 얼마나 오래 걸릴 것으로 예상하나요?
1. 계산 그래프가 GPU에 비해 너무 크다고 가정합시다.
    1. 이를 하나 이상의 GPU에 걸쳐 분할할 수 있나요?
    1. 더 작은 미니배치로 훈련하는 것에 비해 장단점은 무엇인가요?

[Discussions](https://discuss.d2l.ai/t/102)
