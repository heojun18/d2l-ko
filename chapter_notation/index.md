# 표기법
:label:`chap_notation`

이 책 전반에 걸쳐 저희는
다음의 표기 규약을 따릅니다.
이러한 기호 중 일부는 자리 표시자(placeholder)이고,
다른 일부는 특정한 대상을 가리킨다는 점에 유의하세요.
대략적인 일반 규칙으로,
부정관사 "a"는 흔히 해당 기호가 자리 표시자이며
유사한 형식의 기호들이
같은 종류의 다른 대상을 나타낼 수 있음을
의미합니다.
예를 들어 "$x$: a scalar"는
소문자가 일반적으로 스칼라 값을
나타낸다는 뜻이지만,
"$\mathbb{Z}$: the set of integers"는
구체적으로 기호 $\mathbb{Z}$를 가리킵니다.



## 수치적 대상

* $x$: 스칼라
* $\mathbf{x}$: 벡터
* $\mathbf{X}$: 행렬
* $\mathsf{X}$: 일반 텐서
* $\mathbf{I}$: 단위 행렬(주어진 어떤 차원의), 즉 모든 대각 성분이 $1$이고 비대각 성분이 모두 $0$인 정방 행렬
* $x_i$, $[\mathbf{x}]_i$: 벡터 $\mathbf{x}$의 $i^\textrm{th}$ 원소
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: 행렬 $\mathbf{X}$의 $i$행 $j$열에 위치한 원소.



## 집합론


* $\mathcal{X}$: 집합
* $\mathbb{Z}$: 정수의 집합
* $\mathbb{Z}^+$: 양의 정수의 집합
* $\mathbb{R}$: 실수의 집합
* $\mathbb{R}^n$: $n$차원 실수 벡터의 집합
* $\mathbb{R}^{a\times b}$: $a$개의 행과 $b$개의 열로 이루어진 실수 행렬의 집합
* $|\mathcal{X}|$: 집합 $\mathcal{X}$의 카디널리티(원소 개수)
* $\mathcal{A}\cup\mathcal{B}$: 집합 $\mathcal{A}$와 $\mathcal{B}$의 합집합
* $\mathcal{A}\cap\mathcal{B}$: 집합 $\mathcal{A}$와 $\mathcal{B}$의 교집합
* $\mathcal{A}\setminus\mathcal{B}$: $\mathcal{A}$에서 $\mathcal{B}$를 뺀 차집합($\mathcal{B}$에 속하지 않는 $\mathcal{A}$의 원소들만 포함)



## 함수와 연산자


* $f(\cdot)$: 함수
* $\log(\cdot)$: 자연로그(밑이 $e$)
* $\log_2(\cdot)$: 밑이 $2$인 로그
* $\exp(\cdot)$: 지수 함수
* $\mathbf{1}(\cdot)$: 지시 함수, 불리언 인자가 참이면 $1$로, 그렇지 않으면 $0$으로 평가됨
* $\mathbf{1}_{\mathcal{X}}(z)$: 집합 소속 지시 함수, 원소 $z$가 집합 $\mathcal{X}$에 속하면 $1$로, 그렇지 않으면 $0$으로 평가됨
* $\mathbf{(\cdot)}^\top$: 벡터 또는 행렬의 전치
* $\mathbf{X}^{-1}$: 행렬 $\mathbf{X}$의 역행렬
* $\odot$: 아다마르(원소별) 곱
* $[\cdot, \cdot]$: 연결(concatenation)
* $\|\cdot\|_p$: $\ell_p$ 노름
* $\|\cdot\|$: $\ell_2$ 노름
* $\langle \mathbf{x}, \mathbf{y} \rangle$: 벡터 $\mathbf{x}$와 $\mathbf{y}$의 내적(점곱)
* $\sum$: 원소들의 집합에 대한 합
* $\prod$: 원소들의 집합에 대한 곱
* $\stackrel{\textrm{def}}{=}$: 좌변 기호의 정의로서 단언되는 등식



## 미적분

* $\frac{dy}{dx}$: $x$에 대한 $y$의 도함수
* $\frac{\partial y}{\partial x}$: $x$에 대한 $y$의 편도함수
* $\nabla_{\mathbf{x}} y$: $\mathbf{x}$에 대한 $y$의 그레이디언트
* $\int_a^b f(x) \;dx$: $x$에 대한 $f$의 $a$부터 $b$까지의 정적분
* $\int f(x) \;dx$: $x$에 대한 $f$의 부정적분



## 확률과 정보 이론

* $X$: 확률 변수
* $P$: 확률 분포
* $X \sim P$: 확률 변수 $X$가 분포 $P$를 따름
* $P(X=x)$: 확률 변수 $X$가 값 $x$를 취하는 사건에 부여된 확률
* $P(X \mid Y)$: $Y$가 주어졌을 때 $X$의 조건부 확률 분포
* $p(\cdot)$: 분포 $P$와 연관된 확률 밀도 함수(PDF)
* ${E}[X]$: 확률 변수 $X$의 기댓값
* $X \perp Y$: 확률 변수 $X$와 $Y$가 독립
* $X \perp Y \mid Z$: 확률 변수 $X$와 $Y$가 $Z$가 주어졌을 때 조건부 독립
* $\sigma_X$: 확률 변수 $X$의 표준편차
* $\textrm{Var}(X)$: 확률 변수 $X$의 분산, $\sigma^2_X$와 같음
* $\textrm{Cov}(X, Y)$: 확률 변수 $X$와 $Y$의 공분산
* $\rho(X, Y)$: $X$와 $Y$ 사이의 피어슨 상관 계수, $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$와 같음
* $H(X)$: 확률 변수 $X$의 엔트로피
* $D_{\textrm{KL}}(P\|Q)$: 분포 $Q$로부터 분포 $P$까지의 KL 발산(또는 상대 엔트로피)



[토론](https://discuss.d2l.ai/t/25)
