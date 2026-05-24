```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 미적분
:label:`sec_calculus`

오랜 기간 동안, 원의 넓이를
어떻게 계산할지는 미스터리로 남아 있었습니다.
그러던 중 고대 그리스의 수학자 아르키메데스가
원의 내부에 꼭짓점 수를 점점 늘려가는
일련의 다각형을 내접시키는 영리한 아이디어를
떠올렸습니다
(:numref:`fig_circle_area`).
꼭짓점이 $n$개인 다각형에 대해,
저희는 $n$개의 삼각형을 얻습니다.
원을 더 잘게 분할할수록 각 삼각형의 높이는
반지름 $r$에 가까워집니다.
동시에, 꼭짓점 수가 많아지면 호와 현의 비율이 1에 가까워지므로
밑변은 $2 \pi r/n$에 가까워집니다.
따라서, 다각형의 넓이는
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$에 가까워집니다.

![극한 절차로 원의 넓이를 구하기.](../img/polygon-circle.svg)
:label:`fig_circle_area`

이러한 극한 절차는 *미분*과 *적분* 모두의
뿌리에 있습니다.
전자는 함수의 인수를 조작함으로써
함수의 값을 어떻게 증가시키거나
감소시킬 수 있는지를 알려줄 수 있습니다.
이는 손실 함수를 감소시키기 위해 파라미터를 반복적으로 업데이트하는
딥러닝에서 마주하는 *최적화 문제*에서 유용하게 사용됩니다.
최적화는 모델을 훈련 데이터에 어떻게 맞출지를 다루며,
미적분은 이를 위한 핵심 선수 지식입니다.
그러나 저희의 궁극적인 목표는
*이전에 본 적 없는* 데이터에서 좋은 성능을 내는 것임을 잊지 마십시오.
그 문제는 *일반화*라고 불리며
다른 장들에서 핵심적으로 다룰 주제입니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## 도함수와 미분

간단히 말해, *도함수*는 함수의 인수가 변할 때
함수가 변하는 비율입니다.
도함수는 각 파라미터를 무한히 작은 양만큼
*증가*시키거나 *감소*시킬 경우
손실 함수가 얼마나 빠르게 증가하거나
감소하는지를 알려줄 수 있습니다.
형식적으로, 스칼라를 스칼라로 매핑하는
함수 $f: \mathbb{R} \rightarrow \mathbb{R}$에 대해,
[**점 $x$에서 $f$의 *도함수*는 다음과 같이 정의됩니다**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

우변의 항은 *극한*이라고 불리며
지정된 변수가 특정 값에 가까워질 때
식의 값에 어떤 일이 일어나는지를
알려줍니다.
이 극한은 그 크기를 0으로 줄여나갈 때
섭동 $h$와 함수 값의 변화
$f(x + h) - f(x)$ 사이의 비율이
어디로 수렴하는지를 알려줍니다.

$f'(x)$가 존재할 때, $f$는 $x$에서
*미분 가능*하다고 합니다.
그리고 집합, 예를 들어 구간 $[a,b]$의 모든 $x$에 대해
$f'(x)$가 존재할 때,
저희는 $f$가 이 집합 위에서 미분 가능하다고 합니다.
모든 함수가 미분 가능한 것은 아니며,
정확도나 ROC 곡선 아래 면적(AUC)처럼
저희가 최적화하고자 하는 많은 함수들이 이에 해당합니다.
그러나 손실의 도함수를 계산하는 것은
심층 신경망을 훈련시키는 거의 모든 알고리즘에서
중요한 단계이기 때문에,
저희는 종종 그 대신 미분 가능한 *대리* 함수를 최적화합니다.


도함수
$f'(x)$
는 $x$에 대한 $f(x)$의 *순간적인*
변화율로 해석할 수 있습니다.
예제를 통해 직관을 키워봅시다.
(**$u = f(x) = 3x^2-4x$를 정의합니다.**)

```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x=1$로 설정하면, $\frac{f(x+h) - f(x)}{h}$가**] (**$h$가 $0$에 가까워짐에 따라
$2$에 가까워지는 것을 볼 수 있습니다.**)
이 실험은 수학적 증명의 엄밀함은 부족하지만,
실제로 $f'(1) = 2$임을 빠르게 확인할 수 있습니다.

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

도함수에 대한 여러 가지 동등한 표기법이 있습니다.
$y = f(x)$가 주어졌을 때, 다음의 표현들은 모두 동등합니다.

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

여기서 기호 $\frac{d}{dx}$와 $D$는 *미분 연산자*입니다.
아래에서, 몇 가지 일반적인 함수의 도함수를 제시합니다.

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

미분 가능한 함수들로부터 합성된 함수는
종종 그 자체로 미분 가능합니다.
다음 규칙들은 임의의 미분 가능한 함수
$f$와 $g$, 그리고 상수 $C$의
합성을 다루는 데 유용합니다.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{상수배 법칙} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{합 법칙} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{곱 법칙} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{몫 법칙} \end{aligned}$$

이를 이용하여, 저희는 규칙들을 적용해
$3 x^2 - 4x$의 도함수를 다음과 같이 찾을 수 있습니다.

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

$x = 1$을 대입하면, 실제로 이 위치에서
도함수가 $2$와 같음을 보여줍니다.
도함수는 특정 위치에서 함수의 *기울기*를
알려준다는 점에 유의하십시오.

## 시각화 유틸리티

[**저희는 `matplotlib` 라이브러리를 사용하여 함수의 기울기를 시각화할 수 있습니다**].
몇 가지 함수를 정의해야 합니다.
이름에서 알 수 있듯이, `use_svg_display`는
더 선명한 이미지를 위해 SVG 형식으로 그래픽을
출력하도록 `matplotlib`에 알려줍니다.
주석 `#@save`는 함수, 클래스, 또는 다른 코드 블록을
`d2l` 패키지에 저장할 수 있게 해 주는 특수한 수정자로,
나중에 코드를 반복하지 않고
예를 들어 `d2l.use_svg_display()`를 통해
호출할 수 있게 합니다.

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

편리하게도, `set_figsize`로 그림 크기를 설정할 수 있습니다.
`from matplotlib import pyplot as plt` 임포트 문이
`d2l` 패키지에서 `#@save`로 표시되었기 때문에, `d2l.plt`를 호출할 수 있습니다.

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

`set_axes` 함수는 레이블, 범위, 스케일을 포함한
속성들을 축과 연결시킬 수 있습니다.

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

이 세 함수들을 사용하여, 여러 곡선을 겹쳐 그리는
`plot` 함수를 정의할 수 있습니다.
여기 있는 코드의 상당 부분은
입력의 크기와 모양이 일치하도록 하는 것입니다.

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

이제 [**함수 $u = f(x)$와 $x=1$에서의 접선 $y = 2x - 3$을 그릴**] 수 있으며,
계수 $2$는 접선의 기울기입니다.

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 편미분과 그래디언트
:label:`subsec_calculus-grad`

지금까지, 저희는 단 하나의 변수를 가진 함수를
미분해 왔습니다.
딥러닝에서는, *많은* 변수를 가진 함수도
다뤄야 합니다.
이러한 *다변수* 함수에 적용되는
도함수 개념을 간략히 소개합니다.


$y = f(x_1, x_2, \ldots, x_n)$를 $n$개의 변수를 가진 함수라고 합시다.
$i$번째 파라미터 $x_i$에 대한 $y$의
*편미분*은 다음과 같습니다.

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


$\frac{\partial y}{\partial x_i}$를 계산하기 위해,
$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$을 상수로 취급하고
$x_i$에 대한 $y$의 도함수를 계산하면 됩니다.
편미분에 대한 다음의 표기 규약들은
모두 일반적이며 모두 같은 것을 의미합니다.

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

다변수 함수의 모든 변수에 대한
편미분들을 연결하여
함수의 *그래디언트*라고 불리는
벡터를 얻을 수 있습니다.
함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의
입력이 $n$차원 벡터
$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$이고
출력이 스칼라라고 가정합니다.
$\mathbf{x}$에 대한 함수 $f$의 그래디언트는
$n$개의 편미분의 벡터입니다.

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

모호함이 없을 때,
$\nabla_{\mathbf{x}} f(\mathbf{x})$는
일반적으로 $\nabla f(\mathbf{x})$로
대체됩니다.
다음 규칙들은 다변수 함수를 미분하는 데
유용하게 사용됩니다.

* 모든 $\mathbf{A} \in \mathbb{R}^{m \times n}$에 대해 $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$이고 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$입니다.
* 정사각 행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$에 대해 $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$이고 특히
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$입니다.

마찬가지로, 임의의 행렬 $\mathbf{X}$에 대해,
$\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$가 성립합니다.



## 연쇄 법칙

딥러닝에서, 관심 있는 그래디언트들은
종종 계산하기 어렵습니다.
저희가 (함수의 (함수의...))
깊게 중첩된 함수들을 다루기 때문입니다.
다행히도, *연쇄 법칙*이 이를 처리해 줍니다.
단일 변수의 함수로 돌아가서,
$y = f(g(x))$이고
바탕이 되는 함수
$y=f(u)$와 $u=g(x)$가
모두 미분 가능하다고 가정합시다.
연쇄 법칙에 의하면


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$



다변수 함수로 돌아가서,
$y = f(\mathbf{u})$가 변수 
$u_1, u_2, \ldots, u_m$을 가지고,
각 $u_i = g_i(\mathbf{x})$가
변수 $x_1, x_2, \ldots, x_n$을 가진다고 가정합니다.
즉, $\mathbf{u} = g(\mathbf{x})$입니다.
그러면 연쇄 법칙은 다음과 같이 말합니다.

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ and so } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

여기서 $\mathbf{A} \in \mathbb{R}^{n \times m}$은
벡터 $\mathbf{x}$에 대한 벡터 $\mathbf{u}$의 도함수를
포함하는 *행렬*입니다.
따라서, 그래디언트를 평가하려면
벡터-행렬 곱을 계산해야 합니다.
이것이 선형대수가
딥러닝 시스템을 구축하는 데 있어
이토록 핵심적인 구성 요소인 주요 이유 중 하나입니다.



## 논의

저희는 깊은 주제의 표면만 살짝 다뤘을 뿐이지만,
이미 몇 가지 개념들이 부각됩니다.
첫째, 미분에 대한 합성 규칙들은
일상적으로 적용될 수 있으며, 이를 통해
저희는 그래디언트를 *자동으로* 계산할 수 있습니다.
이 작업에는 창의성이 필요하지 않으므로
저희는 인지적 능력을 다른 곳에 집중시킬 수 있습니다.
둘째, 벡터값 함수의 도함수를 계산하려면
출력에서 입력으로 변수의 의존성 그래프를 추적하면서
행렬들을 곱해야 합니다.
특히, 함수를 평가할 때는 이 그래프가
*순방향*으로 순회되고,
그래디언트를 계산할 때는
*역방향*으로 순회됩니다.
이후의 장들에서 연쇄 법칙을 적용하기 위한 계산 절차인
역전파를 형식적으로 소개할 것입니다.

최적화의 관점에서, 그래디언트는 손실을 낮추기 위해
모델의 파라미터를 어떻게 이동시켜야 하는지를
결정할 수 있게 해 주며,
이 책에서 사용되는 최적화 알고리즘의 각 단계는
그래디언트를 계산해야 할 것입니다.

## 연습문제

1. 지금까지 저희는 도함수의 규칙들을 당연한 것으로 받아들였습니다.
   정의와 극한을 사용하여 (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$, (iv) $f(x) = \log x$에 대한
   성질들을 증명해 보세요.
1. 같은 맥락에서, 곱, 합, 몫 법칙을 기본 원리로부터 증명하세요.
1. 상수배 법칙이 곱 법칙의 특수한 경우로 따른다는 것을 증명하세요.
1. $f(x) = x^x$의 도함수를 계산하세요.
1. 어떤 $x$에 대해 $f'(x) = 0$이라는 것은 무엇을 의미합니까?
   이것이 성립할 수 있는 함수 $f$와
   위치 $x$의 예를 들어주세요.
1. 함수 $y = f(x) = x^3 - \frac{1}{x}$를 그리고
   $x = 1$에서의 접선을 그리세요.
1. 함수
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$의 그래디언트를 구하세요.
1. 함수
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$의 그래디언트는 무엇입니까? $\mathbf{x} = \mathbf{0}$에서는 어떤 일이 발생합니까?
1. $u = f(x, y, z)$이고 $x = x(a, b)$, $y = y(a, b)$, $z = z(a, b)$인 경우에 대한
   연쇄 법칙을 적어 보실 수 있나요?
1. 가역인 함수 $f(x)$가 주어졌을 때,
   그 역함수 $f^{-1}(x)$의 도함수를 계산하세요.
   여기서 $f^{-1}(f(x)) = x$이고 반대로 $f(f^{-1}(y)) = y$입니다.
   힌트: 유도 과정에서 이러한 성질들을 사용하세요.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/197)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17969)
:end_tab:
