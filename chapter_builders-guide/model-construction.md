```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 층과 모듈
:label:`sec_model_construction`

저희가 신경망을 처음 소개했을 때는,
출력이 하나뿐인 선형 모델에 초점을 맞췄습니다.
여기서 모델 전체는 단일 뉴런 하나로만 구성됩니다.
단일 뉴런은
(i) 어떤 입력 집합을 받고,
(ii) 그에 대응하는 스칼라 출력을 만들어 내며,
(iii) 관심 있는 어떤 목적 함수를 최적화하기 위해
업데이트할 수 있는 관련 파라미터의 집합을 가진다는 점에 유의하세요.
그다음에 여러 출력을 갖는 신경망을 생각하기 시작하면서,
저희는 벡터화된 산술을 활용해
한 층 전체의 뉴런을 특성화했습니다.
개별 뉴런과 마찬가지로,
층은 (i) 입력 집합을 받고,
(ii) 그에 대응하는 출력을 만들어 내며,
(iii) 조정 가능한 파라미터 집합으로 기술됩니다.
저희가 소프트맥스 회귀를 다룰 때에는,
하나의 층이 그 자체로 모델이었습니다.
그러나 그 뒤에 MLP를 소개할 때에도,
여전히 이 모델은 동일한 기본 구조를 유지한다고
생각할 수 있었습니다.

흥미롭게도, MLP의 경우 전체 모델과 그것을 구성하는 층들이
모두 이 구조를 공유합니다.
전체 모델은 원시 입력(특성)을 받아,
출력(예측)을 만들어 내고,
파라미터(모든 구성 층의 파라미터를 합친 것)를 가집니다.
마찬가지로 각각의 개별 층도 입력
(이전 층이 공급한 것)을 받아들이고,
출력(다음 층의 입력)을 만들어 내며,
다음 층으로부터 거꾸로 흘러오는 신호에 따라
업데이트되는 조정 가능한 파라미터 집합을 가집니다.


뉴런, 층, 모델이 저희가 할 일을 처리하기에 충분한
추상화를 제공한다고 생각하실 수도 있지만,
실제로는 개별 층보다는 크지만 전체 모델보다는 작은
구성 요소를 다루는 것이
편리한 경우가 흔하다는 것이 드러납니다.
예를 들어, 컴퓨터 비전에서 엄청난 인기를 얻고 있는
ResNet-152 아키텍처는 수백 개의 층을 가지고 있습니다.
이러한 층들은 *층들의 그룹*이 반복되는 패턴으로 구성되어 있습니다. 이러한 신경망을 한 번에 한 층씩 구현하면 지루해질 수 있습니다.
이러한 우려는 단지 가정에 그치지 않습니다(이러한
설계 패턴은 실제로도 흔합니다).
앞서 언급한 ResNet 아키텍처는
2015년 ImageNet과 COCO 컴퓨터 비전 대회에서
인식과 검출 양쪽 모두를 석권했으며 :cite:`He.Zhang.Ren.ea.2016`
여전히 많은 비전 과제에서 우선적으로 선택되는 아키텍처로 남아 있습니다.
다양한 반복 패턴으로 층이 배치된 유사한 아키텍처들이
이제는 자연어 처리와 음성을 포함한 다른 분야에서도
어디서나 볼 수 있게 되었습니다.

이러한 복잡한 신경망을 구현하기 위해
저희는 신경망 *모듈*이라는 개념을 도입합니다.
하나의 모듈은 단일 층을 기술할 수도 있고,
여러 층으로 구성된 구성 요소를 기술할 수도 있으며,
모델 전체를 기술할 수도 있습니다!
모듈 추상화를 다루는 데서 얻을 수 있는 한 가지 이점은
이들이 더 큰 산출물로, 흔히 재귀적으로
조합될 수 있다는 점입니다. 이는 :numref:`fig_blocks`에 나타나 있습니다. 임의의 복잡도를 갖는 모듈을 필요에 따라 생성하는
코드를 정의함으로써, 저희는 놀라울 만큼 간결한 코드를 작성하면서도
여전히 복잡한 신경망을 구현할 수 있습니다.

![여러 층이 모듈로 결합되어, 더 큰 모델의 반복 패턴을 형성한다.](../img/blocks.svg)
:label:`fig_blocks`


프로그래밍 관점에서 보면, 모듈은 *클래스*로 표현됩니다.
이 클래스의 모든 하위 클래스는 입력을 출력으로 변환하는
순전파 메서드를 정의해야 하고,
필요한 파라미터를 저장해야 합니다.
일부 모듈은 어떠한 파라미터도 필요로 하지 않는다는 점에 유의하세요.
마지막으로 모듈은 그래디언트를 계산할 목적으로
역전파 메서드도 갖춰야 합니다.
다행히 저희가 직접 모듈을 정의할 때에는,
자동 미분(:numref:`sec_autograd`에서 소개됨)이 제공하는
이면의 마법 덕분에 파라미터와 순전파 메서드만
신경 쓰면 됩니다.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

[**먼저 저희가 MLP를 구현할 때 사용했던
코드를 다시 살펴보겠습니다**]
(:numref:`sec_mlp`).
다음 코드는 256개 단위와 ReLU 활성화를 갖는
완전 연결 은닉 층 하나에 이어,
10개 단위(활성화 함수 없음)를 갖는 완전 연결 출력 층으로 구성된
신경망을 생성합니다.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

# get_key is a d2l saved function returning jax.random.PRNGKey(random_seed)
X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

:begin_tab:`mxnet`
이 예제에서 저희는 `nn.Sequential`을 인스턴스화하고
반환된 객체를 `net` 변수에 할당하여 모델을 구성했습니다.
그다음에는 `add` 메서드를 반복해서 호출하여
실행되어야 할 순서대로 층들을 덧붙입니다.
요약하면, `nn.Sequential`은 Gluon에서 *모듈*을 표현하는 클래스인
`Block`의 특수한 종류를 정의합니다.
이 클래스는 구성 요소인 `Block`들의 순서 있는 목록을 유지합니다.
`add` 메서드는 단지 연속되는 각 `Block`을 목록에
추가하는 일을 돕는 역할을 합니다.
각 층은 `Dense` 클래스의 인스턴스이며,
`Dense` 클래스 자체가 `Block`의 하위 클래스라는 점에 유의하세요.
순전파(`forward`) 메서드도 놀라울 만큼 간단합니다.
즉, 목록 안의 각 `Block`을 차례로 이어 붙이면서,
각각의 출력을 다음 것의 입력으로 전달합니다.
지금까지 저희는 출력을 얻기 위해 `net(X)`라는
구성을 통해 모델을 호출해 왔다는 점에 유의하세요.
이는 사실 `net.forward(X)`의 축약 표현일 뿐이며,
`Block` 클래스의 `__call__` 메서드를 통해 이루어지는
근사한 파이썬 트릭입니다.
:end_tab:

:begin_tab:`pytorch`
이 예제에서 저희는 `nn.Sequential`을 인스턴스화하고,
실행되어야 할 순서대로 층들을 인자로 전달하여 모델을 구성했습니다.
요약하면, (**`nn.Sequential`은 PyTorch에서 모듈을 표현하는 클래스인
`Module`의 특수한 종류를 정의합니다**).
이 클래스는 구성 요소인 `Module`들의 순서 있는 목록을 유지합니다.
두 개의 완전 연결 층은 각각 `Linear` 클래스의 인스턴스이며,
`Linear` 클래스 자체가 `Module`의 하위 클래스라는 점에 유의하세요.
순전파(`forward`) 메서드도 놀라울 만큼 간단합니다.
즉, 목록 안의 각 모듈을 차례로 이어 붙이면서,
각각의 출력을 다음 것의 입력으로 전달합니다.
지금까지 저희는 출력을 얻기 위해 `net(X)`라는
구성을 통해 모델을 호출해 왔다는 점에 유의하세요.
이는 사실 `net.__call__(X)`의 축약 표현일 뿐입니다.
:end_tab:

:begin_tab:`tensorflow`
이 예제에서 저희는 `keras.models.Sequential`을 인스턴스화하고,
실행되어야 할 순서대로 층들을 인자로 전달하여 모델을 구성했습니다.
요약하면, `Sequential`은 Keras에서 모듈을 표현하는 클래스인
`keras.Model`의 특수한 종류를 정의합니다.
이 클래스는 구성 요소인 `Model`들의 순서 있는 목록을 유지합니다.
두 개의 완전 연결 층은 각각 `Dense` 클래스의 인스턴스이며,
`Dense` 클래스 자체가 `Model`의 하위 클래스라는 점에 유의하세요.
순전파(`call`) 메서드도 놀라울 만큼 간단합니다.
즉, 목록 안의 각 모듈을 차례로 이어 붙이면서,
각각의 출력을 다음 것의 입력으로 전달합니다.
지금까지 저희는 출력을 얻기 위해 `net(X)`라는
구성을 통해 모델을 호출해 왔다는 점에 유의하세요.
이는 사실 `net.call(X)`의 축약 표현일 뿐이며,
모듈 클래스의 `__call__` 메서드를 통해 이루어지는
근사한 파이썬 트릭입니다.
:end_tab:

## [**커스텀 모듈**]

모듈이 어떻게 동작하는지에 대한 직관을 기르는
가장 쉬운 방법은 직접 하나를 구현해 보는 것일 겁니다.
그렇게 하기 전에, 각 모듈이 제공해야 하는
기본 기능을 간단히 정리해 보겠습니다.


1. 입력 데이터를 순전파 메서드의 인자로 받아들입니다.
1. 순전파 메서드가 값을 반환하도록 하여 출력을 생성합니다. 출력은 입력과 다른 모양을 가질 수 있다는 점에 유의하세요. 예를 들어, 위에 있는 저희 모델의 첫 번째 완전 연결 층은 임의의 차원의 입력을 받지만 차원이 256인 출력을 반환합니다.
1. 입력에 대한 출력의 그래디언트를 계산하며, 이는 역전파 메서드를 통해 접근할 수 있습니다. 일반적으로 이는 자동으로 이루어집니다.
1. 순전파 계산을 실행하는 데 필요한 파라미터를 저장하고
   접근할 수 있도록 제공합니다.
1. 필요에 따라 모델 파라미터를 초기화합니다.


다음 코드 조각에서, 저희는 256개의 은닉 단위를 갖는
은닉 층 하나와 10차원 출력 층으로 구성된 MLP에 해당하는
모듈을 처음부터 직접 만들어 봅니다.
아래의 `MLP` 클래스는 모듈을 표현하는 클래스를 상속한다는 점에 유의하세요.
저희는 부모 클래스의 메서드에 크게 의존하면서,
직접 작성하는 것은 생성자(파이썬의 `__init__` 메서드)와 순전파 메서드뿐일 것입니다.

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # Call the constructor of the MLP parent class nn.Block to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # Call the constructor of the parent class tf.keras.Model to perform
        # the necessary initialization
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        # Define the layers
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

먼저 순전파 메서드에 초점을 맞춰 보겠습니다.
이 메서드는 `X`를 입력으로 받아,
활성화 함수가 적용된 은닉 표현을 계산하고,
그 로짓(logits)을 출력한다는 점에 유의하세요.
이 `MLP` 구현에서는 두 층 모두 인스턴스 변수입니다.
이것이 합당한 이유를 알아보려면,
두 개의 MLP, `net1`과 `net2`를 인스턴스화하여
서로 다른 데이터로 학습시키는 상황을 상상해 보세요.
자연스럽게 이 둘은 서로 다른 두 개의 학습된 모델을
표현하리라 기대하게 됩니다.

저희는 생성자에서 [**MLP의 층들을 인스턴스화**]하고
(**순전파 메서드가 호출될 때마다 이 층들을 다시 호출**)합니다.
몇 가지 주요 세부 사항에 유의하세요.
첫째, 저희가 사용자 정의한 `__init__` 메서드는
`super().__init__()`를 통해 부모 클래스의 `__init__` 메서드를 호출하며,
이를 통해 대부분의 모듈에 적용되는 보일러플레이트 코드를
다시 적어야 하는 수고를 덜어 줍니다.
그다음 두 개의 완전 연결 층을 인스턴스화하여,
`self.hidden`과 `self.out`에 할당합니다.
새로운 층을 구현하는 경우가 아니라면,
역전파 메서드나 파라미터 초기화에 대해서는 신경 쓸 필요가 없다는 점에 유의하세요.
이러한 메서드들은 시스템이 자동으로 생성해 줍니다.
이를 한번 실험해 보겠습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

```{.python .input}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

모듈 추상화의 핵심적인 미덕은 그 다재다능함입니다.
모듈을 하위 클래스화하여 (완전 연결 층 클래스 같은) 층,
(위의 `MLP` 클래스와 같은) 모델 전체,
혹은 중간 정도 복잡도의 다양한 구성 요소를 만들 수 있습니다.
저희는 합성곱 신경망을 다룰 때 등 앞으로 다룰 장들 전반에서
이 다재다능함을 활용할 것입니다.


## [**Sequential 모듈**]
:label:`subsec_model-construction-sequential`

이제 `Sequential` 클래스가 어떻게 동작하는지
좀 더 자세히 살펴볼 수 있습니다.
`Sequential`은 다른 모듈들을 줄줄이 이어 붙이도록
설계되었다는 점을 떠올려 보세요.
저희만의 단순화된 `MySequential`을 만들기 위해서는,
두 가지 핵심 메서드만 정의하면 됩니다.

1. 모듈들을 하나씩 목록에 덧붙이기 위한 메서드.
1. 모듈들이 덧붙여진 것과 동일한 순서로 모듈 체인을 통과해 입력을 전달하기 위한 순전파 메서드.

다음 `MySequential` 클래스는 기본 `Sequential` 클래스와
동일한 기능을 제공합니다.

```{.python .input}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume that
        # it has a unique name. We save it in the member variable _children of
        # the Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize method, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
%%tab jax
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` 메서드는 순서 있는 사전 `_children`에 하나의 블록을 추가합니다.
모든 Gluon `Block`이 `_children` 속성을 가지는 이유와
저희가 그냥 파이썬 리스트를 직접 정의하는 대신
이것을 사용한 이유가 궁금하실 것입니다.
요약하자면 `_children`의 주된 장점은,
블록의 파라미터 초기화 도중에 Gluon이
파라미터도 초기화되어야 하는 하위 블록을 찾기 위해
`_children` 사전 안을 살펴봐야 한다는 사실을 알고 있다는 점입니다.
:end_tab:

:begin_tab:`pytorch`
`__init__` 메서드에서, 저희는 `add_modules` 메서드를 호출하여
모든 모듈을 추가합니다. 이 모듈들은 나중에 `children` 메서드를 통해 접근할 수 있습니다.
이렇게 하면 시스템이 추가된 모듈들을 인식하고,
각 모듈의 파라미터를 적절히 초기화하게 됩니다.
:end_tab:

저희 `MySequential`의 순전파 메서드가 호출되면,
추가된 각 모듈이 추가된 순서대로 실행됩니다.
이제 저희의 `MySequential` 클래스를 사용해 MLP를
다시 구현할 수 있습니다.

```{.python .input}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

이러한 `MySequential` 사용은 저희가 앞서
`Sequential` 클래스에 대해 작성했던 코드
(:numref:`sec_mlp`에 기술된 것)와 동일하다는 점에 유의하세요.


## [**순전파 메서드 안에서 코드 실행하기**]

`Sequential` 클래스는 모델 구성을 쉽게 만들어,
저희가 자체 클래스를 정의하지 않고도 새로운 아키텍처를
조립할 수 있게 해 줍니다.
그러나 모든 아키텍처가 단순한 직렬 사슬은 아닙니다.
더 큰 유연성이 요구될 때에는, 저희만의 블록을
정의하고 싶을 것입니다.
예를 들어, 순전파 메서드 안에서 파이썬의 제어 흐름을
실행하고 싶을 수 있습니다.
나아가, 미리 정의된 신경망 층에만 의존하지 않고
임의의 수학적 연산을 수행하고 싶을 수도 있습니다.

지금까지 저희 신경망의 모든 연산은 신경망의 활성화와
파라미터에 대해 작용해 왔다는 점을 눈치채셨을 것입니다.
그러나 때로는 이전 층의 결과도 아니고
업데이트 가능한 파라미터도 아닌 항들을
포함시키고 싶을 수 있습니다.
이러한 것들을 *상수 파라미터*라고 부릅니다.
예를 들어 함수 $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$를
계산하는 층을 원한다고 해 보겠습니다.
여기서 $\mathbf{x}$는 입력, $\mathbf{w}$는 저희의 파라미터,
$c$는 최적화 도중에 업데이트되지 않는
어떤 지정된 상수입니다.
따라서 저희는 `FixedHiddenMLP` 클래스를 다음과 같이 구현합니다.

```{.python .input}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # Random weight parameters created with the get_constant method
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the relu and dot
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with tf.constant are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the relu and
        # matmul functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
%%tab jax
class FixedHiddenMLP(nn.Module):
    # Random weight parameters that will not compute gradients and
    # therefore keep constant during training
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

이 모델에서 저희는 가중치(`self.rand_weight`)가
인스턴스화 시 무작위로 초기화되고 그 이후로는 상수가 되는
은닉 층을 구현합니다.
이 가중치는 모델 파라미터가 아니며,
따라서 역전파에 의해 업데이트되지 않습니다.
그다음에 신경망은 이 "고정된" 층의 출력을
완전 연결 층에 통과시킵니다.

출력을 반환하기 전에, 저희 모델이 흔치 않은 일을 했다는 점에 유의하세요.
$\ell_1$ 노름이 $1$보다 크다는 조건을 검사하는 while 루프를 돌리며,
그 조건이 만족될 때까지 출력 벡터를 $2$로 나누었습니다.
마지막으로, `X`의 원소들의 합을 반환했습니다.
저희가 아는 한, 어떤 표준 신경망도
이러한 연산을 수행하지 않습니다.
이 특정한 연산이 어떠한 실제 과제에서도 유용하지 않을 수
있다는 점에 유의하세요.
저희의 요지는 단지 신경망 계산의 흐름에 임의의 코드를
어떻게 통합하는지를 보여 드리려는 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

```{.python .input}
%%tab jax
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

[**모듈들을 조립하는 다양한 방식을 자유롭게 섞고 짝지을 수 있습니다.**]
다음 예제에서, 저희는 모듈들을
몇 가지 창의적인 방식으로 중첩합니다.

```{.python .input}
%%tab mxnet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab jax
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))


chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
chimera.apply(params, X)
```

## 요약

개별 층은 모듈이 될 수 있습니다.
여러 층이 모여 모듈을 구성할 수 있습니다.
여러 모듈이 모여 모듈을 구성할 수 있습니다.

모듈은 코드를 담을 수 있습니다.
모듈은 파라미터 초기화와 역전파를 포함한 많은 관리 작업을 알아서 처리해 줍니다.
층과 모듈의 순차적 연결은 `Sequential` 모듈이 처리합니다.


## 연습문제

1. `MySequential`을 변경하여 모듈을 파이썬 리스트에 저장하도록 하면 어떤 종류의 문제가 발생할까요?
1. 두 개의 모듈, 예를 들어 `net1`과 `net2`를 인자로 받아 순전파에서 두 신경망의 출력을 이어붙인 결과를 반환하는 모듈을 구현해 보세요. 이를 *병렬 모듈*이라고도 부릅니다.
1. 같은 신경망의 여러 인스턴스를 이어 붙이고 싶다고 가정해 보세요. 같은 모듈의 여러 인스턴스를 생성하는 팩토리 함수를 구현하고, 이를 사용해 더 큰 신경망을 만들어 보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17989)
:end_tab:
