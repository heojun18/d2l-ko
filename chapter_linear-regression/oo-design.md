```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 구현을 위한 객체 지향 설계
:label:`sec_oo-design`

선형 회귀에 대한 소개에서 저희는
데이터, 모델, 손실 함수, 최적화 알고리즘을 포함한
다양한 구성 요소들을 살펴보았습니다.
실제로 선형 회귀는 가장 단순한 머신러닝 모델 중 하나입니다.
그러나 이를 훈련하는 데에는
이 책의 다른 모델들이 필요로 하는
많은 동일한 구성 요소를 사용합니다.
따라서 구현 세부 사항으로 뛰어들기 전에
저희가 전반에 걸쳐 사용할 일부 API를
설계해 두는 것이 가치 있습니다.
딥러닝의 구성 요소를 객체로 다룸으로써,
이 객체들과 그 상호작용에 대한 클래스를
정의하는 것에서 시작할 수 있습니다.
구현을 위한 이 객체 지향 설계는
설명을 크게 간소화할 것이며,
여러분의 프로젝트에서도 사용하고 싶어질지도 모릅니다.


[PyTorch Lightning](https://www.pytorchlightning.ai/) 같은 오픈 소스 라이브러리에서 영감을 받아,
고수준에서 저희는 세 가지 클래스를 갖고자 합니다.
(i) `Module`은 모델, 손실, 최적화 메서드를 포함하고,
(ii) `DataModule`은 훈련과 검증을 위한 데이터 로더를 제공하며,
(iii) 두 클래스는 `Trainer` 클래스를 사용해 결합되어,
저희가 다양한 하드웨어 플랫폼에서 모델을 훈련할 수 있게 해 줍니다.
이 책의 대부분의 코드는 `Module`과 `DataModule`을 차용합니다. `Trainer` 클래스는 GPU, CPU, 병렬 훈련, 최적화 알고리즘을 논의할 때에만 다룰 것입니다.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## 유틸리티
:label:`oo-design-utilities`

주피터 노트북에서 객체 지향 프로그래밍을 단순화하기 위해 몇 가지 유틸리티가 필요합니다. 한 가지 어려움은 클래스 정의가 상당히 긴 코드 블록이 되는 경향이 있다는 점입니다. 노트북의 가독성은 설명 사이사이에 짧은 코드 조각을 요구하는데, 이는 Python 라이브러리에서 일반적인 프로그래밍 스타일과 양립할 수 없는 요건입니다. 첫 번째 유틸리티 함수는 클래스가 생성된 *후에* 함수를 클래스의 메서드로 등록할 수 있게 해 줍니다. 실제로 클래스의 인스턴스를 생성한 *후에도* 그렇게 할 수 있습니다! 이를 통해 클래스의 구현을 여러 코드 블록으로 나눌 수 있습니다.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

이를 어떻게 사용하는지 빠르게 살펴봅시다. 저희는 `do` 메서드를 가진 클래스 `A`를 구현할 계획입니다. `A`와 `do`에 대한 코드를 같은 코드 블록에 두는 대신, 먼저 클래스 `A`를 선언하고 인스턴스 `a`를 생성할 수 있습니다.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

다음으로 보통 하듯이 메서드 `do`를 정의하되, 클래스 `A`의 범위 안에서 정의하지 않습니다. 대신 이 메서드를 클래스 `A`를 인수로 하는 `add_to_class`로 데코레이트합니다. 이렇게 하면 이 메서드는 마치 `A`의 정의의 일부로 포함되었던 것처럼 `A`의 멤버 변수에 접근할 수 있습니다. 인스턴스 `a`에 대해 호출하면 어떤 일이 일어나는지 봅시다.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

두 번째는 클래스의 `__init__` 메서드의 모든 인수를 클래스 속성으로 저장하는 유틸리티 클래스입니다. 이를 통해 추가 코드 없이 암묵적으로 생성자 호출 시그니처를 확장할 수 있습니다.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

이 구현은 :numref:`sec_utils`로 미룹니다. 이를 사용하기 위해, `HyperParameters`를 상속받고 `__init__` 메서드에서 `save_hyperparameters`를 호출하는 클래스를 정의합니다.

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

마지막 유틸리티는 실험이 진행되는 동안 실험 진행 상황을 인터랙티브하게 그릴 수 있게 해 줍니다. 훨씬 더 강력한 (그리고 복잡한) [TensorBoard](https://www.tensorflow.org/tensorboard)에 경의를 표하여 이를 `ProgressBoard`라고 이름 지었습니다. 구현은 :numref:`sec_utils`로 미룹니다. 지금은 그저 실제 작동하는 모습을 봅시다.

`draw` 메서드는 범례에 지정된 `label`로 그림에 점 `(x, y)`를 그립니다. 선택적인 `every_n`은 그림에 $1/n` 점만 표시하여 선을 부드럽게 만듭니다. 이들의 값은 원래 그림의 $n$개 이웃 점들로부터 평균을 냅니다.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

다음 예제에서는 다른 부드러움으로 `sin`과 `cos`를 그립니다. 이 코드 블록을 실행하면, 선들이 애니메이션으로 자라나는 것을 보게 될 것입니다.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## 모델
:label:`subsec_oo-design-models`

`Module` 클래스는 저희가 구현할 모든 모델의 기본 클래스입니다. 최소한 세 개의 메서드가 필요합니다. 첫째, `__init__`은 학습 가능한 매개변수를 저장합니다. `training_step` 메서드는 데이터 배치를 받아 손실 값을 반환합니다. 마지막으로 `configure_optimizers`는 학습 가능한 매개변수를 갱신하는 데 사용되는 최적화 메서드(혹은 그 목록)를 반환합니다. 선택적으로 평가 척도를 보고하기 위해 `validation_step`을 정의할 수 있습니다.
때로는 재사용성을 높이기 위해 출력을 계산하는 코드를 별도의 `forward` 메서드에 둡니다.

:begin_tab:`jax`
Python 3.7에서 [dataclasses](https://docs.python.org/3/library/dataclasses.html)가 도입되면서,
`@dataclass`로 데코레이트된 클래스는 `__init__`과 `__repr__` 같은
매직 메서드를 자동으로 추가합니다. 멤버 변수는 타입 주석을 사용해 정의됩니다.
모든 Flax 모듈은 Python 3.7 dataclass입니다.
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # No need for save_hyperparam when using Python dataclass
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Use default_factory to make sure new plots are generated on each run
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
        # and built-in __call__ magic methods for forward pass. Adding here
        # for consistency
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """To be defined later in :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
`Module`은 Gluon의 신경망 기본 클래스인 `nn.Block`의 하위 클래스라는 것을 알 수 있습니다.
이는 신경망을 다루기 위한 편리한 기능을 제공합니다. 예를 들어 `forward(self, X)` 같은 `forward` 메서드를 정의하면, 인스턴스 `a`에 대해 `a(X)`로 이 메서드를 호출할 수 있습니다. 이는 내장된 `__call__` 메서드에서 `forward` 메서드를 호출하기 때문에 작동합니다. `nn.Block`에 대한 더 자세한 내용과 예제는 :numref:`sec_model_construction`에서 찾을 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
`Module`은 PyTorch의 신경망 기본 클래스인 `nn.Module`의 하위 클래스라는 것을 알 수 있습니다.
이는 신경망을 다루기 위한 편리한 기능을 제공합니다. 예를 들어 `forward(self, X)` 같은 `forward` 메서드를 정의하면, 인스턴스 `a`에 대해 `a(X)`로 이 메서드를 호출할 수 있습니다. 이는 내장된 `__call__` 메서드에서 `forward` 메서드를 호출하기 때문에 작동합니다. `nn.Module`에 대한 더 자세한 내용과 예제는 :numref:`sec_model_construction`에서 찾을 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
`Module`은 TensorFlow의 신경망 기본 클래스인 `tf.keras.Model`의 하위 클래스라는 것을 알 수 있습니다.
이는 신경망을 다루기 위한 편리한 기능을 제공합니다. 예를 들어, 내장된 `__call__` 메서드에서 `call` 메서드를 호출합니다. 여기서는 `call`을 `forward` 메서드로 리다이렉트하고, 그 인수들을 클래스 속성으로 저장합니다. 다른 프레임워크 구현과 더 비슷하게 만들기 위해 이렇게 합니다.
:end_tab:

:begin_tab:`jax`
`Module`은 Flax의 신경망 기본 클래스인 `linen.Module`의 하위 클래스라는 것을 알 수 있습니다.
이는 신경망을 다루기 위한 편리한 기능을 제공합니다. 예를 들어, 모델 매개변수를 처리하고, 코드를 단순화하는 `nn.compact` 데코레이터를 제공하며, `__call__` 메서드 등을 호출합니다.
여기서도 `__call__`을 `forward` 메서드로 리다이렉트합니다. 이는 다른 프레임워크 구현과 코드를 더 비슷하게 만들기 위함입니다.
:end_tab:

##  데이터
:label:`oo-design-data`

`DataModule` 클래스는 데이터를 위한 기본 클래스입니다. 꽤 자주 `__init__` 메서드가 데이터를 준비하는 데 사용됩니다. 여기에는 필요한 경우 다운로드와 전처리가 포함됩니다. `train_dataloader`는 훈련 데이터셋용 데이터 로더를 반환합니다. 데이터 로더는 사용될 때마다 데이터 배치를 산출하는 (Python) 제너레이터입니다. 이 배치는 그 후 `Module`의 `training_step` 메서드로 전달되어 손실을 계산합니다. 검증 데이터셋 로더를 반환하는 선택적인 `val_dataloader`도 있습니다. 이는 `Module`의 `validation_step` 메서드를 위한 데이터 배치를 산출한다는 점만 제외하면 같은 방식으로 동작합니다.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## 훈련
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
`Trainer` 클래스는 `DataModule`에 지정된 데이터로 `Module` 클래스의 학습 가능한 매개변수를 훈련합니다. 핵심 메서드는 두 인수를 받는 `fit`입니다. `model`은 `Module`의 인스턴스이고, `data`는 `DataModule`의 인스턴스입니다. 그런 다음 모델을 훈련하기 위해 전체 데이터셋을 `max_epochs` 번 반복합니다. 이전과 마찬가지로, 이 메서드의 구현은 이후 장으로 미룰 것입니다.
:end_tab:

:begin_tab:`jax`
`Trainer` 클래스는 `DataModule`에 지정된 데이터로 학습 가능한 매개변수 `params`를 훈련합니다. 핵심 메서드는 세 인수를 받는 `fit`입니다. `model`은 `Module`의 인스턴스, `data`는 `DataModule`의 인스턴스, `key`는 JAX `PRNGKeyArray`입니다. 인터페이스를 단순화하기 위해 여기서 `key` 인수를 선택 사항으로 만들었지만, JAX와 Flax에서는 항상 루트 키로 모델 매개변수를 전달하고 초기화하는 것이 권장됩니다. 그런 다음 모델을 훈련하기 위해 전체 데이터셋을 `max_epochs` 번 반복합니다. 이전과 마찬가지로, 이 메서드의 구현은 이후 장으로 미룰 것입니다.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## 요약

저희의 향후 딥러닝 구현을 위한 객체 지향 설계를 강조하기 위해,
위의 클래스들은 단순히 그 객체들이 어떻게 데이터를 저장하고
서로 상호작용하는지를 보여줍니다.
저희는 책의 나머지 부분에서 `@add_to_class` 등을 통해
이 클래스들의 구현을 계속 풍부하게 만들 것입니다.
또한, 이 완전히 구현된 클래스들은
[D2L 라이브러리](https://github.com/d2l-ai/d2l-en/tree/master/d2l)에 저장되어 있으며,
이는 딥러닝을 위한 구조화된 모델링을 쉽게 만들어 주는 *경량 툴킷*입니다.
특히, 거의 아무것도 바꾸지 않고도 프로젝트들 사이에서 많은 구성 요소를 재사용하는 것을 용이하게 합니다. 예를 들어, 옵티마이저만, 모델만, 데이터셋만 등을 교체할 수 있습니다.
이러한 수준의 모듈성은 간결함과 단순성 측면에서 책 전반에 걸쳐 보상을 가져다 주며(이것이 저희가 추가한 이유입니다), 여러분 자신의 프로젝트에서도 같은 일을 할 수 있습니다.


## 연습문제

1. [D2L 라이브러리](https://github.com/d2l-ai/d2l-en/tree/master/d2l)에 저장된 위 클래스들의 전체 구현을 찾아보세요. 딥러닝 모델링에 좀 더 익숙해진 후에 그 구현을 자세히 살펴볼 것을 강력히 권장합니다.
1. `B` 클래스에서 `save_hyperparameters` 문을 제거하세요. 여전히 `self.a`와 `self.b`를 출력할 수 있습니까? 선택 사항: `HyperParameters` 클래스의 전체 구현을 깊이 살펴보았다면, 그 이유를 설명할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17974)
:end_tab:
