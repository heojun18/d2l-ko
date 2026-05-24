```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 합성 회귀 데이터
:label:`sec_synthetic-regression-data`


머신러닝은 모두 데이터에서 정보를 추출하는 일입니다.
그렇다면 합성 데이터에서 도대체 무엇을 배울 수 있을지 궁금할 수 있습니다.
저희가 인공적인 데이터 생성 모델에 직접 심어 놓은 패턴 자체에는
본질적인 관심이 없을 수 있지만,
그러한 데이터셋은 교육적 목적에서는 유용합니다.
저희가 학습 알고리즘의 성질을 평가하고
구현이 예상대로 작동하는지 확인하는 데 도움이 됩니다.
예를 들어, 올바른 매개변수가 *사전에* 알려진 데이터를 만든다면,
저희 모델이 실제로 그것들을 복원할 수 있는지 확인할 수 있습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## 데이터셋 생성

이 예제에서는 간결함을 위해 저차원에서 작업하겠습니다.
다음 코드 스니펫은 표준 정규 분포에서 추출된
2차원 특징을 가진 1000개의 예제를 생성합니다.
결과로 나오는 설계 행렬 $\mathbf{X}$는 $\mathbb{R}^{1000 \times 2}$에 속합니다.
저희는 *정답(ground truth)* 선형 함수를 적용하고,
각 예제에 대해 독립적이고 동등하게 추출된
가산 잡음 $\boldsymbol{\epsilon}$으로 교란시켜 각 레이블을 생성합니다.

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

편의를 위해 $\boldsymbol{\epsilon}$이 평균 $\mu= 0$과
표준 편차 $\sigma = 0.01$을 가진 정규 분포에서 추출되었다고 가정합니다.
객체 지향 설계를 위해, `d2l.DataModule`(:numref:`oo-design-data`에서 소개됨)의
하위 클래스의 `__init__` 메서드에 코드를 추가한다는 점에 유의하세요.
어떠한 추가 하이퍼파라미터든 설정할 수 있게 허용하는 것이 좋은 관행입니다.
이를 `save_hyperparameters()`로 달성합니다.
`batch_size`는 나중에 결정됩니다.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

아래에서, 저희는 참 매개변수를 $\mathbf{w} = [2, -3.4]^\top$와 $b = 4.2$로 설정합니다.
나중에, 추정된 매개변수를 이러한 *정답* 값과 대조해 확인할 수 있습니다.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**`features`의 각 행은 $\mathbb{R}^2$의 벡터로 구성되고 `labels`의 각 행은 스칼라입니다.**] 첫 번째 항목을 살펴봅시다.

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## 데이터셋 읽기

머신러닝 모델을 훈련하려면 종종 한 번에 한 미니배치의 예제를 가져오면서
데이터셋을 여러 번 통과해야 합니다.
이 데이터는 그 후 모델을 갱신하는 데 사용됩니다.
이것이 어떻게 작동하는지 설명하기 위해,
[**`get_dataloader` 메서드를 구현하고,**]
`add_to_class`(:numref:`oo-design-utilities`에서 소개됨)를 통해
`SyntheticRegressionData` 클래스에 등록합니다.
이는 (**배치 크기, 특징 행렬, 레이블 벡터를 받아,
크기 `batch_size`의 미니배치를 생성합니다.**)
따라서 각 미니배치는 특징과 레이블의 튜플로 구성됩니다.
저희가 훈련 모드인지 검증 모드인지 유의해야 한다는 점에 주의하세요.
훈련 모드에서는 데이터를 무작위 순서로 읽고 싶을 것이고,
검증 모드에서는 디버깅 목적을 위해
미리 정의된 순서로 데이터를 읽을 수 있는 것이 중요할 수 있습니다.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

직관을 쌓기 위해, 데이터의 첫 번째 미니배치를 살펴봅시다. 특징의 각 미니배치는 그 크기와 입력 특징의 차원성을 모두 알려줍니다.
마찬가지로 저희 레이블의 미니배치는 `batch_size`로 주어지는 일치하는 형태를 가질 것입니다.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

해롭지 않아 보이지만, `iter(data.train_dataloader())`의 호출은
Python의 객체 지향 설계의 위력을 보여 줍니다.
저희가 `data` 객체를 생성한 *후에*
`SyntheticRegressionData` 클래스에 메서드를 추가했다는 점에 유의하세요.
그럼에도 불구하고 객체는 클래스에 *사후적으로* 기능이 추가된 것의 혜택을 누립니다.

반복하는 동안 저희는 전체 데이터셋이 소진될 때까지
서로 다른 미니배치를 얻습니다(이를 시도해 보세요).
위에서 구현한 반복은 교육적 목적으로는 좋지만,
실제 문제에서는 저희를 곤란하게 만들 수 있는 방식으로 비효율적입니다.
예를 들어 모든 데이터를 메모리에 로드해야 하고,
많은 무작위 메모리 접근을 수행해야 합니다.
딥러닝 프레임워크에 구현된 내장 반복자는
훨씬 더 효율적이며, 파일에 저장된 데이터,
스트림을 통해 받은 데이터, 즉석에서 생성되거나 처리된 데이터 같은
출처를 다룰 수 있습니다.
다음으로 내장 반복자를 사용해 같은 메서드를 구현해 봅시다.

## 데이터 로더의 간결한 구현

저희 자신의 반복자를 작성하는 대신,
[**프레임워크의 기존 API를 호출해 데이터를 로드할 수 있습니다.**]
이전과 마찬가지로, 특징 `X`와 레이블 `y`를 가진 데이터셋이 필요합니다.
그 외에는 내장 데이터 로더에서 `batch_size`를 설정하고,
효율적으로 예제를 섞는 일은 데이터 로더에 맡깁니다.

:begin_tab:`jax`
JAX는 디바이스 가속과 함수형 변환을 갖춘 NumPy 같은 API에 관한 것이므로,
적어도 현재 버전에는 데이터 로딩 메서드가 포함되어 있지 않습니다. 다른 라이브러리에는
이미 훌륭한 데이터 로더가 있으며, JAX는 그것들을 대신 사용할 것을 제안합니다.
여기서는 TensorFlow의 데이터 로더를 가져와 JAX에서 작동하도록 약간 수정하겠습니다.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

새로운 데이터 로더는 이전 것과 똑같이 동작하지만, 더 효율적이고 일부 추가 기능을 가지고 있습니다.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

예를 들어, 프레임워크 API가 제공하는 데이터 로더는
내장 `__len__` 메서드를 지원하므로,
그 길이, 즉 배치의 수를 조회할 수 있습니다.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## 요약

데이터 로더는 데이터를 로드하고 조작하는 과정을
추상화하는 편리한 방법입니다.
이러한 방식으로 같은 머신러닝 *알고리즘*이
수정 없이 다양한 유형과 출처의 데이터를 처리할 수 있습니다.
데이터 로더의 좋은 점 중 하나는 그것들이 조합될 수 있다는 것입니다.
예를 들어, 이미지를 로드한 다음
이미지를 자르거나 다른 방식으로 수정하는 후처리 필터를 가질 수 있습니다.
따라서 데이터 로더는 전체 데이터 처리 파이프라인을
기술하는 데 사용될 수 있습니다.

모델 자체에 대해 말하자면, 2차원 선형 모델은
저희가 마주칠 수 있는 가장 단순한 모델에 가깝습니다.
이는 데이터의 양이 부족하거나 방정식 시스템이 부정(underdetermined)이라는
걱정 없이 회귀 모델의 정확도를 시험해 볼 수 있게 해 줍니다.
다음 절에서 이를 잘 활용할 것입니다.


## 연습문제

1. 예제의 수가 배치 크기로 나누어떨어지지 않으면 어떻게 됩니까? 프레임워크의 API를 사용해 다른 인수를 지정함으로써 이 동작을 어떻게 바꾸시겠습니까?
1. 매개변수 벡터 `w`의 크기와 예제의 수 `num_examples`가 모두 큰, 거대한 데이터셋을 생성하고 싶다고 가정합니다.
    1. 모든 데이터를 메모리에 담을 수 없다면 어떻게 됩니까?
    1. 데이터가 디스크에 보관되어 있다면 어떻게 섞으시겠습니까? 여러분의 과제는 너무 많은 무작위 읽기나 쓰기를 요구하지 않는 *효율적인* 알고리즘을 설계하는 것입니다. 힌트: [의사 무작위 순열 생성기](https://en.wikipedia.org/wiki/Pseudorandom_permutation)는 순열 테이블을 명시적으로 저장하지 않고도 재섞기를 설계할 수 있게 해 줍니다 :cite:`Naor.Reingold.1999`. 
1. 반복자가 호출될 때마다 즉석에서 새 데이터를 생산하는 데이터 생성기를 구현하세요. 
1. 호출될 때마다 *동일한* 데이터를 생성하는 무작위 데이터 생성기를 어떻게 설계하시겠습니까?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17975)
:end_tab:
