```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 이미지 분류 데이터셋
:label:`sec_fashion_mnist`

(~~MNIST 데이터셋은 이미지 분류에 널리 사용되는 데이터셋 중 하나이지만, 벤치마크 데이터셋으로는 너무 단순합니다. 저희는 이와 비슷하지만 더 복잡한 Fashion-MNIST 데이터셋을 사용할 것입니다 ~~)

이미지 분류에 널리 사용되는 데이터셋 중 하나는 손글씨 숫자로 이루어진 [MNIST 데이터셋](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`입니다. 1990년대 출시 당시 이 데이터셋은 $28 \times 28$ 픽셀 해상도의 60,000개 이미지(추가로 10,000개의 테스트 이미지)로 구성되어 있어 대부분의 머신러닝 알고리즘에 만만치 않은 도전 과제였습니다. 그 시대를 가늠해 보면, 1995년 당시 무려 64MB의 RAM과 눈부신 5 MFLOPs를 자랑하는 Sun SPARCStation 5는 AT&T 벨 연구소에서 머신러닝을 위한 최첨단 장비로 여겨졌습니다. 숫자 인식에서 높은 정확도를 달성하는 것은 1990년대 USPS의 우편 분류 자동화의 핵심 구성 요소였습니다. LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`와 같은 심층 신경망, 불변성을 가진 서포트 벡터 머신 :cite:`Scholkopf.Burges.Vapnik.1996`, 그리고 탄젠트 거리 분류기 :cite:`Simard.LeCun.Denker.ea.1998` 모두 1% 미만의 오류율에 도달할 수 있었습니다.

10년이 넘는 기간 동안 MNIST는 머신러닝 알고리즘을 비교하기 위한 *기준점*의 역할을 했습니다.
벤치마크 데이터셋으로 좋은 시절을 보냈지만,
오늘날 기준으로 보면 간단한 모델조차도 95%가 넘는 분류 정확도를 달성하여,
강한 모델과 약한 모델을 구별하기에는 부적합해졌습니다. 더 나아가, 이 데이터셋은 *매우* 높은 수준의 정확도를 허용하는데, 이는 많은 분류 문제에서 일반적으로 볼 수 없는 수준입니다. 이는 알고리즘 개발이 깨끗한 데이터셋의 장점을 활용할 수 있는 특정 계열의 알고리즘(예: 능동 집합 방법, 경계 탐색 능동 집합 알고리즘 등)으로 편향되도록 만들었습니다.
오늘날 MNIST는 벤치마크라기보다 새너티 체크(sanity check)의 역할을 합니다. ImageNet :cite:`Deng.Dong.Socher.ea.2009`은 훨씬 더
관련성 있는 도전 과제를 제시합니다. 안타깝게도 ImageNet은 이 책의 많은 예제와 그림에 사용하기에는 너무 크기 때문에, 예제를 인터랙티브하게 만들기 위해 학습하는 데 시간이 너무 오래 걸립니다. 대안으로 저희는 다가오는 절들에서 질적으로 유사하지만 훨씬 더 작은 Fashion-MNIST
데이터셋 :cite:`Xiao.Rasul.Vollgraf.2017`에 초점을 맞춰 논의를 진행할 것입니다. 이 데이터셋은 2017년에 출시되었고, $28 \times 28$ 픽셀 해상도의 10개 의류 범주 이미지로 구성되어 있습니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## 데이터셋 로드

Fashion-MNIST 데이터셋이 매우 유용하기 때문에 모든 주요 프레임워크는 전처리된 버전을 제공합니다. 저희는 [**프레임워크 내장 유틸리티를 사용해 다운로드하고 메모리로 읽어들일 수 있습니다.**]

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST는 10개 범주의 이미지로 구성되어 있으며, 각 범주는
학습 데이터셋에서 6000개의 이미지로, 테스트 데이터셋에서 1000개의 이미지로 표현됩니다.
*테스트 데이터셋(test dataset)*은 모델 성능을 평가하는 데 사용됩니다(학습에 사용되어서는 안 됩니다).
따라서 학습 셋과 테스트 셋은 각각
60,000개와 10,000개의 이미지를 포함합니다.

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

이미지들은 그레이스케일이며 위에서 $32 \times 32$ 픽셀 해상도로 업스케일되었습니다. 이는 (이진) 흑백 이미지로 구성되었던 원래의 MNIST 데이터셋과 유사합니다. 다만 대부분의 현대 이미지 데이터는 세 채널(빨강, 초록, 파랑)을 가지며, 초분광 이미지는 100개를 초과하는 채널을 가질 수 있다는 점에 유의하시기 바랍니다(HyMap 센서는 126개의 채널을 가집니다).
관례적으로 저희는 이미지를 $c \times h \times w$ 텐서로 저장합니다. 여기서 $c$는 컬러 채널의 수, $h$는 높이, $w$는 너비입니다.

```{.python .input}
%%tab all
data.train[0][0].shape
```

[~~데이터셋을 시각화하기 위한 두 가지 유틸리티 함수~~]

Fashion-MNIST의 범주는 사람이 이해할 수 있는 이름을 가집니다.
다음 편의 메서드는 숫자 레이블과 그 이름 사이를 변환합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## 미니배치 읽기

학습 셋과 테스트 셋에서 읽어 들이는 작업을 더 쉽게 하기 위해,
처음부터 만들기보다는 내장된 데이터 이터레이터를 사용합니다.
각 반복에서, 데이터 이터레이터가
[**`batch_size` 크기의 미니배치 데이터를 읽는다는 점을**] 떠올려 보시기 바랍니다.
또한 학습 데이터 이터레이터에 대해서는 예제들을 무작위로 셔플합니다.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

이것이 어떻게 동작하는지 보기 위해, `train_dataloader` 메서드를 호출하여 이미지 미니배치를 로드해 봅시다. 여기에는 64개의 이미지가 포함됩니다.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

이미지를 읽어 들이는 데 걸리는 시간을 살펴봅시다. 내장 로더임에도 불구하고 눈부시게 빠르지는 않습니다. 그럼에도 불구하고, 심층 신경망으로 이미지를 처리하는 데 훨씬 더 오랜 시간이 걸리기 때문에 이것으로 충분합니다. 따라서 신경망 학습이 I/O에 의해 제약을 받지 않을 정도로 충분히 좋은 수준입니다.

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## 시각화

저희는 Fashion-MNIST 데이터셋을 자주 사용하게 될 것입니다. 편의 함수 `show_images`를 사용하면 이미지와 관련 레이블을 시각화할 수 있습니다.
구현 세부 사항은 건너뛰고, 아래에 인터페이스만 보여드립니다. 이러한 유틸리티 함수의 경우 어떻게 작동하는지보다는 `d2l.show_images`를 어떻게 호출하는지만 알면 됩니다.

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

이 함수를 유용하게 활용해 봅시다. 일반적으로 학습에 사용하는 데이터를 시각화하고 검사하는 것은 좋은 습관입니다.
사람은 이상한 점을 발견하는 데 매우 능숙하기 때문에, 시각화는 실험 설계에서 발생하는 실수와 오류에 대한 추가적인 안전장치 역할을 합니다. 여기에 학습 데이터셋의 처음 몇 개 예제에 대한
[**이미지와 그에 대응하는 (텍스트로 된) 레이블**]이 있습니다.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

이제 저희는 다음 절들에서 Fashion-MNIST 데이터셋을 다룰 준비가 되었습니다.

## 요약

이제 저희는 분류에 사용할 약간 더 현실적인 데이터셋을 가지게 되었습니다. Fashion-MNIST는 10개 범주를 나타내는 이미지로 구성된 의류 분류 데이터셋입니다. 저희는 이 데이터셋을 다음 절들과 챕터들에서 단순한 선형 모델부터 고급 잔차 신경망에 이르기까지 다양한 신경망 설계를 평가하는 데 사용할 것입니다. 이미지를 다룰 때 흔히 그러하듯, 저희는 이미지를 (배치 크기, 채널 수, 높이, 너비) 형태의 텐서로 읽어들입니다. 현재로서는 이미지가 그레이스케일이므로 채널이 하나뿐입니다(위의 시각화는 가시성을 높이기 위해 가짜 컬러 팔레트를 사용합니다).

마지막으로, 데이터 이터레이터는 효율적인 성능을 위한 핵심 구성 요소입니다. 예를 들어, 효율적인 이미지 디컴프레션, 비디오 트랜스코딩 또는 기타 전처리를 위해 GPU를 사용할 수도 있습니다. 가능한 한 학습 루프의 속도가 저하되는 것을 막기 위해 고성능 컴퓨팅을 활용하는 잘 구현된 데이터 이터레이터에 의존해야 합니다.


## 연습문제

1. `batch_size`를 줄이면(예: 1로) 읽기 성능에 영향이 있나요?
1. 데이터 이터레이터 성능은 중요합니다. 현재 구현이 충분히 빠르다고 생각하나요? 이를 개선할 다양한 옵션을 탐구해 보시기 바랍니다. 시스템 프로파일러를 사용하여 병목이 어디에 있는지 찾아보시기 바랍니다.
1. 프레임워크의 온라인 API 문서를 확인해 보시기 바랍니다. 어떤 다른 데이터셋들이 사용 가능한가요?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17980)
:end_tab:
