# 객체 검출 데이터셋
:label:`sec_object-detection-dataset`

객체 검출 분야에는 MNIST나 Fashion-MNIST 같은 작은 데이터셋이 없습니다.
객체 검출 모델을 빠르게 시연하기 위해, [**저희는 작은 데이터셋을 수집하고 라벨링했습니다**].
먼저, 저희는 사무실에 있는 무료 바나나의 사진을 찍어 다양한 회전과 크기의 1000개 바나나 이미지를 생성했습니다.
그런 다음 각 바나나 이미지를 어떤 배경 이미지의 랜덤한 위치에 배치했습니다.
마지막으로, 이미지에서 그 바나나의 바운딩 박스를 라벨링했습니다.


## [**데이터셋 다운로드**]

모든 이미지와 csv 라벨 파일이 있는 바나나 검출 데이터셋은 인터넷에서 바로 다운로드할 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 데이터셋 읽기

저희는 아래의 `read_data_bananas` 함수에서 [**바나나 검출 데이터셋을 읽을**] 것입니다.
데이터셋에는 객체 클래스 라벨과 좌상단 및 우하단 꼭짓점의 실측 바운딩 박스 좌표에 대한 csv 파일이 포함되어 있습니다.

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

이미지와 라벨을 읽기 위해 `read_data_bananas` 함수를 사용하면, 다음의 `BananasDataset` 클래스는 저희가 바나나 검출 데이터셋을 로드하기 위해 [**커스터마이즈된 `Dataset` 인스턴스를 만들 수 있게**] 해줍니다.

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

마지막으로, [**훈련 및 테스트 셋 모두에 대한 두 개의 데이터 이터레이터 인스턴스를 반환**]하기 위해 `load_data_bananas` 함수를 정의합니다.
테스트 데이터셋의 경우, 랜덤한 순서로 읽을 필요가 없습니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

[**미니배치를 읽고 이 미니배치에서 이미지와 라벨의 형태를 둘 다 출력**]해 봅시다.
이미지 미니배치의 형태인 (배치 크기, 채널 수, 높이, 너비)는 익숙해 보입니다. 이는 저희의 이전 이미지 분류 작업과 같습니다.
라벨 미니배치의 형태는 (배치 크기, $m$, 5)이며, 여기서 $m$은 데이터셋의 어떤 이미지가 가질 수 있는 바운딩 박스의 최대 수입니다.

미니배치에서의 계산이 더 효율적이지만, 모든 이미지 예제가 연결을 통해 미니배치를 형성하기 위해 같은 수의 바운딩 박스를 포함해야 합니다.
일반적으로, 이미지는 다양한 수의 바운딩 박스를 가질 수 있습니다. 따라서, $m$보다 적은 바운딩 박스를 가진 이미지는 $m$에 도달할 때까지 유효하지 않은 바운딩 박스로 패딩됩니다.
그러면 각 바운딩 박스의 라벨은 길이 5의 배열로 표현됩니다.
배열의 첫 번째 원소는 바운딩 박스에 있는 객체의 클래스이며, 여기서 -1은 패딩을 위한 유효하지 않은 바운딩 박스를 나타냅니다.
배열의 나머지 네 원소는 바운딩 박스의 좌상단 꼭짓점과 우하단 꼭짓점의 ($x$, $y$) 좌표 값입니다(범위는 0과 1 사이).
바나나 데이터셋의 경우, 각 이미지에 바운딩 박스가 하나만 있으므로 $m=1$입니다.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**시연**]

라벨링된 실측 바운딩 박스와 함께 열 개의 이미지를 시연해 봅시다.
이 모든 이미지에서 바나나의 회전, 크기, 위치가 다른 것을 볼 수 있습니다.
물론, 이는 단순한 인공 데이터셋입니다.
실제로 실세계 데이터셋은 보통 훨씬 더 복잡합니다.

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## 요약

* 저희가 수집한 바나나 검출 데이터셋은 객체 검출 모델을 시연하는 데 사용할 수 있습니다.
* 객체 검출을 위한 데이터 로딩은 이미지 분류를 위한 것과 비슷합니다. 하지만, 객체 검출에서 라벨은 실측 바운딩 박스의 정보도 포함하는데, 이는 이미지 분류에서는 누락된 것입니다.


## 연습문제

1. 바나나 검출 데이터셋에서 실측 바운딩 박스가 있는 다른 이미지들을 시연해 보세요. 바운딩 박스와 객체와 관련해 어떻게 다른가요?
1. 객체 검출에 랜덤 자르기와 같은 데이터 증강을 적용하고 싶다고 가정해 봅시다. 이미지 분류에서와 어떻게 다를 수 있을까요? 힌트: 잘린 이미지가 객체의 작은 부분만 포함한다면 어떻게 될까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
