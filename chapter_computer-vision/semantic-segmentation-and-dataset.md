# 시맨틱 분할과 데이터셋
:label:`sec_semantic_segmentation`

:numref:`sec_bbox`(:numref:`sec_rcnn`)에서 객체 검출 작업을 논의할 때, 이미지에서 객체를 라벨링하고 예측하기 위해 직사각형 바운딩 박스를 사용합니다.
이 절에서는 *시맨틱 분할* 문제를 논의할 것인데, 이는 이미지를 다양한 시맨틱 클래스에 속하는 영역으로 어떻게 나눌 것인가에 초점을 맞춥니다.
객체 검출과는 달리, 시맨틱 분할은 이미지에 있는 것이 무엇인지를 픽셀 수준에서 인식하고 이해합니다. 시맨틱 영역의 라벨링과 예측이 픽셀 수준입니다.
:numref:`fig_segmentation`은 시맨틱 분할에서 이미지의 개, 고양이, 배경의 라벨을 보여줍니다.
객체 검출에서와 비교해, 시맨틱 분할에서 라벨링된 픽셀 수준 경계는 분명히 더 세밀합니다.


![시맨틱 분할에서 이미지의 개, 고양이, 배경의 라벨.](../img/segmentation.svg)
:label:`fig_segmentation`


## 이미지 분할과 인스턴스 분할

컴퓨터 비전 분야에는 시맨틱 분할과 유사한 두 가지 중요한 작업도 있습니다. 즉, 이미지 분할과 인스턴스 분할입니다.
저희는 이를 시맨틱 분할과 다음과 같이 간단히 구별할 것입니다.

* *이미지 분할*은 이미지를 여러 구성 영역으로 나눕니다. 이런 유형의 문제에 대한 방법은 보통 이미지의 픽셀 간 상관관계를 활용합니다. 훈련 중에 이미지 픽셀에 대한 라벨 정보는 필요하지 않으며, 분할된 영역이 예측 중에 저희가 얻고자 하는 시맨틱을 가질 것이라고 보장할 수 없습니다. :numref:`fig_segmentation`의 이미지를 입력으로 취하면, 이미지 분할은 개를 두 영역으로 나눌 수 있습니다. 하나는 주로 검은색인 입과 눈을 덮고, 다른 하나는 주로 노란색인 몸의 나머지 부분을 덮습니다.
* *인스턴스 분할*은 *동시 검출 및 분할*이라고도 합니다. 이는 이미지에서 각 객체 인스턴스의 픽셀 수준 영역을 어떻게 인식할지를 연구합니다. 시맨틱 분할과는 달리, 인스턴스 분할은 시맨틱뿐만 아니라 다양한 객체 인스턴스도 구별해야 합니다. 예를 들어, 이미지에 두 마리의 개가 있다면, 인스턴스 분할은 픽셀이 두 개 중 어느 것에 속하는지 구별해야 합니다.



## Pascal VOC2012 시맨틱 분할 데이터셋

[**가장 중요한 시맨틱 분할 데이터셋 중 하나는 [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)입니다.**]
다음에서, 저희는 이 데이터셋을 살펴보겠습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

데이터셋의 tar 파일은 약 2GB이므로, 파일을 다운로드하는 데 시간이 좀 걸릴 수 있습니다.
추출된 데이터셋은 `../data/VOCdevkit/VOC2012`에 위치합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

경로 `../data/VOCdevkit/VOC2012`로 들어간 후, 저희는 데이터셋의 다양한 구성 요소를 볼 수 있습니다.
`ImageSets/Segmentation` 경로에는 훈련 및 테스트 샘플을 지정하는 텍스트 파일이 포함되어 있는 반면, `JPEGImages`와 `SegmentationClass` 경로는 각 예제에 대한 입력 이미지와 라벨을 각각 저장합니다.
여기서 라벨도 이미지 형식이며, 라벨링된 입력 이미지와 같은 크기를 가집니다.
또한, 어떤 라벨 이미지에서든 같은 색상의 픽셀은 같은 시맨틱 클래스에 속합니다.
다음은 [**모든 입력 이미지와 라벨을 메모리에 읽기**] 위한 `read_voc_images` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

[**처음 다섯 개의 입력 이미지와 라벨을 그려**] 봅니다.
라벨 이미지에서, 흰색과 검은색은 각각 경계와 배경을 나타내며, 다른 색상은 다른 클래스에 해당합니다.

```{.python .input}
#@tab mxnet
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

다음으로, 저희는 이 데이터셋의 모든 라벨에 대한 [**RGB 색상 값과 클래스 이름을 열거**]합니다.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

위에 정의된 두 상수를 사용하면, 저희는 편리하게 [**라벨에서 각 픽셀의 클래스 인덱스를 찾**]을 수 있습니다.
저희는 위의 RGB 색상 값에서 클래스 인덱스로의 매핑을 만드는 `voc_colormap2label` 함수와 이 Pascal VOC2012 데이터셋에서 어떤 RGB 값이든 그들의 클래스 인덱스로 매핑하는 `voc_label_indices` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**예를 들어**], 첫 번째 예제 이미지에서, 비행기의 앞부분에 대한 클래스 인덱스는 1이고, 배경 인덱스는 0입니다.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### 데이터 전처리

:numref:`sec_alexnet`(:numref:`sec_googlenet`)와 같은 이전 실험에서, 이미지는 모델이 요구하는 입력 형태에 맞게 크기가 조정됩니다.
하지만, 시맨틱 분할에서, 그렇게 하는 것은 예측된 픽셀 클래스를 입력 이미지의 원래 형태로 다시 크기 조정해야 합니다.
이러한 크기 조정은 부정확할 수 있는데, 특히 다양한 클래스의 분할된 영역의 경우 그렇습니다. 이 문제를 피하기 위해, 저희는 이미지를 크기 조정하는 대신 *고정된* 형태로 자릅니다. 구체적으로, [**이미지 증강에서 랜덤 자르기를 사용해, 저희는 입력 이미지와 라벨의 같은 영역을 자릅니다**].

```{.python .input}
#@tab mxnet
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab mxnet
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**커스텀 시맨틱 분할 데이터셋 클래스**]

저희는 고수준 API에서 제공하는 `Dataset` 클래스를 상속받아 커스텀 시맨틱 분할 데이터셋 클래스 `VOCSegDataset`을 정의합니다.
`__getitem__` 함수를 구현함으로써, 저희는 데이터셋에서 `idx`로 인덱싱된 입력 이미지와 이 이미지의 각 픽셀의 클래스 인덱스에 임의로 접근할 수 있습니다.
데이터셋의 일부 이미지가 랜덤 자르기의 출력 크기보다 작은 크기를 가지므로, 이러한 예제는 커스텀 `filter` 함수에 의해 필터링됩니다.
또한, 입력 이미지의 세 RGB 채널의 값을 표준화하기 위해 `normalize_image` 함수도 정의합니다.

```{.python .input}
#@tab mxnet
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**데이터셋 읽기**]

저희는 커스텀 `VOCSegDatase`t 클래스를 사용해 훈련 셋과 테스트 셋의 인스턴스를 각각 생성합니다.
저희가 랜덤하게 잘린 이미지의 출력 형태를 $320\times 480$로 지정한다고 가정합니다.
아래에서 저희는 훈련 셋과 테스트 셋에 유지된 예제의 수를 볼 수 있습니다.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

배치 크기를 64로 설정하고, 저희는 훈련 셋에 대한 데이터 이터레이터를 정의합니다.
첫 번째 미니배치의 형태를 출력해 보겠습니다.
이미지 분류나 객체 검출과는 달리, 여기서 라벨은 3차원 텐서입니다.

```{.python .input}
#@tab mxnet
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**모두 합치기**]

마지막으로, 저희는 Pascal VOC2012 시맨틱 분할 데이터셋을 다운로드하고 읽기 위해 다음 `load_data_voc` 함수를 정의합니다.
이는 훈련 및 테스트 데이터셋 모두에 대한 데이터 이터레이터를 반환합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## 요약

* 시맨틱 분할은 이미지를 다양한 시맨틱 클래스에 속하는 영역으로 나눔으로써 이미지에 있는 것이 무엇인지를 픽셀 수준에서 인식하고 이해합니다.
* 가장 중요한 시맨틱 분할 데이터셋 중 하나는 Pascal VOC2012입니다.
* 시맨틱 분할에서, 입력 이미지와 라벨이 픽셀에서 일대일로 대응하므로, 입력 이미지는 크기가 조정되는 것이 아니라 고정된 형태로 랜덤하게 잘립니다.


## 연습문제

1. 시맨틱 분할이 자율 주행 차량과 의료 영상 진단에 어떻게 적용될 수 있을까요? 다른 응용을 생각해 볼 수 있나요?
1. :numref:`sec_image_augmentation`의 데이터 증강 설명을 떠올려 보세요. 이미지 분류에서 사용된 이미지 증강 방법 중 어떤 것이 시맨틱 분할에 적용하기 실현 불가능할까요?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab:
