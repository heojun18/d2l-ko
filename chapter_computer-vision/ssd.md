# SSD(Single Shot Multibox Detection)
:label:`sec_ssd`

:numref:`sec_bbox`(:numref:`sec_object-detection-dataset`)에서 저희는 바운딩 박스, 앵커 박스, 멀티스케일 객체 검출, 객체 검출용 데이터셋을 소개했습니다.
이제 그러한 배경 지식을 사용해 객체 검출 모델인 SSD(single shot multibox detection) :cite:`Liu.Anguelov.Erhan.ea.2016`를 설계할 준비가 되었습니다.
이 모델은 간단하고 빠르며 폭넓게 사용됩니다.
이는 방대한 양의 객체 검출 모델 중 하나일 뿐이지만, 이 절의 일부 설계 원칙과 구현 세부사항은 다른 모델에도 적용될 수 있습니다.


## 모델

:numref:`fig_ssd`는 SSD의 설계에 대한 개요를 제공합니다.
이 모델은 주로 기본 신경망과 그 뒤에 오는 여러 멀티스케일 특징 맵 블록으로 구성됩니다.
기본 신경망은 입력 이미지에서 특징을 추출하기 위한 것이므로, 심층 CNN을 사용할 수 있습니다.
예를 들어, 원래의 SSD 논문은 분류 계층 이전에서 잘린 VGG 네트워크를 채택하지만 :cite:`Liu.Anguelov.Erhan.ea.2016`, ResNet도 일반적으로 사용되어 왔습니다.
저희의 설계를 통해 기본 신경망이 더 큰 특징 맵을 출력하도록 만들 수 있는데, 그러면 더 작은 객체를 검출하기 위해 더 많은 앵커 박스를 생성할 수 있습니다.
이후, 각 멀티스케일 특징 맵 블록은 이전 블록의 특징 맵의 높이와 너비를 (예: 절반으로) 줄이고, 특징 맵의 각 단위가 입력 이미지에 대한 수용 영역을 늘릴 수 있게 합니다.


:numref:`sec_multiscale-object-detection`의 심층 신경망에 의한 이미지의 계층별 표현을 통한 멀티스케일 객체 검출의 설계를 떠올려 보세요.
:numref:`fig_ssd`의 위쪽에 더 가까운 멀티스케일 특징 맵은 더 작지만 더 큰 수용 영역을 가지므로, 더 적지만 더 큰 객체를 검출하는 데 적합합니다.

요컨대, 기본 신경망과 여러 멀티스케일 특징 맵 블록을 통해, SSD는 다양한 크기의 다양한 수의 앵커 박스를 생성하고, 이러한 앵커 박스(그리고 따라서 바운딩 박스)의 클래스와 오프셋을 예측함으로써 다양한 크기의 객체를 검출합니다. 따라서, 이는 멀티스케일 객체 검출 모델입니다.


![멀티스케일 객체 검출 모델로서, SSD는 주로 기본 신경망과 그 뒤에 오는 여러 멀티스케일 특징 맵 블록으로 구성됩니다.](../img/ssd.svg)
:label:`fig_ssd`


다음에서, 저희는 :numref:`fig_ssd`의 다양한 블록의 구현 세부사항을 설명할 것입니다. 먼저, 클래스와 바운딩 박스 예측을 어떻게 구현하는지 논의합니다.



### [**클래스 예측 계층**]

객체 클래스의 수를 $q$라고 합시다.
그러면 앵커 박스는 $q+1$개의 클래스를 가지며, 여기서 클래스 0은 배경입니다.
어떤 스케일에서, 특징 맵의 높이와 너비가 각각 $h$와 $w$라고 가정합니다.
이러한 특징 맵의 각 공간적 위치를 중심으로 $a$개의 앵커 박스가 생성될 때, 총 $hwa$개의 앵커 박스가 분류되어야 합니다.
이는 종종 완전 연결 계층을 사용한 분류가 무거운 매개변수화 비용 가능성 때문에 실현 불가능하게 만듭니다.
:numref:`sec_nin`에서 합성곱 계층의 채널을 사용해 클래스를 예측한 방법을 떠올려 보세요.
SSD는 모델 복잡도를 줄이기 위해 동일한 기법을 사용합니다.

구체적으로, 클래스 예측 계층은 특징 맵의 너비나 높이를 변경하지 않는 합성곱 계층을 사용합니다.
이렇게 하면, 특징 맵의 같은 공간 차원(너비와 높이)에서 출력과 입력 사이에 일대일 대응이 있을 수 있습니다.
보다 구체적으로, 임의의 공간적 위치 ($x$, $y$)에서의 출력 특징 맵의 채널은 입력 특징 맵의 ($x$, $y$)를 중심으로 하는 모든 앵커 박스에 대한 클래스 예측을 나타냅니다.
유효한 예측을 생성하려면, $a(q+1)$개의 출력 채널이 있어야 하며, 같은 공간적 위치에 대해 인덱스 $i(q+1) + j$의 출력 채널은 앵커 박스 $i$ ($0 \leq i < a$)에 대한 클래스 $j$ ($0 \leq j \leq q$)의 예측을 나타냅니다.

아래에서 저희는 그러한 클래스 예측 계층을 정의하는데, $a$와 $q$를 인자 `num_anchors`와 `num_classes`로 각각 지정합니다.
이 계층은 패딩이 1인 $3\times3$ 합성곱 계층을 사용합니다.
이 합성곱 계층의 입력과 출력의 너비와 높이는 변하지 않습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**바운딩 박스 예측 계층**)

바운딩 박스 예측 계층의 설계는 클래스 예측 계층의 설계와 유사합니다.
유일한 차이는 각 앵커 박스에 대한 출력 수에 있습니다. 여기서는 $q+1$개의 클래스가 아니라 네 개의 오프셋을 예측해야 합니다.

```{.python .input}
#@tab mxnet
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**여러 스케일에 대한 예측 연결**]

저희가 언급했듯이, SSD는 멀티스케일 특징 맵을 사용해 앵커 박스를 생성하고 그 클래스와 오프셋을 예측합니다.
다양한 스케일에서, 특징 맵의 형태나 같은 단위를 중심으로 하는 앵커 박스의 수가 다를 수 있습니다.
따라서, 다양한 스케일에서 예측 출력의 형태가 다를 수 있습니다.

다음 예제에서, 저희는 같은 미니배치에 대해 두 가지 다른 스케일의 특징 맵 `Y1`과 `Y2`를 구성하는데, 여기서 `Y2`의 높이와 너비는 `Y1`의 절반입니다.
클래스 예측을 예로 들어 보겠습니다.
`Y1`과 `Y2`의 모든 단위에 대해 각각 5개와 3개의 앵커 박스가 생성된다고 가정합니다.
나아가 객체 클래스 수가 10이라고 가정합니다.
특징 맵 `Y1`과 `Y2`에 대해 클래스 예측 출력의 채널 수는 각각 $5\times(10+1)=55$와 $3\times(10+1)=33$이며, 두 출력 형태 모두 (배치 크기, 채널 수, 높이, 너비)입니다.

```{.python .input}
#@tab mxnet
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

보시다시피, 배치 크기 차원을 제외하고는, 다른 세 차원 모두 크기가 다릅니다.
보다 효율적인 계산을 위해 이 두 예측 출력을 연결하기 위해, 저희는 이러한 텐서를 더 일관된 형식으로 변환할 것입니다.

채널 차원이 같은 중심을 가진 앵커 박스에 대한 예측을 가진다는 점에 유의하세요.
저희는 먼저 이 차원을 가장 안쪽으로 옮깁니다.
배치 크기는 다양한 스케일에서도 동일하게 유지되므로, 저희는 예측 출력을 (배치 크기, 높이 $\times$ 너비 $\times$ 채널 수) 형태의 2차원 텐서로 변환할 수 있습니다.
그런 다음 저희는 차원 1을 따라 그러한 다양한 스케일의 출력을 연결할 수 있습니다.

```{.python .input}
#@tab mxnet
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

이런 식으로, `Y1`과 `Y2`가 채널, 높이, 너비에서 다른 크기를 가지더라도, 저희는 여전히 같은 미니배치에 대해 두 가지 다른 스케일의 이 두 예측 출력을 연결할 수 있습니다.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**다운샘플링 블록**]

여러 스케일에서 객체를 검출하기 위해, 저희는 입력 특징 맵의 높이와 너비를 절반으로 줄이는 다음 다운샘플링 블록 `down_sample_blk`를 정의합니다.
사실, 이 블록은 :numref:`subsec_vgg-blocks`의 VGG 블록의 설계를 적용합니다.
보다 구체적으로, 각 다운샘플링 블록은 패딩이 1인 두 개의 $3\times3$ 합성곱 계층과 그 뒤에 오는 스트라이드가 2인 $2\times2$ 최대 풀링 계층으로 구성됩니다.
저희가 알다시피, 패딩이 1인 $3\times3$ 합성곱 계층은 특징 맵의 형태를 변경하지 않습니다.
하지만, 후속하는 $2\times2$ 최대 풀링은 입력 특징 맵의 높이와 너비를 절반으로 줄입니다.
이 다운샘플링 블록의 입력과 출력 특징 맵 모두에 대해, $1\times 2+(3-1)+(3-1)=6$이므로, 출력의 각 단위는 입력에서 $6\times6$ 수용 영역을 가집니다.
따라서, 다운샘플링 블록은 출력 특징 맵의 각 단위의 수용 영역을 확대합니다.

```{.python .input}
#@tab mxnet
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

다음 예제에서, 저희가 구성한 다운샘플링 블록은 입력 채널 수를 변경하고 입력 특징 맵의 높이와 너비를 절반으로 줄입니다.

```{.python .input}
#@tab mxnet
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**기본 신경망 블록**]

기본 신경망 블록은 입력 이미지에서 특징을 추출하는 데 사용됩니다.
단순화를 위해, 저희는 각 블록에서 채널 수를 두 배로 늘리는 세 개의 다운샘플링 블록으로 구성된 작은 기본 신경망을 구성합니다.
$256\times256$ 입력 이미지가 주어지면, 이 기본 신경망 블록은 $32 \times 32$ 특징 맵($256/2^3=32$)을 출력합니다.

```{.python .input}
#@tab mxnet
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### 완전한 모델


[**완전한 SSD 모델은 다섯 개의 블록으로 구성됩니다.**]
각 블록에 의해 생성된 특징 맵은 (i) 앵커 박스를 생성하는 것과 (ii) 이러한 앵커 박스의 클래스와 오프셋을 예측하는 데 모두 사용됩니다.
이 다섯 블록 중, 첫 번째는 기본 신경망 블록이고, 두 번째에서 네 번째는 다운샘플링 블록이며, 마지막 블록은 전역 최대 풀링을 사용해 높이와 너비를 모두 1로 줄입니다.
기술적으로, 두 번째부터 다섯 번째 블록은 모두 :numref:`fig_ssd`의 그 멀티스케일 특징 맵 블록입니다.

```{.python .input}
#@tab mxnet
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

이제 저희는 각 블록에 대한 [**순전파를 정의**]합니다.
이미지 분류 작업과는 달리, 여기서의 출력은 (i) CNN 특징 맵 `Y`, (ii) 현재 스케일에서 `Y`를 사용해 생성된 앵커 박스, (iii) 이러한 앵커 박스에 대해 (`Y`를 기반으로) 예측된 클래스와 오프셋을 포함합니다.

```{.python .input}
#@tab mxnet
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

:numref:`fig_ssd`에서 위쪽에 더 가까운 멀티스케일 특징 맵 블록이 더 큰 객체를 검출하기 위한 것임을 기억하세요. 따라서, 더 큰 앵커 박스를 생성해야 합니다.
위의 순전파에서, 각 멀티스케일 특징 맵 블록에서 저희는 호출된 `multibox_prior` 함수(:numref:`sec_anchor`에서 설명)의 `sizes` 인자를 통해 두 스케일 값의 목록을 전달합니다.
다음에서, 0.2와 1.05 사이의 구간은 다섯 블록에서 더 작은 스케일 값을 결정하기 위해 다섯 부분으로 균등하게 분할됩니다. 0.2, 0.37, 0.54, 0.71, 0.88입니다.
그런 다음 더 큰 스케일 값은 $\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$ 등으로 주어집니다.

[~~각 블록에 대한 하이퍼파라미터~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

이제 저희는 다음과 같이 [**완전한 모델**] `TinySSD`를 정의할 수 있습니다.

```{.python .input}
#@tab mxnet
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

저희는 [**모델 인스턴스를 생성하고 이를 사용해**] $256 \times 256$ 이미지 `X`의 미니배치에 대해 [**순전파를 수행**]합니다.

이 절의 앞부분에 표시된 것처럼, 첫 번째 블록은 $32 \times 32$ 특징 맵을 출력합니다.
두 번째에서 네 번째 다운샘플링 블록은 높이와 너비를 절반으로 줄이고 다섯 번째 블록은 전역 풀링을 사용한다는 것을 기억하세요.
특징 맵의 공간 차원을 따라 각 단위에 대해 4개의 앵커 박스가 생성되므로, 다섯 스케일 모두에서 각 이미지에 대해 총 $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$개의 앵커 박스가 생성됩니다.

```{.python .input}
#@tab mxnet
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 훈련

이제 저희는 객체 검출을 위한 SSD 모델을 훈련하는 방법을 설명할 것입니다.


### 데이터셋 읽기 및 모델 초기화

먼저, :numref:`sec_object-detection-dataset`에서 설명된 [**바나나 검출 데이터셋을 읽어**] 봅시다.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

바나나 검출 데이터셋에는 클래스가 하나만 있습니다. 모델을 정의한 후, 저희는 (**그 매개변수를 초기화하고 최적화 알고리즘을 정의**)해야 합니다.

```{.python .input}
#@tab mxnet
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**손실 및 평가 함수 정의**]

객체 검출에는 두 가지 유형의 손실이 있습니다.
첫 번째 손실은 앵커 박스의 클래스와 관련됩니다. 이 계산은 단순히 이미지 분류에서 저희가 사용한 교차 엔트로피 손실 함수를 재사용할 수 있습니다.
두 번째 손실은 양성(비배경) 앵커 박스의 오프셋과 관련됩니다. 이는 회귀 문제입니다.
하지만 이 회귀 문제의 경우, 여기서 저희는 :numref:`subsec_normal_distribution_and_squared_loss`에서 설명된 제곱 손실을 사용하지 않습니다.
대신, 저희는 $\ell_1$ 노름 손실, 즉 예측과 실측값 사이의 차이의 절댓값을 사용합니다.
마스크 변수 `bbox_masks`는 손실 계산에서 음성 앵커 박스와 유효하지 않은(패딩된) 앵커 박스를 필터링합니다.
마지막으로, 저희는 앵커 박스 클래스 손실과 앵커 박스 오프셋 손실을 합산해 모델에 대한 손실 함수를 얻습니다.

```{.python .input}
#@tab mxnet
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

저희는 분류 결과를 평가하는 데 정확도를 사용할 수 있습니다.
오프셋에 사용된 $\ell_1$ 노름 손실 때문에, 저희는 예측된 바운딩 박스를 평가하는 데 *평균 절대 오차*를 사용합니다.
이러한 예측 결과는 생성된 앵커 박스와 그에 대한 예측된 오프셋으로부터 얻어집니다.

```{.python .input}
#@tab mxnet
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**모델 훈련**]

모델을 훈련할 때, 저희는 순전파에서 멀티스케일 앵커 박스(`anchors`)를 생성하고 그 클래스(`cls_preds`)와 오프셋(`bbox_preds`)을 예측해야 합니다.
그런 다음 저희는 라벨 정보 `Y`를 기반으로 그러한 생성된 앵커 박스의 클래스(`cls_labels`)와 오프셋(`bbox_labels`)을 라벨링합니다.
마지막으로, 저희는 클래스와 오프셋의 예측 값과 라벨 값을 사용해 손실 함수를 계산합니다.
간결한 구현을 위해, 여기서는 테스트 데이터셋의 평가는 생략됩니다.

```{.python .input}
#@tab mxnet
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**예측**]

예측 중에, 목표는 이미지에서 모든 관심 객체를 검출하는 것입니다.
아래에서 저희는 테스트 이미지를 읽고 크기를 조정해 합성곱 계층에서 필요로 하는 4차원 텐서로 변환합니다.

```{.python .input}
#@tab mxnet
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

아래의 `multibox_detection` 함수를 사용해, 예측된 바운딩 박스는 앵커 박스와 그 예측된 오프셋으로부터 얻어집니다.
그런 다음 비최대 억제가 비슷한 예측된 바운딩 박스를 제거하는 데 사용됩니다.

```{.python .input}
#@tab mxnet
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

마지막으로, 저희는 [**신뢰도가 0.9 이상인 모든 예측된 바운딩 박스를 표시**]하여 출력합니다.

```{.python .input}
#@tab mxnet
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## 요약

* SSD는 멀티스케일 객체 검출 모델입니다. 기본 신경망과 여러 멀티스케일 특징 맵 블록을 통해, SSD는 다양한 크기의 다양한 수의 앵커 박스를 생성하고, 이러한 앵커 박스(그리고 따라서 바운딩 박스)의 클래스와 오프셋을 예측함으로써 다양한 크기의 객체를 검출합니다.
* SSD 모델을 훈련할 때, 손실 함수는 앵커 박스 클래스와 오프셋의 예측 값과 라벨 값을 기반으로 계산됩니다.



## 연습문제

1. 손실 함수를 개선해 SSD를 개선할 수 있나요? 예를 들어, 예측된 오프셋에 대해 $\ell_1$ 노름 손실을 smooth $\ell_1$ 노름 손실로 대체해 보세요. 이 손실 함수는 부드러움을 위해 0 주변에서 제곱 함수를 사용하며, 이는 하이퍼파라미터 $\sigma$에 의해 제어됩니다.

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \textrm{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \textrm{otherwise}
    \end{cases}
$$

$\sigma$가 매우 클 때, 이 손실은 $\ell_1$ 노름 손실과 유사합니다. 그 값이 더 작을 때, 손실 함수는 더 부드럽습니다.

```{.python .input}
#@tab mxnet
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

게다가, 실험에서 저희는 클래스 예측에 교차 엔트로피 손실을 사용했습니다. 실측 클래스 $j$에 대한 예측 확률을 $p_j$로 표기할 때, 교차 엔트로피 손실은 $-\log p_j$입니다. 저희는 또한 focal loss :cite:`Lin.Goyal.Girshick.ea.2017`를 사용할 수 있습니다. 하이퍼파라미터 $\gamma > 0$과 $\alpha > 0$이 주어졌을 때, 이 손실은 다음과 같이 정의됩니다.

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

보시다시피, $\gamma$를 증가시키는 것은 잘 분류된 예제(예: $p_j > 0.5$)에 대한 상대적인 손실을 효과적으로 줄일 수 있어, 훈련이 잘못 분류된 어려운 예제에 더 집중할 수 있습니다.

```{.python .input}
#@tab mxnet
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. 공간 제약 때문에, 저희는 이 절에서 SSD 모델의 일부 구현 세부사항을 생략했습니다. 다음 측면에서 모델을 추가로 개선할 수 있나요:
    1. 객체가 이미지에 비해 훨씬 작을 때, 모델이 입력 이미지의 크기를 더 크게 조정할 수 있습니다.
    1. 일반적으로 음성 앵커 박스가 방대한 수로 있습니다. 클래스 분포를 더 균형 있게 만들기 위해, 저희는 음성 앵커 박스를 다운샘플링할 수 있습니다.
    1. 손실 함수에서, 클래스 손실과 오프셋 손실에 다른 가중치 하이퍼파라미터를 할당해 보세요.
    1. SSD 논문 :cite:`Liu.Anguelov.Erhan.ea.2016`의 방법과 같은 다른 방법을 사용해 객체 검출 모델을 평가해 보세요.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:
