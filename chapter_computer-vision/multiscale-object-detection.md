# 멀티스케일 객체 검출
:label:`sec_multiscale-object-detection`


:numref:`sec_anchor`에서 저희는 입력 이미지의 각 픽셀을 중심으로 여러 앵커 박스를 생성했습니다.
본질적으로 이러한 앵커 박스는 이미지의 다양한 영역의 샘플을 나타냅니다.
하지만, *모든* 픽셀에 대해 앵커 박스가 생성되면 계산하기에 너무 많은 앵커 박스가 생기게 됩니다.
$561 \times 728$ 입력 이미지를 생각해 보세요.
각 픽셀을 중심으로 다양한 형태의 다섯 개의 앵커 박스가 생성되면, 200만 개가 넘는 앵커 박스($561 \times 728 \times 5$)가 이미지에 라벨링되고 예측되어야 합니다.

## 멀티스케일 앵커 박스
:label:`subsec_multiscale-anchor-boxes`

이미지의 앵커 박스를 줄이는 것은 어렵지 않다는 것을 깨달으실 수 있습니다.
예를 들어, 저희는 단순히 입력 이미지에서 픽셀의 작은 부분을 균등하게 샘플링해 그것들을 중심으로 앵커 박스를 생성할 수 있습니다.
또한, 다양한 스케일에서 다양한 크기의 앵커 박스를 다양한 수로 생성할 수 있습니다.
직관적으로, 작은 객체가 큰 객체보다 이미지에 나타날 가능성이 더 큽니다.
예를 들어, $1 \times 1$, $1 \times 2$, $2 \times 2$ 객체는 $2 \times 2$ 이미지에 각각 4가지, 2가지, 1가지 가능한 방식으로 나타날 수 있습니다.
따라서, 더 작은 객체를 검출하기 위해 더 작은 앵커 박스를 사용할 때, 저희는 더 많은 영역을 샘플링할 수 있으며, 더 큰 객체에 대해서는 더 적은 영역을 샘플링할 수 있습니다.

여러 스케일에서 앵커 박스를 생성하는 방법을 시연하기 위해, 이미지 하나를 읽어보겠습니다.
이미지의 높이와 너비는 각각 561픽셀과 728픽셀입니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

:numref:`sec_conv_layer`에서 저희는 합성곱 계층의 2차원 배열 출력을 특징 맵이라고 부른다는 것을 기억하세요.
특징 맵 형태를 정의함으로써, 저희는 어떤 이미지에서도 균등하게 샘플링된 앵커 박스의 중심을 결정할 수 있습니다.


아래에 `display_anchors` 함수가 정의되어 있습니다.
[**저희는 각 단위(픽셀)를 앵커 박스 중심으로 사용해 특징 맵(`fmap`)에서 앵커 박스(`anchors`)를 생성합니다.**]
앵커 박스(`anchors`)의 $(x, y)$축 좌표 값이 특징 맵(`fmap`)의 너비와 높이로 나누어졌기 때문에, 이러한 값은 0과 1 사이이며, 이는 특징 맵에서 앵커 박스의 상대적 위치를 나타냅니다.

앵커 박스(`anchors`)의 중심이 특징 맵(`fmap`)의 모든 단위에 걸쳐 퍼져 있기 때문에, 이러한 중심은 상대적인 공간적 위치 측면에서 어떤 입력 이미지에서도 *균등하게* 분포되어야 합니다.
보다 구체적으로, 특징 맵의 너비와 높이가 각각 `fmap_w`와 `fmap_h`로 주어졌을 때, 다음 함수는 어떤 입력 이미지에서도 `fmap_h` 행과 `fmap_w` 열의 픽셀들을 *균등하게* 샘플링합니다.
이러한 균등하게 샘플링된 픽셀을 중심으로, 스케일 `s`(목록 `s`의 길이가 1이라고 가정)와 다양한 종횡비(`ratios`)의 앵커 박스가 생성됩니다.

```{.python .input}
#@tab mxnet
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

먼저, [**작은 객체의 검출을 고려**]해 봅시다.
표시될 때 구별하기 더 쉽도록, 여기서 다른 중심을 가진 앵커 박스는 겹치지 않습니다. 앵커 박스 스케일은 0.15로 설정되고 특징 맵의 높이와 너비는 4로 설정됩니다. 저희는 이미지의 4행 4열의 앵커 박스 중심이 균등하게 분포되어 있는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

다음으로 [**특징 맵의 높이와 너비를 절반으로 줄이고 더 큰 객체를 검출하기 위해 더 큰 앵커 박스를 사용**]합니다. 스케일을 0.4로 설정하면, 일부 앵커 박스가 서로 겹칠 것입니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

마지막으로, 저희는 [**특징 맵의 높이와 너비를 추가로 절반으로 줄이고 앵커 박스 스케일을 0.8로 증가**]시킵니다. 이제 앵커 박스의 중심은 이미지의 중심입니다.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## 멀티스케일 검출


멀티스케일 앵커 박스를 생성했으므로, 저희는 이를 사용해 다양한 스케일에서 다양한 크기의 객체를 검출할 것입니다.
다음에서 저희는 :numref:`sec_ssd`에서 구현할 CNN 기반 멀티스케일 객체 검출 방법을 소개합니다.

어떤 스케일에서, 저희가 형태 $h \times w$의 $c$개의 특징 맵을 가지고 있다고 가정합시다.
:numref:`subsec_multiscale-anchor-boxes`의 방법을 사용해, 저희는 $hw$개의 앵커 박스 집합을 생성합니다. 여기서 각 집합은 같은 중심을 가진 $a$개의 앵커 박스를 가집니다.
예를 들어, :numref:`subsec_multiscale-anchor-boxes`의 실험에서 첫 번째 스케일에서, 10개(채널 수)의 $4 \times 4$ 특징 맵이 주어졌을 때, 저희는 16개의 앵커 박스 집합을 생성했는데, 여기서 각 집합은 같은 중심을 가진 3개의 앵커 박스를 포함합니다.
다음으로, 각 앵커 박스는 실측 바운딩 박스를 기반으로 클래스와 오프셋으로 라벨링됩니다. 현재 스케일에서, 객체 검출 모델은 입력 이미지에 대해 $hw$개의 앵커 박스 집합의 클래스와 오프셋을 예측해야 하는데, 여기서 다른 집합은 다른 중심을 가집니다.


여기서 $c$개의 특징 맵이 입력 이미지를 기반으로 한 CNN 순전파에 의해 얻어진 중간 출력이라고 가정합니다. 각 특징 맵에 $hw$개의 다른 공간적 위치가 있기 때문에, 같은 공간적 위치는 $c$개의 단위를 가진다고 생각할 수 있습니다.
:numref:`sec_conv_layer`의 수용 영역의 정의에 따라, 특징 맵의 같은 공간적 위치에 있는 이러한 $c$개의 단위는 입력 이미지에서 같은 수용 영역을 가집니다. 즉, 같은 수용 영역에 있는 입력 이미지 정보를 나타냅니다.
따라서, 저희는 특징 맵의 같은 공간적 위치에 있는 $c$개의 단위를 이 공간적 위치를 사용해 생성된 $a$개의 앵커 박스의 클래스와 오프셋으로 변환할 수 있습니다.
본질적으로, 저희는 어떤 수용 영역에 있는 입력 이미지의 정보를 사용해 입력 이미지에서 그 수용 영역에 가까운 앵커 박스의 클래스와 오프셋을 예측합니다.


다른 계층의 특징 맵이 입력 이미지에서 다른 크기의 수용 영역을 가질 때, 그것들은 다른 크기의 객체를 검출하는 데 사용될 수 있습니다.
예를 들어, 저희는 출력 계층에 더 가까운 특징 맵의 단위가 더 넓은 수용 영역을 가지도록 신경망을 설계할 수 있는데, 그러면 입력 이미지에서 더 큰 객체를 검출할 수 있습니다.

요컨대, 저희는 멀티스케일 객체 검출을 위해 심층 신경망에 의한 이미지의 여러 수준에서의 계층별 표현을 활용할 수 있습니다.
:numref:`sec_ssd`에서 구체적인 예제를 통해 이것이 어떻게 작동하는지 보여드리겠습니다.




## 요약

* 여러 스케일에서, 저희는 다양한 크기의 객체를 검출하기 위해 다양한 크기의 앵커 박스를 생성할 수 있습니다.
* 특징 맵의 형태를 정의함으로써, 저희는 어떤 이미지에서도 균등하게 샘플링된 앵커 박스의 중심을 결정할 수 있습니다.
* 저희는 어떤 수용 영역에 있는 입력 이미지의 정보를 사용해 입력 이미지에서 그 수용 영역에 가까운 앵커 박스의 클래스와 오프셋을 예측합니다.
* 딥러닝을 통해, 저희는 멀티스케일 객체 검출을 위해 여러 수준에서의 이미지의 계층별 표현을 활용할 수 있습니다.


## 연습문제

1. :numref:`sec_alexnet`의 논의에 따라, 심층 신경망은 이미지에 대해 점점 더 추상화 수준이 높아지는 계층적 특징을 학습합니다. 멀티스케일 객체 검출에서, 다른 스케일의 특징 맵이 다른 수준의 추상화에 해당하나요? 왜 그런가요, 아니면 왜 그렇지 않은가요?
1. :numref:`subsec_multiscale-anchor-boxes`의 실험에서 첫 번째 스케일(`fmap_w=4, fmap_h=4`)에서, 겹칠 수 있는 균등하게 분포된 앵커 박스를 생성해 보세요.
1. 형태 $1 \times c \times h \times w$의 특징 맵 변수가 주어졌다고 합시다. 여기서 $c$, $h$, $w$는 각각 특징 맵의 채널 수, 높이, 너비입니다. 이 변수를 어떻게 앵커 박스의 클래스와 오프셋으로 변환할 수 있을까요? 출력의 형태는 어떻게 되나요?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
