# 객체 검출과 바운딩 박스
:label:`sec_bbox`


앞 절들(예: :numref:`sec_alexnet`(:numref:`sec_googlenet`)에서 저희는 이미지 분류를 위한 다양한 모델들을 소개했습니다.
이미지 분류 작업에서는 이미지에 *하나의* 주요 객체만 존재한다고 가정하고, 그 카테고리를 어떻게 인식할 것인가에만 초점을 맞춥니다.
하지만 관심 있는 이미지에는 *여러* 객체가 있는 경우가 많습니다.
저희는 그 카테고리뿐만 아니라 이미지에서의 구체적인 위치도 알고 싶습니다.
컴퓨터 비전에서는 이러한 작업을 *객체 검출*(또는 *객체 인식*)이라고 부릅니다.

객체 검출은 많은 분야에서 폭넓게 적용되어 왔습니다.
예를 들어, 자율 주행은 캡처된 비디오 이미지에서 차량, 보행자, 도로, 장애물의 위치를 검출하여 주행 경로를 계획해야 합니다.
또한, 로봇은 환경을 탐색하는 과정에서 관심 객체를 검출하고 위치를 파악하기 위해 이 기법을 사용할 수 있습니다.
나아가, 보안 시스템은 침입자나 폭탄과 같은 비정상적인 객체를 검출해야 할 수도 있습니다.

다음 몇 절에서는 객체 검출을 위한 여러 딥러닝 방법을 소개합니다.
저희는 객체의 *위치*(또는 *지점*)에 대한 소개부터 시작하겠습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

이 절에서 사용할 샘플 이미지를 로드하겠습니다. 이미지 왼쪽에는 개가 있고 오른쪽에는 고양이가 있는 것을 볼 수 있습니다.
이 둘이 이미지의 두 가지 주요 객체입니다.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## 바운딩 박스


객체 검출에서는 일반적으로 *바운딩 박스*를 사용해 객체의 공간적 위치를 기술합니다.
바운딩 박스는 직사각형이며, 직사각형의 좌상단 꼭짓점의 $x$, $y$ 좌표와 우하단 꼭짓점의 같은 좌표로 결정됩니다.
또 다른 일반적으로 사용되는 바운딩 박스 표현은 바운딩 박스 중심의 $(x, y)$축 좌표와 박스의 너비 및 높이입니다.

[**여기서는 이 (**두 가지 표현**) 사이를 변환하는 함수들을 정의합니다**].
`box_corner_to_center`는 두 꼭짓점 표현에서 중심-너비-높이 표현으로 변환하고, `box_center_to_corner`는 그 반대입니다.
입력 인자 `boxes`는 ($n$, 4) 형태의 2차원 텐서여야 하며, 여기서 $n$은 바운딩 박스의 개수입니다.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

저희는 좌표 정보를 기반으로 [**이미지에서 개와 고양이의 바운딩 박스를 정의**]하겠습니다.
이미지에서 좌표의 원점은 이미지의 좌상단 꼭짓점이며, 오른쪽과 아래쪽이 각각 $x$축과 $y$축의 양의 방향입니다.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

두 번 변환해 봄으로써 두 바운딩 박스 변환 함수의 정확성을 검증할 수 있습니다.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

[**이미지에 바운딩 박스를 그려보고**] 정확한지 확인해 보겠습니다.
그리기 전에 헬퍼 함수 `bbox_to_rect`를 정의합니다. 이는 바운딩 박스를 `matplotlib` 패키지의 바운딩 박스 형식으로 표현합니다.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

이미지에 바운딩 박스를 추가하면 두 객체의 주요 윤곽이 기본적으로 두 박스 안에 들어 있는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## 요약

* 객체 검출은 이미지에서 관심 있는 모든 객체뿐만 아니라 그 위치까지 인식합니다. 위치는 일반적으로 직사각형 바운딩 박스로 표현됩니다.
* 일반적으로 사용되는 두 가지 바운딩 박스 표현 사이를 변환할 수 있습니다.

## 연습문제

1. 다른 이미지를 찾아서 객체를 포함하는 바운딩 박스를 라벨링해 보세요. 바운딩 박스와 카테고리를 라벨링하는 것을 비교해 보세요. 어느 쪽이 보통 더 오래 걸리나요?
1. `box_corner_to_center`와 `box_center_to_corner`의 입력 인자 `boxes`의 가장 안쪽 차원이 항상 4인 이유는 무엇인가요?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:
