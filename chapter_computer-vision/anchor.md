# 앵커 박스
:label:`sec_anchor`


객체 검출 알고리즘은 보통 입력 이미지에서 많은 수의 영역을 샘플링하고, 이 영역이 관심 객체를 포함하는지 결정한 다음, 객체의 *실측(ground-truth) 바운딩 박스*를 보다 정확하게 예측하기 위해 영역의 경계를 조정합니다.
다양한 모델은 다양한 영역 샘플링 방식을 채택할 수 있습니다.
여기서는 그러한 방법 중 하나를 소개합니다. 이는 각 픽셀을 중심으로 다양한 스케일과 종횡비의 여러 바운딩 박스를 생성합니다.
이러한 바운딩 박스들을 *앵커 박스*라고 부릅니다.
저희는 :numref:`sec_ssd`에서 앵커 박스 기반의 객체 검출 모델을 설계할 것입니다.

먼저, 더 간결한 출력을 위해 인쇄 정확도를 수정해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## 여러 앵커 박스 생성

입력 이미지의 높이가 $h$이고 너비가 $w$라고 가정합니다.
저희는 이미지의 각 픽셀을 중심으로 다양한 형태의 앵커 박스를 생성합니다.
*스케일*을 $s\in (0, 1]$이라 하고, *종횡비*(너비와 높이의 비율)는 $r > 0$이라고 합시다.
그러면 [**앵커 박스의 너비와 높이는 각각 $ws\sqrt{r}$과 $hs/\sqrt{r}$입니다.**]
중심 위치가 주어질 때, 알려진 너비와 높이를 가진 앵커 박스가 결정된다는 점에 유의하세요.

다양한 형태의 여러 앵커 박스를 생성하기 위해, 일련의 스케일 $s_1,\ldots, s_n$과 일련의 종횡비 $r_1,\ldots, r_m$을 설정해 보겠습니다.
이러한 스케일과 종횡비의 모든 조합을 각 픽셀을 중심으로 사용할 때, 입력 이미지는 총 $whnm$개의 앵커 박스를 가지게 됩니다. 이러한 앵커 박스들이 모든 실측 바운딩 박스를 커버할 수도 있지만, 계산 복잡도가 너무 높아지기 쉽습니다.
실제로는 $s_1$ 또는 $r_1$을 (**포함하는 조합들만 고려**)할 수 있습니다.

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**)

즉, 같은 픽셀을 중심으로 하는 앵커 박스의 수는 $n+m-1$입니다. 전체 입력 이미지에 대해 저희는 총 $wh(n+m-1)$개의 앵커 박스를 생성합니다.

앵커 박스를 생성하는 위의 방법은 다음 `multibox_prior` 함수에 구현되어 있습니다. 입력 이미지, 스케일 목록, 종횡비 목록을 지정하면, 이 함수가 모든 앵커 박스를 반환합니다.

```{.python .input}
#@tab mxnet
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

[**반환된 앵커 박스 변수 `Y`의 형태**]가 (배치 크기, 앵커 박스 수, 4)임을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

앵커 박스 변수 `Y`의 형태를 (이미지 높이, 이미지 너비, 같은 픽셀을 중심으로 하는 앵커 박스 수, 4)로 변경한 후에는, 지정된 픽셀 위치를 중심으로 하는 모든 앵커 박스를 얻을 수 있습니다.
이어서, 저희는 [**(250, 250)을 중심으로 하는 첫 번째 앵커 박스에 접근**]합니다. 여기에는 네 가지 요소가 있습니다. 앵커 박스의 좌상단 꼭짓점의 $(x, y)$축 좌표와 우하단 꼭짓점의 $(x, y)$축 좌표입니다.
두 축의 좌표 값은 각각 이미지의 너비와 높이로 나누어집니다.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

[**이미지에서 한 픽셀을 중심으로 하는 모든 앵커 박스를 표시**]하기 위해, 이미지에 여러 바운딩 박스를 그리는 다음 `show_bboxes` 함수를 정의합니다.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

방금 본 것처럼, 변수 `boxes`의 $x$축과 $y$축의 좌표 값은 각각 이미지의 너비와 높이로 나누어졌습니다.
앵커 박스를 그릴 때는 원래의 좌표 값으로 복원해야 합니다. 따라서, 아래에서 변수 `bbox_scale`을 정의합니다.
이제, 이미지에서 (250, 250)을 중심으로 하는 모든 앵커 박스를 그릴 수 있습니다.
보시다시피, 스케일이 0.75이고 종횡비가 1인 파란색 앵커 박스가 이미지의 개를 잘 둘러싸고 있습니다.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**IoU(Intersection over Union)**]

저희는 방금 앵커 박스가 이미지의 개를 "잘" 둘러싼다고 언급했습니다.
객체의 실측 바운딩 박스가 알려져 있다면, 여기서 "잘"을 어떻게 정량화할 수 있을까요?
직관적으로, 저희는 앵커 박스와 실측 바운딩 박스 사이의 유사성을 측정할 수 있습니다.
저희는 *자카드 지수(Jaccard index)*가 두 집합 사이의 유사성을 측정할 수 있다는 것을 알고 있습니다. 집합 $\mathcal{A}$와 $\mathcal{B}$가 주어졌을 때, 그들의 자카드 지수는 교집합의 크기를 합집합의 크기로 나눈 것입니다.

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$


실제로 저희는 어떤 바운딩 박스의 픽셀 영역을 픽셀의 집합으로 간주할 수 있습니다.
이런 식으로 저희는 픽셀 집합의 자카드 지수로 두 바운딩 박스의 유사성을 측정할 수 있습니다. 두 바운딩 박스에 대해, 저희는 보통 그들의 자카드 지수를 *IoU(intersection over union)*라고 부르는데, 이는 :numref:`fig_iou`에 표시된 것처럼 교집합 영역과 합집합 영역의 비율입니다.
IoU의 범위는 0과 1 사이입니다.
0은 두 바운딩 박스가 전혀 겹치지 않음을 의미하고, 1은 두 바운딩 박스가 동일함을 나타냅니다.

![IoU는 두 바운딩 박스의 합집합 영역에 대한 교집합 영역의 비율입니다.](../img/iou.svg)
:label:`fig_iou`

이 절의 나머지 부분에서, 저희는 IoU를 사용해 앵커 박스와 실측 바운딩 박스 사이, 그리고 다양한 앵커 박스 사이의 유사성을 측정합니다.
두 앵커 또는 바운딩 박스 목록이 주어졌을 때, 다음 `box_iou`는 이 두 목록에 걸쳐 그들의 쌍별 IoU를 계산합니다.

```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## 훈련 데이터에서 앵커 박스 라벨링
:label:`subsec_labeling-anchor-boxes`


훈련 데이터셋에서 저희는 각 앵커 박스를 하나의 훈련 예제로 간주합니다.
객체 검출 모델을 훈련하기 위해, 저희는 각 앵커 박스에 대해 *클래스*와 *오프셋* 라벨이 필요합니다. 여기서 전자는 앵커 박스와 관련된 객체의 클래스이며 후자는 앵커 박스에 대한 실측 바운딩 박스의 오프셋입니다.
예측 중에는, 각 이미지에 대해 여러 앵커 박스를 생성하고, 모든 앵커 박스에 대해 클래스와 오프셋을 예측하고, 예측된 오프셋에 따라 위치를 조정해 예측된 바운딩 박스를 얻고, 마지막으로 특정 기준을 충족하는 예측된 바운딩 박스만 출력합니다.


저희가 알고 있듯이, 객체 검출 훈련 셋에는 *실측 바운딩 박스*의 위치와 그것이 둘러싼 객체의 클래스에 대한 라벨이 함께 제공됩니다.
생성된 *앵커 박스*에 라벨을 붙이기 위해, 저희는 앵커 박스와 가장 가까운 *할당된* 실측 바운딩 박스의 라벨된 위치와 클래스를 참조합니다.
이어서, 저희는 앵커 박스에 가장 가까운 실측 바운딩 박스를 할당하기 위한 알고리즘을 설명합니다.

### [**앵커 박스에 실측 바운딩 박스 할당**]

이미지가 주어졌을 때, 앵커 박스가 $A_1, A_2, \ldots, A_{n_a}$이고 실측 바운딩 박스가 $B_1, B_2, \ldots, B_{n_b}$라고 가정합니다. 여기서 $n_a \geq n_b$입니다.
$i^\textrm{th}$ 행과 $j^\textrm{th}$ 열의 원소 $x_{ij}$가 앵커 박스 $A_i$와 실측 바운딩 박스 $B_j$의 IoU인 행렬 $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$를 정의합시다. 알고리즘은 다음 단계로 구성됩니다.

1. 행렬 $\mathbf{X}$에서 가장 큰 원소를 찾고 그 행 인덱스와 열 인덱스를 각각 $i_1$과 $j_1$로 표기합니다. 그러면 실측 바운딩 박스 $B_{j_1}$이 앵커 박스 $A_{i_1}$에 할당됩니다. 이는 $A_{i_1}$과 $B_{j_1}$이 모든 앵커 박스와 실측 바운딩 박스 쌍 중에서 가장 가깝기 때문에 매우 직관적입니다. 첫 번째 할당 후, 행렬 $\mathbf{X}$의 ${i_1}^\textrm{th}$ 행과 ${j_1}^\textrm{th}$ 열에 있는 모든 원소를 폐기합니다.
1. 행렬 $\mathbf{X}$의 남은 원소들 중에서 가장 큰 것을 찾고 그 행 인덱스와 열 인덱스를 각각 $i_2$와 $j_2$로 표기합니다. 실측 바운딩 박스 $B_{j_2}$를 앵커 박스 $A_{i_2}$에 할당하고 행렬 $\mathbf{X}$의 ${i_2}^\textrm{th}$ 행과 ${j_2}^\textrm{th}$ 열에 있는 모든 원소를 폐기합니다.
1. 이 시점에서, 행렬 $\mathbf{X}$의 두 행과 두 열의 원소들이 폐기되었습니다. 행렬 $\mathbf{X}$의 $n_b$ 열의 모든 원소가 폐기될 때까지 계속 진행합니다. 이때, 저희는 $n_b$개의 앵커 박스 각각에 실측 바운딩 박스를 할당하게 됩니다.
1. 나머지 $n_a - n_b$개의 앵커 박스만 순회합니다. 예를 들어, 어떤 앵커 박스 $A_i$가 주어졌을 때, 행렬 $\mathbf{X}$의 $i^\textrm{th}$ 행 전체에서 $A_i$와 가장 큰 IoU를 갖는 실측 바운딩 박스 $B_j$를 찾고, 이 IoU가 미리 정의된 임계값보다 큰 경우에만 $B_j$를 $A_i$에 할당합니다.

위의 알고리즘을 구체적인 예제로 설명해 보겠습니다.
:numref:`fig_anchor_label` (왼쪽)에 표시된 것처럼, 행렬 $\mathbf{X}$의 최댓값이 $x_{23}$이라고 가정하면, 저희는 실측 바운딩 박스 $B_3$를 앵커 박스 $A_2$에 할당합니다.
그런 다음, 행렬의 2행과 3열의 모든 원소를 폐기하고, 남은 원소들(음영 영역) 중에서 가장 큰 $x_{71}$을 찾고, 실측 바운딩 박스 $B_1$을 앵커 박스 $A_7$에 할당합니다.
다음으로, :numref:`fig_anchor_label` (중간)에 표시된 것처럼, 행렬의 7행과 1열의 모든 원소를 폐기하고, 남은 원소들(음영 영역) 중에서 가장 큰 $x_{54}$를 찾고, 실측 바운딩 박스 $B_4$를 앵커 박스 $A_5$에 할당합니다.
마지막으로, :numref:`fig_anchor_label` (오른쪽)에 표시된 것처럼, 행렬의 5행과 4열의 모든 원소를 폐기하고, 남은 원소들(음영 영역) 중에서 가장 큰 $x_{92}$를 찾고, 실측 바운딩 박스 $B_2$를 앵커 박스 $A_9$에 할당합니다.
그 후, 저희는 나머지 앵커 박스 $A_1, A_3, A_4, A_6, A_8$만 순회하고 임계값에 따라 실측 바운딩 박스를 할당할지 결정하면 됩니다.

![앵커 박스에 실측 바운딩 박스 할당.](../img/anchor-label.svg)
:label:`fig_anchor_label`

이 알고리즘은 다음 `assign_anchor_to_bbox` 함수에 구현되어 있습니다.

```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### 클래스와 오프셋 라벨링

이제 저희는 각 앵커 박스에 대해 클래스와 오프셋에 라벨을 붙일 수 있습니다. 앵커 박스 $A$가 실측 바운딩 박스 $B$에 할당되었다고 가정합니다.
한편으로, 앵커 박스 $A$의 클래스는 $B$의 클래스로 라벨링됩니다.
다른 한편으로, 앵커 박스 $A$의 오프셋은 $B$와 $A$의 중심 좌표 사이의 상대적 위치와 이 두 박스 사이의 상대적 크기에 따라 라벨링됩니다.
데이터셋의 다양한 박스들이 다양한 위치와 크기를 가지므로, 저희는 이러한 상대적 위치와 크기에 변환을 적용하여 보다 균일하게 분포된, 적합시키기 더 쉬운 오프셋을 만들 수 있습니다.
여기서는 일반적인 변환을 설명합니다.
[**$A$와 $B$의 중심 좌표를 각각 $(x_a, y_a)$와 $(x_b, y_b)$, 너비를 $w_a$와 $w_b$, 높이를 $h_a$와 $h_b$로 두었을 때.
저희는 $A$의 오프셋을 다음과 같이 라벨링할 수 있습니다.

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
여기서 상수들의 기본 값은 $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, $\sigma_w=\sigma_h=0.2$입니다.
이 변환은 아래의 `offset_boxes` 함수에 구현되어 있습니다.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

만약 앵커 박스에 실측 바운딩 박스가 할당되지 않으면, 저희는 앵커 박스의 클래스를 "배경"으로 라벨링합니다.
배경 클래스인 앵커 박스는 종종 *음성* 앵커 박스라고 부르며, 나머지는 *양성* 앵커 박스라고 부릅니다.
저희는 실측 바운딩 박스(`labels` 인자)를 사용해 [**앵커 박스(`anchors` 인자)에 대한 클래스와 오프셋을 라벨링**]하기 위해 다음 `multibox_target` 함수를 구현합니다.
이 함수는 배경 클래스를 0으로 설정하고 새 클래스의 정수 인덱스를 1씩 증가시킵니다.

```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### 예제

구체적인 예제를 통해 앵커 박스 라벨링을 설명해 보겠습니다.
저희는 로드된 이미지에서 개와 고양이의 실측 바운딩 박스를 정의합니다. 여기서 첫 번째 요소는 클래스(개는 0, 고양이는 1)이고 나머지 네 요소는 좌상단 꼭짓점과 우하단 꼭짓점의 $(x, y)$축 좌표(범위는 0과 1 사이)입니다.
또한 좌상단 꼭짓점과 우하단 꼭짓점의 좌표를 사용해 라벨링될 다섯 개의 앵커 박스도 구성합니다. $A_0, \ldots, A_4$ (인덱스는 0부터 시작합니다).
그런 다음 [**이러한 실측 바운딩 박스와 앵커 박스를 이미지에 플롯합니다.**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

위에서 정의한 `multibox_target` 함수를 사용해, 저희는 개와 고양이의 [**실측 바운딩 박스를 기반으로 이러한 앵커 박스의 클래스와 오프셋을 라벨링**]할 수 있습니다.
이 예제에서, 배경, 개, 고양이 클래스의 인덱스는 각각 0, 1, 2입니다.
아래에서 저희는 앵커 박스와 실측 바운딩 박스의 예제에 대한 차원을 추가합니다.

```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

반환된 결과에는 세 가지 항목이 있는데, 모두 텐서 형식입니다.
세 번째 항목은 입력 앵커 박스의 라벨링된 클래스를 포함합니다.

이미지에서 앵커 박스와 실측 바운딩 박스의 위치를 기반으로 아래의 반환된 클래스 라벨을 분석해 보겠습니다.
먼저, 모든 앵커 박스와 실측 바운딩 박스의 쌍 중에서, 앵커 박스 $A_4$와 고양이의 실측 바운딩 박스의 IoU가 가장 큽니다.
따라서, $A_4$의 클래스는 고양이로 라벨링됩니다.
$A_4$ 또는 고양이의 실측 바운딩 박스를 포함하는 쌍을 빼면, 나머지 중에서 앵커 박스 $A_1$과 개의 실측 바운딩 박스의 쌍이 가장 큰 IoU를 가집니다.
따라서 $A_1$의 클래스는 개로 라벨링됩니다.
다음으로, 저희는 라벨링되지 않은 나머지 세 개의 앵커 박스 $A_0$, $A_2$, $A_3$를 순회해야 합니다.
$A_0$의 경우, IoU가 가장 큰 실측 바운딩 박스의 클래스는 개이지만, IoU가 미리 정의된 임계값(0.5) 아래이므로, 클래스는 배경으로 라벨링됩니다.
$A_2$의 경우, IoU가 가장 큰 실측 바운딩 박스의 클래스는 고양이이고 IoU가 임계값을 초과하므로, 클래스는 고양이로 라벨링됩니다.
$A_3$의 경우, IoU가 가장 큰 실측 바운딩 박스의 클래스는 고양이이지만, 값이 임계값 아래이므로, 클래스는 배경으로 라벨링됩니다.

```{.python .input}
#@tab all
labels[2]
```

두 번째 반환된 항목은 (배치 크기, 앵커 박스 수의 네 배) 형태의 마스크 변수입니다.
마스크 변수의 매 네 개의 원소는 각 앵커 박스의 네 오프셋 값에 해당합니다.
배경 검출에 대해서는 신경 쓰지 않으므로, 이 음성 클래스의 오프셋은 목적 함수에 영향을 미치지 않아야 합니다.
원소별 곱셈을 통해, 마스크 변수의 0은 목적 함수를 계산하기 전에 음성 클래스 오프셋을 필터링합니다.

```{.python .input}
#@tab all
labels[1]
```

첫 번째 반환된 항목에는 각 앵커 박스에 대해 라벨링된 네 오프셋 값이 포함됩니다.
음성 클래스 앵커 박스의 오프셋은 0으로 라벨링됨에 유의하세요.

```{.python .input}
#@tab all
labels[0]
```

## 비최대 억제를 사용한 바운딩 박스 예측
:label:`subsec_predicting-bounding-boxes-nms`

예측 중에, 저희는 이미지에 대해 여러 앵커 박스를 생성하고 그 각각에 대해 클래스와 오프셋을 예측합니다.
*예측된 바운딩 박스*는 따라서 예측된 오프셋과 함께 앵커 박스에 따라 얻어집니다.
아래에서 저희는 앵커와 오프셋 예측을 입력으로 받아 [**역 오프셋 변환을 적용하여 예측된 바운딩 박스 좌표를 반환**]하는 `offset_inverse` 함수를 구현합니다.

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

많은 앵커 박스가 있을 때, 같은 객체를 둘러싸기 위해 비슷한(상당히 겹치는) 예측된 바운딩 박스가 많이 출력될 수 있습니다.
출력을 단순화하기 위해, 저희는 *비최대 억제(non-maximum suppression, NMS)*를 사용해 같은 객체에 속하는 비슷한 예측된 바운딩 박스들을 병합할 수 있습니다.

비최대 억제는 다음과 같이 작동합니다.
예측된 바운딩 박스 $B$에 대해, 객체 검출 모델은 각 클래스에 대한 예측 가능도(likelihood)를 계산합니다.
가장 큰 예측 가능도를 $p$로 표기할 때, 이 확률에 해당하는 클래스가 $B$에 대한 예측된 클래스입니다.
구체적으로, 저희는 $p$를 예측된 바운딩 박스 $B$의 *신뢰도*(점수)라고 부릅니다.
같은 이미지에서, 모든 예측된 비배경 바운딩 박스는 신뢰도를 내림차순으로 정렬해 목록 $L$을 생성합니다.
그런 다음 저희는 정렬된 목록 $L$을 다음 단계로 조작합니다.

1. $L$에서 신뢰도가 가장 높은 예측된 바운딩 박스 $B_1$을 기준으로 선택하고, $B_1$과의 IoU가 미리 정의된 임계값 $\epsilon$을 초과하는 모든 기준이 아닌 예측된 바운딩 박스를 $L$에서 제거합니다. 이 시점에서, $L$은 가장 높은 신뢰도의 예측된 바운딩 박스는 유지하지만 너무 비슷한 다른 것들은 버립니다. 요컨대, *비최대* 신뢰도 점수를 가진 것들이 *억제*됩니다.
1. $L$에서 두 번째로 높은 신뢰도의 예측된 바운딩 박스 $B_2$를 또 다른 기준으로 선택하고, $B_2$와의 IoU가 $\epsilon$을 초과하는 모든 기준이 아닌 예측된 바운딩 박스를 $L$에서 제거합니다.
1. $L$의 모든 예측된 바운딩 박스가 기준으로 사용될 때까지 위의 과정을 반복합니다. 이때, $L$의 예측된 바운딩 박스 쌍의 IoU는 임계값 $\epsilon$ 아래이므로, 어떤 쌍도 서로 너무 비슷하지 않습니다.
1. 목록 $L$의 모든 예측된 바운딩 박스를 출력합니다.

[**다음 `nms` 함수는 신뢰도 점수를 내림차순으로 정렬하고 그 인덱스를 반환합니다.**]

```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

[**바운딩 박스 예측에 비최대 억제를 적용**]하기 위해 다음 `multibox_detection`을 정의합니다.
구현이 약간 복잡하다고 느껴진다면 걱정하지 마세요. 구현 바로 다음에 구체적인 예제로 어떻게 작동하는지 보여드리겠습니다.

```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

이제 [**위의 구현을 네 개의 앵커 박스가 있는 구체적인 예제에 적용**]해 보겠습니다.
단순화를 위해, 저희는 예측된 오프셋이 모두 0이라고 가정합니다.
이는 예측된 바운딩 박스가 앵커 박스임을 의미합니다.
배경, 개, 고양이 각 클래스에 대한 예측 가능도도 정의합니다.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

[**이러한 예측된 바운딩 박스를 이미지에 신뢰도와 함께 플롯**]할 수 있습니다.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

이제 저희는 `multibox_detection` 함수를 호출해 비최대 억제를 수행할 수 있는데, 여기서 임계값은 0.5로 설정합니다.
텐서 입력의 예제에 대한 차원을 추가한다는 점에 유의하세요.

[**반환된 결과의 형태**]가 (배치 크기, 앵커 박스 수, 6)임을 볼 수 있습니다.
가장 안쪽 차원의 여섯 원소는 같은 예측된 바운딩 박스에 대한 출력 정보를 제공합니다.
첫 번째 원소는 예측된 클래스 인덱스로, 0부터 시작합니다(0은 개이고 1은 고양이입니다). 값 -1은 배경 또는 비최대 억제에서의 제거를 나타냅니다.
두 번째 원소는 예측된 바운딩 박스의 신뢰도입니다.
나머지 네 원소는 각각 예측된 바운딩 박스의 좌상단 꼭짓점과 우하단 꼭짓점의 $(x, y)$축 좌표입니다(범위는 0과 1 사이).

```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

클래스 -1의 예측된 바운딩 박스를 제거한 후, 저희는 [**비최대 억제에 의해 유지된 최종 예측된 바운딩 박스를 출력**]할 수 있습니다.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

실제로 저희는 비최대 억제를 수행하기 전에 신뢰도가 더 낮은 예측된 바운딩 박스를 제거하여 이 알고리즘에서 계산을 줄일 수 있습니다.
또한 비최대 억제의 출력을 후처리할 수도 있습니다. 예를 들어, 최종 출력에서 신뢰도가 더 높은 결과만 유지하는 식입니다.


## 요약

* 저희는 이미지의 각 픽셀을 중심으로 다양한 형태의 앵커 박스를 생성합니다.
* IoU(Intersection over Union)는 자카드 지수(Jaccard index)라고도 알려져 있으며, 두 바운딩 박스의 유사성을 측정합니다. 이는 그들의 합집합 영역에 대한 교집합 영역의 비율입니다.
* 훈련 셋에서, 저희는 각 앵커 박스에 대해 두 가지 유형의 라벨이 필요합니다. 하나는 앵커 박스와 관련된 객체의 클래스이고 다른 하나는 앵커 박스에 대한 실측 바운딩 박스의 오프셋입니다.
* 예측 중에, 저희는 비최대 억제(NMS)를 사용해 비슷한 예측된 바운딩 박스를 제거함으로써 출력을 단순화할 수 있습니다.


## 연습문제

1. `multibox_prior` 함수의 `sizes`와 `ratios` 값을 변경해 보세요. 생성된 앵커 박스에 어떤 변화가 있나요?
1. IoU가 0.5인 두 개의 바운딩 박스를 구성하고 시각화해 보세요. 그들이 서로 어떻게 겹치나요?
1. :numref:`subsec_labeling-anchor-boxes`와 :numref:`subsec_predicting-bounding-boxes-nms`의 변수 `anchors`를 수정해 보세요. 결과가 어떻게 바뀌나요?
1. 비최대 억제는 예측된 바운딩 박스를 *제거*함으로써 억제하는 탐욕적 알고리즘입니다. 이러한 제거된 것들 중 일부가 실제로 유용할 가능성이 있을까요? 이 알고리즘을 *부드럽게* 억제하도록 어떻게 수정할 수 있을까요? Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`를 참조하세요.
1. 비최대 억제가 수작업으로 만들어진 것이 아니라 학습될 수 있을까요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:
