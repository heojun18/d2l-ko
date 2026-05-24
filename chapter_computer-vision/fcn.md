# 완전 합성곱 신경망
:label:`sec_fcn`

:numref:`sec_semantic_segmentation`에서 논의했듯이, 시맨틱 분할은 이미지를 픽셀 수준에서 분류합니다.
완전 합성곱 신경망(FCN)은 이미지 픽셀을 픽셀 클래스로 변환하기 위해 합성곱 신경망을 사용합니다 :cite:`Long.Shelhamer.Darrell.2015`.
이전에 이미지 분류나 객체 검출을 위해 마주친 CNN과는 달리, 완전 합성곱 신경망은 중간 특징 맵의 높이와 너비를 입력 이미지의 그것으로 다시 변환합니다. 이는 :numref:`sec_transposed_conv`에서 소개된 전치 합성곱 계층에 의해 달성됩니다.
그 결과, 분류 출력과 입력 이미지가 픽셀 수준에서 일대일 대응을 가집니다. 임의의 출력 픽셀의 채널 차원이 같은 공간적 위치에 있는 입력 픽셀에 대한 분류 결과를 가집니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## 모델

여기서는 완전 합성곱 신경망 모델의 기본 설계를 설명합니다.
:numref:`fig_fcn`에 표시된 것처럼, 이 모델은 먼저 CNN을 사용해 이미지 특징을 추출한 다음, $1\times 1$ 합성곱 계층을 통해 채널 수를 클래스 수로 변환하고, 마지막으로 :numref:`sec_transposed_conv`에서 소개된 전치 합성곱을 통해 특징 맵의 높이와 너비를 입력 이미지의 그것으로 변환합니다.
그 결과, 모델 출력은 입력 이미지와 같은 높이와 너비를 가지며, 여기서 출력 채널은 같은 공간적 위치에 있는 입력 픽셀에 대한 예측된 클래스를 포함합니다.


![완전 합성곱 신경망.](../img/fcn.svg)
:label:`fig_fcn`

아래에서, 저희는 [**ImageNet 데이터셋에서 사전 훈련된 ResNet-18 모델을 사용해 이미지 특징을 추출**]하고 모델 인스턴스를 `pretrained_net`으로 표기합니다.
이 모델의 마지막 몇 계층은 전역 평균 풀링 계층과 완전 연결 계층을 포함합니다. 완전 합성곱 신경망에서는 이들이 필요하지 않습니다.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

다음으로, 저희는 [**완전 합성곱 신경망 인스턴스 `net`을 생성**]합니다.
이는 출력에 가장 가까운 마지막 전역 평균 풀링 계층과 완전 연결 계층을 제외한 ResNet-18의 모든 사전 훈련된 계층을 복사합니다.

```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

높이와 너비가 각각 320과 480인 입력이 주어졌을 때, `net`의 순전파는 입력 높이와 너비를 원래의 1/32, 즉 10과 15로 줄입니다.

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

다음으로, 저희는 [**$1\times 1$ 합성곱 계층을 사용해 출력 채널 수를 Pascal VOC2012 데이터셋의 클래스 수(21)로 변환**]합니다.
마지막으로, 저희는 (**특징 맵의 높이와 너비를 32배 증가**)시켜 입력 이미지의 높이와 너비로 다시 변경해야 합니다.
:numref:`sec_padding`에서 합성곱 계층의 출력 형태를 어떻게 계산하는지 떠올려 보세요.
$(320-64+16\times2+32)/32=10$이고 $(480-64+16\times2+32)/32=15$이므로, 저희는 스트라이드 $32$의 전치 합성곱 계층을 구성하는데, 커널의 높이와 너비를 $64$, 패딩을 $16$으로 설정합니다.
일반적으로, 저희는 스트라이드 $s$, 패딩 $s/2$ ($s/2$가 정수라고 가정), 커널의 높이와 너비 $2s$의 경우, 전치 합성곱이 입력의 높이와 너비를 $s$배 증가시킬 것임을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**전치 합성곱 계층 초기화**]


저희는 이미 전치 합성곱 계층이 특징 맵의 높이와 너비를 증가시킬 수 있다는 것을 알고 있습니다.
이미지 처리에서, 저희는 이미지를 확대해야 할 수 있는데, 즉 *업샘플링*입니다.
*쌍선형 보간(bilinear interpolation)*은 일반적으로 사용되는 업샘플링 기법 중 하나입니다.
이는 또한 전치 합성곱 계층을 초기화하는 데 자주 사용됩니다.

쌍선형 보간을 설명하기 위해, 입력 이미지가 주어졌을 때 업샘플링된 출력 이미지의 각 픽셀을 계산하고 싶다고 가정합시다.
좌표 $(x, y)$에서 출력 이미지의 픽셀을 계산하기 위해, 먼저 $(x, y)$를 입력 이미지의 좌표 $(x', y')$로 매핑합니다. 예를 들어, 입력 크기와 출력 크기의 비율에 따라서입니다.
매핑된 $x'$와 $y'$는 실수임에 유의하세요.
그런 다음, 입력 이미지에서 좌표 $(x', y')$에 가장 가까운 네 픽셀을 찾습니다.
마지막으로, 좌표 $(x, y)$에서 출력 이미지의 픽셀은 입력 이미지의 이 네 가장 가까운 픽셀과 $(x', y')$로부터의 상대 거리를 기반으로 계산됩니다.

쌍선형 보간의 업샘플링은 다음의 `bilinear_kernel` 함수에 의해 구성된 커널을 가진 전치 합성곱 계층에 의해 구현될 수 있습니다.
공간 제약으로, 저희는 알고리즘 설계에 대한 논의 없이 아래 `bilinear_kernel` 함수의 구현만 제공합니다.

```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

전치 합성곱 계층에 의해 구현된 [**쌍선형 보간의 업샘플링을 실험**]해 봅시다.
저희는 높이와 너비를 두 배로 만드는 전치 합성곱 계층을 구성하고, `bilinear_kernel` 함수로 그 커널을 초기화합니다.

```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

이미지 `X`를 읽고 업샘플링 출력을 `Y`에 할당합니다. 이미지를 출력하기 위해, 저희는 채널 차원의 위치를 조정해야 합니다.

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

보시다시피, 전치 합성곱 계층은 이미지의 높이와 너비를 모두 두 배로 증가시킵니다.
좌표의 다른 스케일을 제외하면, 쌍선형 보간으로 확대된 이미지와 :numref:`sec_bbox`에서 출력된 원래 이미지는 동일해 보입니다.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

완전 합성곱 신경망에서, 저희는 [**쌍선형 보간의 업샘플링으로 전치 합성곱 계층을 초기화합니다. $1\times 1$ 합성곱 계층에 대해서는, 저희는 Xavier 초기화를 사용합니다.**]

```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**데이터셋 읽기**]

저희는 :numref:`sec_semantic_segmentation`에서 소개된 시맨틱 분할 데이터셋을 읽습니다.
랜덤 자르기의 출력 이미지 형태는 $320\times 480$로 지정됩니다. 높이와 너비 모두 $32$로 나누어 떨어집니다.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**훈련**]


이제 저희가 구성한 완전 합성곱 신경망을 훈련할 수 있습니다.
여기서 손실 함수와 정확도 계산은 본질적으로 이전 장의 이미지 분류와 다르지 않습니다.
각 픽셀에 대한 클래스를 예측하기 위해 전치 합성곱 계층의 출력 채널을 사용하므로, 채널 차원이 손실 계산에서 지정됩니다.
또한, 정확도는 모든 픽셀에 대한 예측된 클래스의 정확성을 기반으로 계산됩니다.

```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**예측**]


예측할 때, 저희는 각 채널에서 입력 이미지를 표준화하고 이미지를 CNN이 요구하는 4차원 입력 형식으로 변환해야 합니다.

```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

각 픽셀의 [**예측된 클래스를 시각화**]하기 위해, 예측된 클래스를 데이터셋에서의 라벨 색상으로 다시 매핑합니다.

```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

테스트 데이터셋의 이미지는 크기와 형태가 다양합니다.
모델이 스트라이드 32의 전치 합성곱 계층을 사용하므로, 입력 이미지의 높이나 너비가 32로 나누어 떨어지지 않을 때, 전치 합성곱 계층의 출력 높이나 너비가 입력 이미지의 형태에서 벗어날 것입니다.
이 문제를 해결하기 위해, 저희는 이미지에서 높이와 너비가 32의 정수배인 여러 직사각형 영역을 자르고, 이러한 영역의 픽셀에 대해 순전파를 따로 수행할 수 있습니다.
이러한 직사각형 영역의 합집합이 입력 이미지를 완전히 덮어야 한다는 점에 유의하세요.
픽셀이 여러 직사각형 영역에 의해 덮일 때, 같은 픽셀에 대한 별도의 영역에서의 전치 합성곱 출력의 평균이 클래스를 예측하기 위해 소프트맥스 연산에 입력될 수 있습니다.


단순화를 위해, 저희는 몇 개의 더 큰 테스트 이미지만 읽고, 이미지의 좌상단 꼭짓점에서 시작해 $320\times480$ 영역을 예측을 위해 자릅니다.
이러한 테스트 이미지에 대해, 저희는 잘린 영역, 예측 결과, 실측값을 행별로 출력합니다.

```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## 요약

* 완전 합성곱 신경망은 먼저 CNN을 사용해 이미지 특징을 추출하고, 그런 다음 $1\times 1$ 합성곱 계층을 통해 채널 수를 클래스 수로 변환하고, 마지막으로 전치 합성곱을 통해 특징 맵의 높이와 너비를 입력 이미지의 그것으로 변환합니다.
* 완전 합성곱 신경망에서, 저희는 쌍선형 보간의 업샘플링을 사용해 전치 합성곱 계층을 초기화할 수 있습니다.


## 연습문제

1. 실험에서 전치 합성곱 계층에 Xavier 초기화를 사용하면, 결과가 어떻게 변하나요?
1. 하이퍼파라미터를 조정해 모델의 정확도를 추가로 개선할 수 있나요?
1. 테스트 이미지의 모든 픽셀의 클래스를 예측해 보세요.
1. 원래의 완전 합성곱 신경망 논문은 일부 중간 CNN 계층의 출력도 사용합니다 :cite:`Long.Shelhamer.Darrell.2015`. 이 아이디어를 구현해 보세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:
