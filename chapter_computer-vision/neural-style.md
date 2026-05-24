# 신경망 스타일 전이

사진 애호가라면, 필터에 익숙하실 수 있습니다.
이는 사진의 색상 스타일을 변경하여 풍경 사진을 더 선명하게 만들거나 인물 사진의 피부를 화이트닝할 수 있게 합니다.
하지만, 하나의 필터는 보통 사진의 한 측면만 바꿉니다.
사진에 이상적인 스타일을 적용하기 위해, 아마도 많은 다양한 필터 조합을 시도해야 할 것입니다.
이 과정은 모델의 하이퍼파라미터를 조정하는 것만큼 복잡합니다.



이 절에서, 저희는 CNN의 계층별 표현을 활용해 한 이미지의 스타일을 다른 이미지에 자동으로 적용하는 것, 즉 *스타일 전이* :cite:`Gatys.Ecker.Bethge.2016`를 할 것입니다.
이 작업에는 두 개의 입력 이미지가 필요합니다. 하나는 *컨텐츠 이미지*이고 다른 하나는 *스타일 이미지*입니다.
저희는 신경망을 사용해 컨텐츠 이미지를 수정해 스타일에서 스타일 이미지에 가깝게 만들 것입니다.
예를 들어, :numref:`fig_style_transfer`의 컨텐츠 이미지는 시애틀 교외의 마운트 레이니어 국립공원에서 저희가 찍은 풍경 사진이고, 스타일 이미지는 가을 참나무를 주제로 한 유화입니다.
출력 합성 이미지에서는, 스타일 이미지의 유화 붓 터치가 적용되어 더 생생한 색상이 나오면서도, 컨텐츠 이미지의 객체의 주요 형태는 보존됩니다.

![컨텐츠 이미지와 스타일 이미지가 주어졌을 때, 스타일 전이는 합성 이미지를 출력합니다.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## 방법

:numref:`fig_style_transfer_model`은 단순화된 예제로 CNN 기반 스타일 전이 방법을 설명합니다.
먼저, 저희는 합성 이미지를 예를 들어 컨텐츠 이미지로 초기화합니다.
이 합성 이미지가 스타일 전이 과정에서 갱신되어야 하는 유일한 변수, 즉 훈련 중에 갱신될 모델 매개변수입니다.
그런 다음 이미지 특징을 추출하기 위해 사전 훈련된 CNN을 선택하고 훈련 중에 그 모델 매개변수를 고정합니다.
이 심층 CNN은 이미지에 대한 계층적 특징을 추출하기 위해 여러 계층을 사용합니다.
저희는 이러한 계층 중 일부의 출력을 컨텐츠 특징 또는 스타일 특징으로 선택할 수 있습니다.
:numref:`fig_style_transfer_model`을 예로 들어 보겠습니다.
여기서 사전 훈련된 신경망에는 3개의 합성곱 계층이 있으며, 두 번째 계층은 컨텐츠 특징을 출력하고, 첫 번째와 세 번째 계층은 스타일 특징을 출력합니다.

![CNN 기반 스타일 전이 과정. 실선은 순전파의 방향을 보여주고 점선은 역전파를 보여줍니다.](../img/neural-style.svg)
:label:`fig_style_transfer_model`

다음으로, 저희는 순전파(실선 화살표의 방향)를 통해 스타일 전이의 손실 함수를 계산하고, 역전파(점선 화살표의 방향)를 통해 모델 매개변수(출력을 위한 합성 이미지)를 갱신합니다.
스타일 전이에서 일반적으로 사용되는 손실 함수는 세 부분으로 구성됩니다.
(i) *컨텐츠 손실*은 합성 이미지와 컨텐츠 이미지가 컨텐츠 특징에서 가까워지도록 합니다.
(ii) *스타일 손실*은 합성 이미지와 스타일 이미지가 스타일 특징에서 가까워지도록 합니다.
(iii) *총 변동 손실(total variation loss)*은 합성 이미지의 잡음을 줄이는 데 도움이 됩니다.
마지막으로, 모델 훈련이 끝나면, 저희는 최종 합성 이미지를 생성하기 위해 스타일 전이의 모델 매개변수를 출력합니다.



다음에서, 저희는 구체적인 실험을 통해 스타일 전이의 기술적 세부사항을 설명할 것입니다.


## [**컨텐츠 이미지와 스타일 이미지 읽기**]

먼저, 저희는 컨텐츠 이미지와 스타일 이미지를 읽습니다.
출력된 좌표축으로부터, 저희는 이러한 이미지가 다양한 크기를 가짐을 알 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
#@tab mxnet
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**전처리와 후처리**]

아래에서, 저희는 이미지의 전처리와 후처리를 위한 두 함수를 정의합니다.
`preprocess` 함수는 입력 이미지의 세 RGB 채널 각각을 표준화하고 그 결과를 CNN 입력 형식으로 변환합니다.
`postprocess` 함수는 출력 이미지의 픽셀 값을 표준화 이전의 원래 값으로 복원합니다.
이미지 출력 함수가 각 픽셀이 0에서 1까지의 부동 소수점 값을 가질 것을 요구하므로, 저희는 0보다 작거나 1보다 큰 값을 각각 0 또는 1로 대체합니다.

```{.python .input}
#@tab mxnet
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [**특징 추출**]

저희는 이미지 특징을 추출하기 위해 ImageNet 데이터셋에서 사전 훈련된 VGG-19 모델을 사용합니다 :cite:`Gatys.Ecker.Bethge.2016`.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

이미지의 컨텐츠 특징과 스타일 특징을 추출하기 위해, 저희는 VGG 신경망의 특정 계층의 출력을 선택할 수 있습니다.
일반적으로 말해서, 입력 계층에 더 가까울수록 이미지의 세부 사항을 추출하기 더 쉽고, 그 반대도 마찬가지로 이미지의 전역 정보를 추출하기 더 쉽습니다. 합성 이미지에서 컨텐츠 이미지의 세부 사항을 과도하게 유지하는 것을 피하기 위해, 저희는 이미지의 컨텐츠 특징을 출력하기 위한 *컨텐츠 계층*으로 출력에 더 가까운 VGG 계층을 선택합니다.
저희는 또한 지역적 및 전역적 스타일 특징을 추출하기 위해 다양한 VGG 계층의 출력을 선택합니다.
이러한 계층은 *스타일 계층*이라고도 부릅니다.
:numref:`sec_vgg`에서 언급했듯이, VGG 신경망은 5개의 합성곱 블록을 사용합니다.
실험에서, 저희는 네 번째 합성곱 블록의 마지막 합성곱 계층을 컨텐츠 계층으로, 각 합성곱 블록의 첫 번째 합성곱 계층을 스타일 계층으로 선택합니다.
이러한 계층의 인덱스는 `pretrained_net` 인스턴스를 출력하여 얻을 수 있습니다.

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

VGG 계층을 사용해 특징을 추출할 때, 저희는 입력 계층에서 출력 계층에 가장 가까운 컨텐츠 계층 또는 스타일 계층까지의 모든 계층만 사용하면 됩니다.
특징 추출에 사용될 모든 VGG 계층만 유지하는 새로운 신경망 인스턴스 `net`을 구성해 봅시다.

```{.python .input}
#@tab mxnet
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

입력 `X`가 주어졌을 때, 단순히 순전파 `net(X)`를 호출하면 마지막 계층의 출력만 얻을 수 있습니다.
저희는 중간 계층의 출력도 필요하므로, 계층별 계산을 수행하고 컨텐츠와 스타일 계층 출력을 유지해야 합니다.

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

아래에 두 함수가 정의되어 있습니다.
`get_contents` 함수는 컨텐츠 이미지로부터 컨텐츠 특징을 추출하고, `get_styles` 함수는 스타일 이미지로부터 스타일 특징을 추출합니다.
훈련 중에 사전 훈련된 VGG의 모델 매개변수를 갱신할 필요가 없으므로, 저희는 훈련이 시작되기 전에도 컨텐츠와 스타일 특징을 추출할 수 있습니다.
합성 이미지가 스타일 전이를 위해 갱신될 모델 매개변수의 집합이므로, 저희는 훈련 중에 `extract_features` 함수를 호출함으로써만 합성 이미지의 컨텐츠와 스타일 특징을 추출할 수 있습니다.

```{.python .input}
#@tab mxnet
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**손실 함수 정의**]

이제 스타일 전이의 손실 함수를 설명할 것입니다. 손실 함수는 컨텐츠 손실, 스타일 손실, 총 변동 손실로 구성됩니다.

### 컨텐츠 손실

선형 회귀의 손실 함수와 유사하게, 컨텐츠 손실은 제곱 손실 함수를 통해 합성 이미지와 컨텐츠 이미지 사이의 컨텐츠 특징의 차이를 측정합니다.
제곱 손실 함수의 두 입력은 모두 `extract_features` 함수에 의해 계산된 컨텐츠 계층의 출력입니다.

```{.python .input}
#@tab mxnet
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

### 스타일 손실

스타일 손실은 컨텐츠 손실과 유사하게, 합성 이미지와 스타일 이미지 사이의 스타일 차이를 측정하기 위해 제곱 손실 함수도 사용합니다.
임의의 스타일 계층의 스타일 출력을 표현하기 위해, 저희는 먼저 `extract_features` 함수를 사용해 스타일 계층 출력을 계산합니다.
출력이 1개의 예제, $c$개의 채널, 높이 $h$, 너비 $w$를 가진다고 가정하면, 저희는 이 출력을 $c$개의 행과 $hw$개의 열을 가진 행렬 $\mathbf{X}$로 변환할 수 있습니다.
이 행렬은 각각 길이가 $hw$인 $c$개의 벡터 $\mathbf{x}_1, \ldots, \mathbf{x}_c$의 연결로 생각할 수 있습니다.
여기서, 벡터 $\mathbf{x}_i$는 채널 $i$의 스타일 특징을 나타냅니다.

이러한 벡터의 *그람 행렬(Gram matrix)* $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$에서, 행 $i$와 열 $j$의 원소 $x_{ij}$는 벡터 $\mathbf{x}_i$와 $\mathbf{x}_j$의 내적입니다.
이는 채널 $i$와 $j$의 스타일 특징의 상관관계를 나타냅니다.
저희는 이 그람 행렬을 임의의 스타일 계층의 스타일 출력을 나타내는 데 사용합니다.
$hw$의 값이 더 크면, 아마도 그람 행렬에서 더 큰 값으로 이어진다는 점에 유의하세요.
그람 행렬의 높이와 너비가 모두 채널 수 $c$라는 점도 유의하세요.
스타일 손실이 이러한 값에 영향을 받지 않도록, 아래의 `gram` 함수는 그람 행렬을 그 원소 수, 즉 $chw$로 나눕니다.

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

분명히, 스타일 손실에 대한 제곱 손실 함수의 두 그람 행렬 입력은 합성 이미지와 스타일 이미지에 대한 스타일 계층 출력을 기반으로 합니다.
여기서는 스타일 이미지를 기반으로 한 그람 행렬 `gram_Y`가 사전 계산되었다고 가정합니다.

```{.python .input}
#@tab mxnet
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### 총 변동 손실

때때로, 학습된 합성 이미지는 많은 고주파 잡음, 즉 특히 밝거나 어두운 픽셀을 가집니다.
일반적인 잡음 감소 방법 중 하나는 *총 변동 노이즈 제거(total variation denoising)*입니다.
좌표 $(i, j)$에서의 픽셀 값을 $x_{i, j}$로 표기합니다.
총 변동 손실

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

을 줄이는 것은 합성 이미지의 인접 픽셀의 값을 더 가깝게 만듭니다.

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### 손실 함수

[**스타일 전이의 손실 함수는 컨텐츠 손실, 스타일 손실, 총 변동 손실의 가중합입니다**].
이러한 가중치 하이퍼파라미터를 조정함으로써, 저희는 합성 이미지에서 컨텐츠 보존, 스타일 전이, 잡음 감소 사이의 균형을 맞출 수 있습니다.

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e4, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**합성 이미지 초기화**]

스타일 전이에서, 합성 이미지는 훈련 중에 갱신되어야 하는 유일한 변수입니다.
따라서, 저희는 간단한 모델 `SynthesizedImage`를 정의하고 합성 이미지를 모델 매개변수로 취급할 수 있습니다.
이 모델에서, 순전파는 단순히 모델 매개변수를 반환합니다.

```{.python .input}
#@tab mxnet
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

다음으로, 저희는 `get_inits` 함수를 정의합니다.
이 함수는 합성 이미지 모델 인스턴스를 생성하고 이를 이미지 `X`로 초기화합니다.
다양한 스타일 계층에서의 스타일 이미지에 대한 그람 행렬 `styles_Y_gram`은 훈련 전에 계산됩니다.

```{.python .input}
#@tab mxnet
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**훈련**]


스타일 전이를 위해 모델을 훈련할 때, 저희는 합성 이미지의 컨텐츠 특징과 스타일 특징을 지속적으로 추출하고 손실 함수를 계산합니다.
아래는 훈련 루프를 정의합니다.

```{.python .input}
#@tab mxnet
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

이제 [**모델을 훈련하기 시작**]합니다.
저희는 컨텐츠 이미지와 스타일 이미지의 높이와 너비를 300 x 450 픽셀로 크기 조정합니다.
저희는 합성 이미지를 초기화하기 위해 컨텐츠 이미지를 사용합니다.

```{.python .input}
#@tab mxnet
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

합성 이미지가 컨텐츠 이미지의 풍경과 객체를 유지하면서 동시에 스타일 이미지의 색상을 전이하는 것을 볼 수 있습니다.
예를 들어, 합성 이미지는 스타일 이미지의 것과 같은 색상 블록을 가집니다.
이러한 블록 중 일부는 심지어 붓 터치의 미묘한 텍스처를 가집니다.




## 요약

* 스타일 전이에서 일반적으로 사용되는 손실 함수는 세 부분으로 구성됩니다. (i) 컨텐츠 손실은 합성 이미지와 컨텐츠 이미지가 컨텐츠 특징에서 가까워지도록 합니다. (ii) 스타일 손실은 합성 이미지와 스타일 이미지가 스타일 특징에서 가까워지도록 합니다. (iii) 총 변동 손실은 합성 이미지의 잡음을 줄이는 데 도움이 됩니다.
* 저희는 사전 훈련된 CNN을 사용해 이미지 특징을 추출하고 손실 함수를 최소화하여 훈련 중에 모델 매개변수로서 합성 이미지를 지속적으로 갱신할 수 있습니다.
* 저희는 스타일 계층의 스타일 출력을 나타내기 위해 그람 행렬을 사용합니다.


## 연습문제

1. 다른 컨텐츠 계층과 스타일 계층을 선택하면 출력이 어떻게 변하나요?
1. 손실 함수의 가중치 하이퍼파라미터를 조정해 보세요. 출력이 더 많은 컨텐츠를 유지하거나 더 적은 잡음을 가지나요?
1. 다른 컨텐츠 이미지와 스타일 이미지를 사용해 보세요. 더 흥미로운 합성 이미지를 만들 수 있나요?
1. 스타일 전이를 텍스트에 적용할 수 있을까요? 힌트: :citet:`10.1145/3544903.3544906`의 서베이 논문을 참조하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:
