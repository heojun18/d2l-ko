# 파인튜닝
:label:`sec_fine_tuning`

앞 장들에서 저희는 단 60,000개의 이미지로 구성된 Fashion-MNIST 훈련 데이터셋에서 모델을 훈련하는 방법을 논의했습니다. 또한 학계에서 가장 폭넓게 사용되는 대규모 이미지 데이터셋인 ImageNet도 설명했는데, 여기에는 천만 개 이상의 이미지와 1000개의 객체가 있습니다. 그러나 저희가 보통 마주치는 데이터셋의 크기는 이 두 데이터셋 사이입니다.


이미지에서 다양한 종류의 의자를 인식하고 사용자에게 구매 링크를 추천하려고 한다고 가정해 보겠습니다.
한 가지 가능한 방법은 먼저 100개의 일반적인 의자를 식별하고, 각 의자에 대해 다양한 각도의 1000개 이미지를 촬영한 다음, 수집된 이미지 데이터셋으로 분류 모델을 훈련하는 것입니다.
이 의자 데이터셋이 Fashion-MNIST 데이터셋보다 클 수 있지만, 예제 수는 여전히 ImageNet의 10분의 1보다 적습니다.
이는 ImageNet에 적합한 복잡한 모델이 이 의자 데이터셋에서 과적합을 일으킬 수 있습니다.
또한, 훈련 예제의 양이 제한적이기 때문에, 훈련된 모델의 정확도가 실제 요구사항을 충족하지 못할 수도 있습니다.


위의 문제를 해결하기 위해, 분명한 해결책 하나는 더 많은 데이터를 수집하는 것입니다.
그러나 데이터를 수집하고 라벨링하는 데에는 많은 시간과 비용이 들 수 있습니다.
예를 들어, ImageNet 데이터셋을 수집하기 위해 연구자들은 연구 기금에서 수백만 달러를 지출했습니다.
현재 데이터 수집 비용이 크게 줄어들었지만, 이 비용은 여전히 무시할 수 없습니다.


또 다른 해결책은 *전이 학습*을 적용해 *원본 데이터셋*에서 학습된 지식을 *대상 데이터셋*으로 전이하는 것입니다.
예를 들어, ImageNet 데이터셋의 대부분의 이미지가 의자와 관련이 없지만, 이 데이터셋에서 훈련된 모델은 엣지, 텍스처, 형태, 객체 구성을 식별하는 데 도움이 될 수 있는 더 일반적인 이미지 특징을 추출할 수 있습니다.
이러한 유사한 특징은 의자를 인식하는 데도 효과적일 수 있습니다.


## 단계


이 절에서는 전이 학습의 일반적인 기법인 *파인튜닝*을 소개합니다. :numref:`fig_finetune`에 나와 있듯이, 파인튜닝은 다음 네 단계로 구성됩니다.


1. 원본 데이터셋(예: ImageNet 데이터셋)에서 신경망 모델, 즉 *원본 모델*을 사전 훈련합니다.
1. 새로운 신경망 모델, 즉 *대상 모델*을 생성합니다. 이는 출력 계층을 제외한 원본 모델의 모든 모델 설계와 매개변수를 복사합니다. 저희는 이 모델 매개변수에 원본 데이터셋에서 학습된 지식이 포함되어 있으며, 이 지식이 대상 데이터셋에도 적용될 것이라고 가정합니다. 또한 원본 모델의 출력 계층은 원본 데이터셋의 라벨과 밀접하게 관련되어 있으므로 대상 모델에서 사용되지 않는다고 가정합니다.
1. 대상 모델에 출력 계층을 추가하는데, 이 출력 계층의 출력 수는 대상 데이터셋의 카테고리 수입니다. 그런 다음 이 계층의 모델 매개변수를 랜덤하게 초기화합니다.
1. 의자 데이터셋과 같은 대상 데이터셋에서 대상 모델을 훈련합니다. 출력 계층은 처음부터 훈련되며, 다른 모든 계층의 매개변수는 원본 모델의 매개변수를 기반으로 파인튜닝됩니다.

![파인튜닝.](../img/finetune.svg)
:label:`fig_finetune`

대상 데이터셋이 원본 데이터셋보다 훨씬 작을 때, 파인튜닝은 모델의 일반화 능력을 향상시키는 데 도움이 됩니다.


## 핫도그 인식

구체적인 사례를 통해 파인튜닝을 시연해 보겠습니다. 핫도그 인식입니다.
저희는 ImageNet 데이터셋에서 사전 훈련된 ResNet 모델을 작은 데이터셋에서 파인튜닝합니다.
이 작은 데이터셋은 핫도그가 있는 이미지와 없는 이미지 수천 개로 구성됩니다.
저희는 파인튜닝된 모델을 사용해 이미지에서 핫도그를 인식합니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### 데이터셋 읽기

[**저희가 사용하는 핫도그 데이터셋은 온라인 이미지에서 가져왔습니다**].
이 데이터셋은 핫도그를 포함하는 1400개의 양성 클래스 이미지와 다른 음식을 포함하는 동일한 수의 음성 클래스 이미지로 구성됩니다.
두 클래스의 1000개 이미지가 훈련에 사용되고 나머지는 테스트에 사용됩니다.


다운로드한 데이터셋의 압축을 풀면 `hotdog/train`과 `hotdog/test`라는 두 폴더를 얻을 수 있습니다. 두 폴더 모두 `hotdog`와 `not-hotdog` 하위 폴더를 가지며, 각 폴더에는 해당 클래스의 이미지가 들어 있습니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

훈련 및 테스트 데이터셋의 모든 이미지 파일을 읽기 위해 두 개의 인스턴스를 각각 생성합니다.

```{.python .input}
#@tab mxnet
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

처음 8개의 양성 예제와 마지막 8개의 음성 이미지가 아래에 표시됩니다. 보시다시피, [**이미지는 크기와 종횡비가 다양합니다**].

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

훈련 중에는 먼저 이미지에서 랜덤한 크기와 랜덤한 종횡비의 랜덤 영역을 자른 다음, 이 영역을 $224 \times 224$ 입력 이미지로 스케일링합니다.
테스트 중에는 이미지의 높이와 너비를 모두 256픽셀로 스케일링한 다음, 중앙의 $224 \times 224$ 영역을 입력으로 자릅니다.
또한, 세 RGB(빨강, 초록, 파랑) 색상 채널의 값을 채널별로 *표준화*합니다.
구체적으로, 채널의 평균값이 해당 채널의 각 값에서 차감된 다음, 그 결과가 해당 채널의 표준편차로 나누어집니다.

[~~데이터 증강~~]

```{.python .input}
#@tab mxnet
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**모델 정의 및 초기화**]

저희는 ImageNet 데이터셋에서 사전 훈련된 ResNet-18을 원본 모델로 사용합니다. 여기서는 사전 훈련된 모델 매개변수를 자동으로 다운로드하기 위해 `pretrained=True`를 지정합니다.
이 모델을 처음 사용하는 경우, 다운로드를 위해 인터넷 연결이 필요합니다.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
사전 훈련된 원본 모델 인스턴스에는 두 개의 멤버 변수가 있습니다. `features`와 `output`입니다. 전자는 출력 계층을 제외한 모델의 모든 계층을 포함하고, 후자는 모델의 출력 계층입니다.
이러한 구분의 주요 목적은 출력 계층을 제외한 모든 계층의 모델 매개변수의 파인튜닝을 용이하게 하기 위함입니다. 원본 모델의 멤버 변수 `output`이 아래에 표시되어 있습니다.
:end_tab:

:begin_tab:`pytorch`
사전 훈련된 원본 모델 인스턴스에는 여러 특징 계층과 출력 계층 `fc`가 포함되어 있습니다.
이러한 구분의 주요 목적은 출력 계층을 제외한 모든 계층의 모델 매개변수의 파인튜닝을 용이하게 하기 위함입니다. 원본 모델의 멤버 변수 `fc`는 아래에 나와 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

완전 연결 계층으로서, 이는 ResNet의 최종 전역 평균 풀링 출력을 ImageNet 데이터셋의 1000개 클래스 출력으로 변환합니다.
그런 다음 저희는 새로운 신경망을 대상 모델로 구성합니다. 이는 마지막 계층의 출력 수가 1000개가 아니라 대상 데이터셋의 클래스 수로 설정된다는 점을 제외하고는 사전 훈련된 원본 모델과 같은 방식으로 정의됩니다.

아래 코드에서, 대상 모델 인스턴스 `finetune_net`의 출력 계층 이전 모델 매개변수는 원본 모델의 해당 계층의 모델 매개변수로 초기화됩니다.
이러한 모델 매개변수는 ImageNet에서의 사전 훈련을 통해 얻어졌기 때문에 효과적입니다.
따라서 저희는 작은 학습률만 사용해 이러한 사전 훈련된 매개변수를 *파인튜닝*할 수 있습니다.
대조적으로, 출력 계층의 모델 매개변수는 랜덤하게 초기화되며 일반적으로 처음부터 학습되기 위해 더 큰 학습률이 필요합니다.
기본 학습률을 $\eta$라고 하면, 출력 계층의 모델 매개변수를 반복하는 데 $10\eta$의 학습률이 사용됩니다.

```{.python .input}
#@tab mxnet
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**모델 파인튜닝**]

먼저, 파인튜닝을 사용하는 훈련 함수 `train_fine_tuning`을 정의해 여러 번 호출할 수 있도록 합니다.

```{.python .input}
#@tab mxnet
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

저희는 사전 훈련을 통해 얻은 모델 매개변수를 *파인튜닝*하기 위해 [**기본 학습률을 작은 값으로 설정**]합니다. 이전 설정에 따라, 대상 모델의 출력 계층 매개변수를 10배 더 큰 학습률을 사용하여 처음부터 훈련합니다.

```{.python .input}
#@tab mxnet
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**비교를 위해**] 동일한 모델을 정의하되, (**모든 모델 매개변수를 랜덤 값으로 초기화**)합니다. 전체 모델을 처음부터 훈련해야 하므로 더 큰 학습률을 사용할 수 있습니다.

```{.python .input}
#@tab mxnet
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

보시다시피, 파인튜닝된 모델은 초기 매개변수 값이 더 효과적이기 때문에 같은 에폭에서 더 나은 성능을 보이는 경향이 있습니다.


## 요약

* 전이 학습은 원본 데이터셋에서 학습된 지식을 대상 데이터셋으로 전이합니다. 파인튜닝은 전이 학습을 위한 일반적인 기법입니다.
* 대상 모델은 출력 계층을 제외한 모든 모델 설계와 그 매개변수를 원본 모델에서 복사하고, 이러한 매개변수를 대상 데이터셋을 기반으로 파인튜닝합니다. 대조적으로, 대상 모델의 출력 계층은 처음부터 훈련되어야 합니다.
* 일반적으로 매개변수를 파인튜닝할 때는 더 작은 학습률을 사용하고, 출력 계층을 처음부터 훈련할 때는 더 큰 학습률을 사용할 수 있습니다.


## 연습문제

1. `finetune_net`의 학습률을 계속 증가시켜 보세요. 모델의 정확도가 어떻게 변하나요?
2. 비교 실험에서 `finetune_net`과 `scratch_net`의 하이퍼파라미터를 추가로 조정해 보세요. 여전히 정확도에 차이가 있나요?
3. `finetune_net`의 출력 계층 이전의 매개변수를 원본 모델의 매개변수로 설정하고, 훈련 중에 이를 갱신하지 *않도록* 해 보세요. 모델의 정확도가 어떻게 변하나요? 다음 코드를 사용할 수 있습니다.

```{.python .input}
#@tab mxnet
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. 실제로 `ImageNet` 데이터셋에는 "hotdog" 클래스가 있습니다. 출력 계층에서 해당 가중치 매개변수는 다음 코드를 통해 얻을 수 있습니다. 이 가중치 매개변수를 어떻게 활용할 수 있을까요?

```{.python .input}
#@tab mxnet
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1439)
:end_tab:
