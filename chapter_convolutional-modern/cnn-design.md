```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 합성곱 네트워크 아키텍처 설계
:label:`sec_cnn-design`

이전 절들은 컴퓨터 비전을 위한 현대 네트워크 설계의 둘러보기를 제공했습니다. 저희가 다룬 모든 작업에 공통적인 점은 과학자들의 직관에 크게 의존했다는 것입니다. 많은 아키텍처는 인간의 창의성에 의해 크게 영향을 받았으며, 심층 네트워크가 제공하는 설계 공간의 체계적인 탐색에 의한 영향은 훨씬 적었습니다. 그럼에도 불구하고, 이러한 *네트워크 엔지니어링* 접근법은 엄청나게 성공적이었습니다.

AlexNet(:numref:`sec_alexnet`)이
ImageNet에서 기존의 컴퓨터 비전 모델을 이긴 이후,
모두 같은 패턴에 따라 설계된 합성곱 블록을 쌓아
매우 깊은 네트워크를 구성하는 것이 인기를 얻었습니다.
특히, $3 \times 3$ 합성곱은
VGG 네트워크(:numref:`sec_vgg`)에 의해 대중화되었습니다.
NiN(:numref:`sec_nin`)은 $1 \times 1$ 합성곱도
국소적 비선형성을 추가함으로써 유익할 수 있음을 보여주었습니다.
더욱이, NiN은 모든 위치에 걸쳐 통합함으로써
네트워크의 머리에서 정보를 통합하는 문제를 해결했습니다.
GoogLeNet(:numref:`sec_googlenet`)은 인셉션 블록에서 VGG와 NiN의 장점을 결합하면서,
서로 다른 합성곱 너비의 여러 분기를 추가했습니다.
ResNet(:numref:`sec_resnet`)은
귀납 편향을 항등 매핑으로 변경했습니다($f(x) = 0$에서). 이는 매우 깊은 네트워크를 가능하게 했습니다. 거의 10년이 지난 지금도, ResNet 설계는 여전히 인기가 있으며, 그 설계의 증거입니다. 마지막으로, ResNeXt(:numref:`subsec_resnext`)는 파라미터와 계산 간의 더 나은 트레이드오프를 제공하는 그룹화된 합성곱을 추가했습니다. 비전을 위한 트랜스포머의 전조인 Squeeze-and-Excitation Networks(SENets)는 위치 간 효율적인 정보 전송을 가능하게 합니다
:cite:`Hu.Shen.Sun.2018`. 이는 채널별 전역 어텐션 함수를 계산함으로써 달성되었습니다.

지금까지 저희는 *신경망 아키텍처 탐색*(NAS) :cite:`zoph2016neural,liu2018darts`을 통해 얻은 네트워크를 생략했습니다. 그 비용이 일반적으로 엄청나며, 무차별 대입 탐색, 유전 알고리즘, 강화 학습, 또는 다른 형태의 하이퍼파라미터 최적화에 의존하기 때문에 저희는 그렇게 하기로 선택했습니다. 고정된 탐색 공간이 주어지면,
NAS는 반환된 성능 추정치에 기반하여
아키텍처를 자동으로 선택하기 위한 탐색 전략을 사용합니다.
NAS의 결과는
단일 네트워크 인스턴스입니다. EfficientNets는 이 탐색의 주목할 만한 결과입니다 :cite:`tan2019efficientnet`.

다음에서, 저희는 *단일 최선의 네트워크*에 대한 탐구와는 상당히 다른 아이디어를 논의합니다. 이는 계산적으로 비교적 저렴하고, 그 과정에서 과학적 통찰로 이어지며, 결과의 품질 측면에서 상당히 효과적입니다. :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 *네트워크 설계 공간 설계* 전략을 검토해 봅시다. 이 전략은 수동 설계와 NAS의 강점을 결합합니다. 이는 *네트워크의 분포*에서 작동하고 네트워크 전체 패밀리에 대해 좋은 성능을 얻기 위해 분포를 최적화함으로써 이를 달성합니다. 그 결과는 *RegNets*, 특히 RegNetX와 RegNetY, 그리고 성능이 좋은 CNN 설계를 위한 일련의 안내 원칙입니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## AnyNet 설계 공간
:label:`subsec_the-anynet-design-space`

아래의 설명은 책의 범위에 맞도록 약간의 축약과 함께 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 추론을 밀접하게 따릅니다.
시작하기 위해, 저희는 탐색할 네트워크 패밀리에 대한 템플릿이 필요합니다. 이 장의 설계들의 공통점 중 하나는 네트워크가 *줄기*, *본체*, *머리*로 구성된다는 것입니다. 줄기는 종종 더 큰 윈도우 크기의 합성곱을 통해 초기 이미지 처리를 수행합니다. 본체는 여러 블록으로 구성되며, 원시 이미지에서 객체 표현으로 가는 데 필요한 변환의 대부분을 수행합니다. 마지막으로, 머리는 예를 들어 다중 클래스 분류를 위한 소프트맥스 회귀를 통해 이를 원하는 출력으로 변환합니다.
본체는, 차례로, 여러 스테이지로 구성되며, 감소하는 해상도에서 이미지에 대해 작동합니다. 사실, 줄기와 각 후속 스테이지는 공간 해상도를 1/4로 줄입니다. 마지막으로, 각 스테이지는 하나 이상의 블록으로 구성됩니다. 이 패턴은 VGG에서 ResNeXt까지 모든 네트워크에 공통입니다. 실제로, 일반적인 AnyNet 네트워크의 설계를 위해, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`은 :numref:`fig_resnext_block`의 ResNeXt 블록을 사용했습니다.


![AnyNet 설계 공간. 각 화살표를 따라 있는 숫자 $(\mathit{c}, \mathit{r})$는 채널의 수 $c$와 그 지점에서 이미지의 해상도 $\mathit{r} \times \mathit{r}$을 나타냅니다. 왼쪽에서 오른쪽으로: 줄기, 본체, 머리로 구성된 일반적인 네트워크 구조; 네 개의 스테이지로 구성된 본체; 스테이지의 상세 구조; 블록에 대한 두 가지 대안적 구조, 하나는 다운샘플링이 없고 하나는 각 차원에서 해상도를 절반으로 만드는 것. 설계 선택에는 깊이 $\mathit{d_i}$, 출력 채널의 수 $\mathit{c_i}$, 그룹의 수 $\mathit{g_i}$, 그리고 어떤 스테이지 $\mathit{i}$에 대한 병목 비율 $\mathit{k_i}$가 포함됩니다.](../img/anynet.svg)
:label:`fig_anynet_full`

:numref:`fig_anynet_full`에 윤곽이 그려진 구조를 자세히 검토해 봅시다. 언급한 바와 같이, AnyNet은 줄기, 본체, 머리로 구성됩니다. 줄기는 RGB 이미지(3채널)를 입력으로 받아, 스트라이드 $2$의 $3 \times 3$ 합성곱과 그 다음 배치 노름을 사용하여 해상도를 $r \times r$에서 $r/2 \times r/2$로 절반으로 만듭니다. 또한, 본체에 대한 입력 역할을 하는 $c_0$ 채널을 생성합니다.

네트워크는 $224 \times 224 \times 3$ 모양의 ImageNet 이미지와 잘 작동하도록 설계되었으므로, 본체는 이를 4개의 스테이지($224 / 2^{1+4} = 7$임을 떠올려 보세요)를 통해 $7 \times 7 \times c_4$로 줄이는 역할을 하며, 각각 최종적으로 스트라이드 $2$를 가집니다. 마지막으로, 머리는 NiN(:numref:`sec_nin`)과 유사한 전역 평균 풀링을 통해 완전히 표준적인 설계를 채택하고, 그 다음 $n$-클래스 분류를 위한 $n$-차원 벡터를 방출하는 완전 연결 층을 사용합니다.

관련 설계 결정의 대부분은 네트워크의 본체에 내재되어 있습니다. 이는 스테이지로 진행되며, 각 스테이지는 저희가 :numref:`subsec_resnext`에서 논의한 것과 동일한 유형의 ResNeXt 블록으로 구성됩니다. 거기서의 설계는 다시 한번 완전히 일반적입니다: 저희는 스트라이드 $2$를 사용하여 해상도를 절반으로 만드는 블록으로 시작합니다(:numref:`fig_anynet_full`의 가장 오른쪽). 이를 일치시키기 위해, ResNeXt 블록의 잔차 분기는 $1 \times 1$ 합성곱을 통과해야 합니다. 이 블록 다음에는 해상도와 채널의 수를 변경하지 않는 가변적인 수의 추가 ResNeXt 블록이 옵니다. 일반적인 설계 관행은 합성곱 블록의 설계에 약간의 병목을 추가하는 것임에 주목하세요.
이와 같이, 병목 비율 $k_i \geq 1$로 저희는 스테이지 $i$의 각 블록 내에서 일정 수의 채널 $c_i/k_i$를 허용합니다(실험이 보여주듯이, 이것은 실제로 효과적이지 않으며 건너뛰어야 합니다). 마지막으로, 저희가 ResNeXt 블록을 다루고 있으므로, 스테이지 $i$에서 그룹화된 합성곱에 대한 그룹의 수 $g_i$도 선택해야 합니다.

이 겉보기에 일반적인 설계 공간은 그럼에도 불구하고 많은 파라미터를 제공합니다: 저희는 블록 너비(채널의 수) $c_0, \ldots c_4$, 스테이지당 깊이(블록의 수) $d_1, \ldots d_4$, 병목 비율 $k_1, \ldots k_4$, 그리고 그룹 너비(그룹의 수) $g_1, \ldots g_4$를 설정할 수 있습니다.
총 17개의 파라미터에 달하며, 탐색할 가치가 있는 비합리적으로 많은 수의 구성을 초래합니다. 이 거대한 설계 공간을 효과적으로 줄이기 위해 일부 도구가 필요합니다. 이것이 설계 공간의 개념적 아름다움이 발휘되는 곳입니다. 그렇게 하기 전에, 먼저 일반적인 설계를 구현해 봅시다.

```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

각 스테이지는 `depth` ResNeXt 블록으로 구성되며,
`num_channels`는 블록 너비를 지정합니다.
첫 번째 블록은 입력 이미지의 높이와 너비를 절반으로 만듦에 주목하세요.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

네트워크 줄기, 본체, 머리를 함께 모아,
저희는 AnyNet의 구현을 완성합니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## 설계 공간의 분포와 파라미터

:numref:`subsec_the-anynet-design-space`에서 방금 논의한 바와 같이, 설계 공간의 파라미터는 그 설계 공간 내 네트워크의 하이퍼파라미터입니다.
AnyNet 설계 공간에서 좋은 파라미터를 식별하는 문제를 고려하세요. 저희는 주어진 양의 계산(예: FLOPs와 계산 시간)에 대해 *단일 최선*의 파라미터 선택을 찾으려고 시도할 수 있습니다. 각 파라미터에 대해 *두 가지* 가능한 선택만 허용하더라도, 최선의 해를 찾기 위해 $2^{17} = 131072$ 조합을 탐색해야 합니다. 이는 그 엄청난 비용 때문에 분명히 실현 불가능합니다. 더 나쁘게는, 네트워크를 설계해야 하는 방법 측면에서 이 연습에서 정말로 아무것도 배우지 못합니다. 다음에 예를 들어 X-스테이지 또는 시프트 연산 등을 추가하면, 처음부터 다시 시작해야 합니다. 더 나쁘게는, 학습의 확률성(반올림, 셔플링, 비트 오류)으로 인해, 어떤 두 실행도 정확히 같은 결과를 생성할 가능성이 없습니다. 더 나은 전략은 파라미터의 선택이 어떻게 관련되어야 하는지에 대한 일반적인 지침을 결정하려고 시도하는 것입니다. 예를 들어, 병목 비율, 채널, 블록, 그룹의 수, 또는 층 간의 그들의 변경은 이상적으로 일련의 간단한 규칙에 의해 지배되어야 합니다. :citet:`radosavovic2019network`의 접근법은 다음 네 가지 가정에 의존합니다:

1. 저희는 일반적인 설계 원칙이 실제로 존재한다고 가정하므로, 이러한 요구 사항을 만족하는 많은 네트워크가 좋은 성능을 제공해야 합니다. 결과적으로, 네트워크 *분포*를 식별하는 것이 합리적인 전략이 될 수 있습니다. 즉, 저희는 건초더미에 좋은 바늘이 많다고 가정합니다.
1. 저희는 네트워크가 좋은지 평가할 수 있기 전에 수렴까지 네트워크를 학습시킬 필요가 없습니다. 대신, 중간 결과를 최종 정확도에 대한 신뢰할 수 있는 안내로 사용하는 것으로 충분합니다. 목적을 최적화하기 위해 (근사) 프록시를 사용하는 것을 다중-충실도 최적화라고 합니다 :cite:`forrester2007multi`. 결과적으로, 설계 최적화는 데이터셋을 몇 번만 통과한 후 달성된 정확도를 기반으로 수행되어, 비용을 크게 줄입니다.
1. 더 작은 규모(더 작은 네트워크)에서 얻은 결과는 더 큰 것으로 일반화됩니다. 결과적으로, 최적화는 구조적으로 유사하지만, 더 적은 수의 블록, 더 적은 채널 등을 가진 네트워크에 대해 수행됩니다. 결국에만 저희는 찾은 네트워크도 규모에서 좋은 성능을 제공하는지 확인해야 할 것입니다.
1. 설계의 측면은 근사적으로 인수분해될 수 있으므로, 결과의 품질에 미치는 영향을 어느 정도 독립적으로 추론할 수 있습니다. 즉, 최적화 문제는 적당히 쉽습니다.

이러한 가정은 많은 네트워크를 저렴하게 테스트할 수 있게 합니다. 특히, 저희는 구성의 공간에서 균일하게 *샘플링*하고 그 성능을 평가할 수 있습니다. 이후, 저희는 해당 네트워크로 달성할 수 있는 오류/정확도의 *분포*를 검토하여 파라미터 선택의 품질을 평가할 수 있습니다. 확률 분포 $p$를 사용하여 추출된 주어진 설계 공간의 네트워크에 의해 저질러진 오류에 대한 누적 분포 함수(CDF)를 $F(e)$로 표시합니다. 즉,

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

이제 저희의 목표는 대부분의 네트워크가 매우 낮은 오류율을 가지고 $p$의 지지가 간결한 *네트워크*에 대한 분포 $p$를 찾는 것입니다. 물론, 이는 정확하게 수행하기에는 계산적으로 실현 불가능합니다. 저희는 $p$로부터 네트워크 $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$의 샘플(각각 오류 $e_1, \ldots, e_n$을 가짐)에 의지하고 대신 경험적 CDF $\hat{F}(e, \mathcal{Z})$를 사용합니다:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

한 선택 세트에 대한 CDF가 다른 CDF를 우세하게 만들거나 (또는 일치하게 만들) 때마다 그 파라미터의 선택이 우월하다(또는 무관하다)는 것이 따릅니다. 이에 따라
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020`은 네트워크의 모든 스테이지 $i$에 대해 공유된 네트워크 병목 비율 $k_i = k$를 실험했습니다. 이는 병목 비율을 지배하는 네 가지 파라미터 중 세 가지를 제거합니다. 이것이 성능에 (부정적으로) 영향을 미치는지 평가하기 위해, 제약된 분포와 제약되지 않은 분포에서 네트워크를 추출하고 해당 CDF를 비교할 수 있습니다. :numref:`fig_regnet-fig`의 첫 번째 패널에서 볼 수 있듯이, 이 제약은 네트워크 분포의 정확도에 전혀 영향을 미치지 않는 것으로 밝혀졌습니다.
마찬가지로, 저희는 네트워크의 다양한 스테이지에서 발생하는 동일한 그룹 너비 $g_i = g$를 선택할 수 있습니다. 다시 한번, 이는 :numref:`fig_regnet-fig`의 두 번째 패널에서 볼 수 있듯이, 성능에 영향을 미치지 않습니다.
두 단계를 결합하면 자유 파라미터의 수를 6개 줄입니다.

![설계 공간의 오류 경험적 분포 함수 비교. $\textrm{AnyNet}_\mathit{A}$는 원래 설계 공간입니다; $\textrm{AnyNet}_\mathit{B}$는 병목 비율을 묶고, $\textrm{AnyNet}_\mathit{C}$는 그룹 너비도 묶으며, $\textrm{AnyNet}_\mathit{D}$는 스테이지에 걸쳐 네트워크 깊이를 증가시킵니다. 왼쪽에서 오른쪽으로: (i) 병목 비율을 묶는 것은 성능에 영향을 미치지 않습니다; (ii) 그룹 너비를 묶는 것은 성능에 영향을 미치지 않습니다; (iii) 스테이지에 걸쳐 네트워크 너비(채널)를 증가시키는 것은 성능을 향상시킵니다; (iv) 스테이지에 걸쳐 네트워크 깊이를 증가시키는 것은 성능을 향상시킵니다. 그림은 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 제공입니다.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

다음으로 저희는 스테이지의 너비와 깊이에 대한 잠재적 선택의 다수를 줄이는 방법을 찾습니다. 더 깊어질수록, 채널의 수가 증가해야 한다고 가정하는 것이 합리적입니다. 즉, $c_i \geq c_{i-1}$ (:numref:`fig_regnet-fig`의 그들의 표기법에 따라 $w_{i+1} \geq w_i$), 이는
$\textrm{AnyNetX}_D$를 산출합니다. 마찬가지로, 스테이지가 진행됨에 따라, 더 깊어져야 한다고 가정하는 것도 똑같이 합리적입니다. 즉, $d_i \geq d_{i-1}$, 이는 $\textrm{AnyNetX}_E$를 산출합니다. 이는 :numref:`fig_regnet-fig`의 세 번째와 네 번째 패널에서 실험적으로 각각 확인할 수 있습니다.

## RegNet

결과 $\textrm{AnyNetX}_E$ 설계 공간은
해석하기 쉬운 설계 원칙을 따르는 간단한 네트워크로 구성됩니다:

* 모든 스테이지 $i$에 대해 병목 비율 $k_i = k$를 공유합니다;
* 모든 스테이지 $i$에 대해 그룹 너비 $g_i = g$를 공유합니다;
* 스테이지에 걸쳐 네트워크 너비를 증가시킵니다: $c_{i} \leq c_{i+1}$;
* 스테이지에 걸쳐 네트워크 깊이를 증가시킵니다: $d_{i} \leq d_{i+1}$.

이는 저희에게 마지막 선택 세트를 남깁니다: 결국의 $\textrm{AnyNetX}_E$ 설계 공간의 위 파라미터에 대한 특정 값을 어떻게 선택할지. $\textrm{AnyNetX}_E$ 분포에서 가장 성능이 좋은 네트워크를 연구함으로써 다음을 관찰할 수 있습니다: 네트워크의 너비는 이상적으로 네트워크에 걸쳐 블록 인덱스와 함께 선형적으로 증가합니다. 즉, $c_j \approx c_0 + c_a j$, 여기서 $j$는 블록 인덱스이고 기울기 $c_a > 0$입니다. 스테이지당 다른 블록 너비만 선택할 수 있다는 점을 고려하면, 저희는 이 의존성과 일치하도록 설계된 조각별 상수 함수에 도달합니다. 더욱이, 실험은 또한 병목 비율 $k = 1$이 가장 잘 수행됨을 보여줍니다. 즉, 저희는 병목을 전혀 사용하지 않도록 권고됩니다.

저희는 관심 있는 독자가 :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`을 자세히 검토하여 다양한 양의 계산에 대한 특정 네트워크 설계의 추가 세부 사항을 권장합니다. 예를 들어, 효과적인 32층 RegNetX 변형은 $k = 1$(병목 없음), $g = 16$(그룹 너비는 16), 첫 번째와 두 번째 스테이지에 각각 $c_1 = 32$와 $c_2 = 80$ 채널로 주어지며, $d_1=4$와 $d_2=6$ 블록 깊이로 선택되었습니다. 그 설계에서 놀라운 통찰은 이것이 더 큰 규모에서 네트워크를 조사할 때에도 여전히 적용된다는 것입니다. 더 좋은 점은, 전역 채널 활성화를 가진 Squeeze-and-Excitation (SE) 네트워크 설계(RegNetY)에도 적용됩니다 :cite:`Hu.Shen.Sun.2018`.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

각 RegNetX 스테이지가 점진적으로 해상도를 줄이고 출력 채널을 증가시키는 것을 볼 수 있습니다.

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## 학습

Fashion-MNIST 데이터셋에서 32층 RegNetX를 학습시키는 것은 이전과 동일합니다.

```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## 논의

비전을 위한 지역성과 이동 불변성(:numref:`sec_why-conv`)과 같은
바람직한 귀납 편향(가정 또는 선호도)을 가진 CNN은 이 영역에서 지배적인 아키텍처였습니다. 이는 트랜스포머(:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training`가 정확도 측면에서 CNN을 능가하기 시작할 때까지 LeNet부터 유지되었습니다. 비전 트랜스포머 측면에서의 최근 진보의 많은 부분이 CNN으로 *역포팅*될 수 *있지만* :cite:`liu2022convnet`, 그것은 더 높은 계산 비용에서만 가능합니다. 똑같이 중요한 것은, 최근의 하드웨어 최적화(NVIDIA Ampere 및 Hopper)가 트랜스포머에 유리한 격차를 더 넓혀왔다는 것입니다.

트랜스포머는 CNN보다 지역성과 이동 불변성에 대한 귀납 편향이 상당히 낮다는 점에 주목할 가치가 있습니다. 학습된 구조가 우세했다는 것은 적게는 LAION-400m과 LAION-5B :cite:`schuhmann2022laion`와 같이 최대 50억 개의 이미지를 가진 대규모 이미지 컬렉션의 가용성 덕분입니다. 꽤 놀랍게도, 이 맥락에서 더 관련성 있는 일부 작업에는 MLP :cite:`tolstikhin2021mlp`도 포함됩니다.

요약하자면, 비전 트랜스포머(:numref:`sec_vision-transformer`)는 이제 대규모 이미지 분류에서
최첨단 성능 측면에서 선두를 달리며,
*확장성이 귀납 편향을 압도한다*는 것을 보여줍니다 :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
이는 멀티-헤드 셀프-어텐션(:numref:`sec_multihead-attention`)을 가진 대규모 트랜스포머 사전 학습(:numref:`sec_large-pretraining-transformers`)을 포함합니다. 훨씬 더 자세한 논의를 위해 독자가 이러한 장들을 깊이 살펴보기를 권장합니다.

## 연습문제

1. 스테이지의 수를 4개로 늘리세요. 더 잘 수행되는 더 깊은 RegNetX를 설계할 수 있나요?
1. ResNeXt 블록을 ResNet 블록으로 대체하여 RegNets를 De-ResNeXt 화하세요. 새로운 모델은 어떻게 수행되나요?
1. RegNetX의 설계 원칙을 *위반*하여 "VioNet" 패밀리의 여러 인스턴스를 구현하세요. 어떻게 수행되나요? ($d_i$, $c_i$, $g_i$, $b_i$) 중 어느 것이 가장 중요한 요소인가요?
1. 여러분의 목표는 "완벽한" MLP를 설계하는 것입니다. 위에서 소개한 설계 원칙을 사용하여 좋은 아키텍처를 찾을 수 있나요? 작은 네트워크에서 큰 네트워크로 외삽하는 것이 가능한가요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8738)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18009)
:end_tab:

