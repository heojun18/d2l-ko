```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 심층 합성곱 신경망 (AlexNet)
:label:`sec_alexnet`


LeNet :cite:`LeCun.Jackel.Bottou.ea.1995`의 도입 이후 컴퓨터 비전 및 머신러닝 커뮤니티에서
CNN이 잘 알려져 있었지만,
이들이 즉시 이 분야를 지배하지는 못했습니다.
LeNet은 초기의 작은 데이터셋에서 좋은 결과를 얻었지만,
더 크고 더 현실적인 데이터셋에서 CNN을 학습시키는 성능과 실행 가능성은
아직 확립되지 않았습니다.
실제로 1990년대 초반부터
2012년의 분수령적 결과 :cite:`Krizhevsky.Sutskever.Hinton.2012` 사이의 많은 시간 동안,
신경망은 종종 커널 방법 :cite:`Scholkopf.Smola.2002`, 앙상블 방법 :cite:`Freund.Schapire.ea.1996`,
구조적 추정 :cite:`Taskar.Guestrin.Koller.2004`과 같은 다른 머신러닝 방법에 의해 능가되었습니다.

컴퓨터 비전의 경우, 이 비교가 완전히 정확하지는 않을 수 있습니다.
즉, 합성곱 네트워크의 입력은
원시 또는 가볍게 처리된(예: 중심화에 의한) 픽셀 값으로 구성되지만, 실무자들은 절대 원시 픽셀을 전통적인 모델에 입력하지 않습니다.
대신, 일반적인 컴퓨터 비전 파이프라인은
SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, 시각적 단어 가방 :cite:`Sivic.Zisserman.2003`과 같은 특성 추출 파이프라인을 수작업으로 설계하는 것으로 구성되었습니다.
특성을 *학습*하는 대신, 특성은 *수작업으로 만들어졌습니다*.
대부분의 발전은 한편으로는 특성 추출에 대한 더 영리한 아이디어와 다른 한편으로는 기하학에 대한 깊은 통찰 :cite:`Hartley.Zisserman.2000`에서 비롯되었습니다. 학습 알고리즘은 종종 부차적인 것으로 간주되었습니다.

1990년대에 일부 신경망 가속기를 사용할 수 있었지만,
이들은 많은 수의 파라미터를 가진 심층 다채널, 다층 CNN을 만들기에
충분히 강력하지 않았습니다. 예를 들어, 1999년 NVIDIA의 GeForce 256은
게임 외의 작업을 위한 의미 있는
프로그래밍 프레임워크 없이 초당 최대 4억 8천만 부동 소수점 연산(덧셈과 곱셈 등) (MFLOPS)을 처리할 수 있었습니다. 오늘날의 가속기는 장치당 1000 TFLOPs를 초과하여 수행할 수 있습니다.
더욱이, 데이터셋은 여전히 비교적 작았습니다: $28 \times 28$ 픽셀의 저해상도 이미지 60,000개에 대한 OCR이 매우 도전적인 작업으로 간주되었습니다.
이러한 장애물에 더해, 파라미터 초기화 휴리스틱 :cite:`Glorot.Bengio.2010`,
영리한 확률적 경사 하강법 변형 :cite:`Kingma.Ba.2014`,
비압축 활성화 함수 :cite:`Nair.Hinton.2010`,
효과적인 정규화 기법 :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`을 포함한 신경망 학습을 위한 주요 트릭이 여전히 빠져 있었습니다.

따라서 *종단 간*(픽셀에서 분류까지) 시스템을 학습시키는 대신,
고전적인 파이프라인은 다음과 같이 보였습니다.

1. 흥미로운 데이터셋을 얻습니다. 초기에는 이러한 데이터셋이 비싼 센서를 필요로 했습니다. 예를 들어, 1994년 [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake)은 0.3 메가픽셀(VGA) 해상도라는 엄청난 성능을 자랑했으며, 최대 8장의 이미지를 저장할 수 있었고, 모두 \$1000의 가격이었습니다.
1. 광학, 기하학, 기타 분석 도구에 대한 지식, 그리고 때로는 운 좋은 대학원생의 우연한 발견을 기반으로 한 수작업 특성으로 데이터셋을 전처리합니다.
1. SIFT(scale-invariant feature transform) :cite:`Lowe.2004`, SURF(speeded up robust features) :cite:`Bay.Tuytelaars.Van-Gool.2006`, 또는 다른 여러 수동 조정 파이프라인과 같은 표준 특성 추출기 세트를 통해 데이터를 처리합니다. OpenCV는 오늘날까지도 SIFT 추출기를 제공합니다!
1. 결과로 나온 표현을 좋아하는 분류기(일반적으로 선형 모델 또는 커널 방법)에 입력하여 분류기를 학습시킵니다.

머신러닝 연구자들과 이야기해보면,
그들은 머신러닝이 중요하고도 아름답다고 답할 것입니다.
우아한 이론은 다양한 분류기의 속성을 증명했고 :cite:`boucheron2005theory`, 볼록
최적화 :cite:`Boyd.Vandenberghe.2004`는 이를 얻는 데 있어 주된 방법이 되었습니다.
머신러닝 분야는 번창했고, 엄격했으며, 매우 유용했습니다. 그러나
컴퓨터 비전 연구자와 이야기해보면,
매우 다른 이야기를 들을 것입니다.
이미지 인식의 더러운 진실은,
새로운 학습 알고리즘이 아니라 특성, 기하학 :cite:`Hartley.Zisserman.2000,hartley2009global`, 그리고 엔지니어링이
발전을 이끌었다는 것이라고 그들은 말할 것입니다.
컴퓨터 비전 연구자들은
약간 더 크거나 깨끗한 데이터셋이나
약간 개선된 특성 추출 파이프라인이
어떤 학습 알고리즘보다 최종 정확도에 훨씬 더 중요하다고 정당하게 믿었습니다.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 표현 학습

상황을 다른 방식으로 표현하자면,
파이프라인에서 가장 중요한 부분은 표현이었습니다.
그리고 2012년까지 표현은 대부분 기계적으로 계산되었습니다.
실제로, 새로운 특성 함수 세트를 설계하고, 결과를 개선하고, 방법을 작성하는 것
모두가 논문에서 두드러지게 다루어졌습니다.
SIFT :cite:`Lowe.2004`,
SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`,
HOG(histograms of oriented gradient) :cite:`Dalal.Triggs.2005`,
시각적 단어 가방 :cite:`Sivic.Zisserman.2003`,
그리고 유사한 특성 추출기가 지배하고 있었습니다.

Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari, Juergen Schmidhuber를 포함한
또 다른 연구자 그룹은
다른 계획을 가지고 있었습니다.
그들은 특성 자체가 학습되어야 한다고 믿었습니다.
또한, 합리적으로 복잡하려면,
특성은 여러 개의 공동으로 학습된 층(각각 학습 가능한 파라미터가 있는)으로
계층적으로 구성되어야 한다고 믿었습니다.
이미지의 경우, 가장 낮은 층은
동물의 시각 시스템이 입력을 처리하는 방식과 유사하게
가장자리, 색상, 질감을 감지하게 될 수 있습니다. 특히, 희소 코딩 :cite:`olshausen1996emergence`에 의해 얻은 것과 같은
시각 특성의 자동 설계는 현대 CNN이 등장하기 전까지 여전히 미해결 과제였습니다.
이미지 데이터로부터 자동으로 특성을 생성한다는 아이디어가 상당한 추진력을 얻은 것은
:citet:`Dean.Corrado.Monga.ea.2012,le2013building`에 이르러서였습니다.

발명자 중 한 명인 Alex Krizhevsky의 이름을 따서
*AlexNet*이라고 명명된 첫 번째 현대 CNN :cite:`Krizhevsky.Sutskever.Hinton.2012`은 LeNet에 비해
주로 점진적인 개선입니다. 이는 2012년 ImageNet 챌린지에서 우수한 성능을 달성했습니다.

![AlexNet의 첫 번째 층이 학습한 이미지 필터. :citet:`Krizhevsky.Sutskever.Hinton.2012`의 제공에 따라 재현됨.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

흥미롭게도, 네트워크의 가장 낮은 층에서,
모델은 일부 전통적인 필터와 유사한 특성 추출기를 학습했습니다.
:numref:`fig_filters`는
낮은 수준의 이미지 기술자를 보여줍니다.
네트워크의 더 높은 층은 이러한 표현 위에
눈, 코, 풀잎 등과 같이 더 큰 구조를 나타내기 위해 구축될 수 있습니다.
그보다 더 높은 층은 사람, 비행기, 개, 또는 프리스비와 같은
전체 객체를 표현할 수 있습니다.
궁극적으로, 최종 은닉 상태는 이미지의 내용을 요약하는 압축된 표현을 학습하여
다른 범주에 속하는 데이터를 쉽게 분리할 수 있도록 합니다.

AlexNet(2012)과 그 선조인 LeNet(1995)은 많은 아키텍처 요소를 공유합니다. 이는 왜 그렇게 오래 걸렸을까라는 질문을 제기합니다.
주요 차이점은, 지난 20년 동안 사용 가능한 데이터의 양과 컴퓨팅 파워가 크게 증가했다는 것입니다. 그래서 AlexNet은 훨씬 더 컸습니다: 훨씬 더 많은 데이터와 1995년에 사용 가능했던 CPU에 비해 훨씬 더 빠른 GPU에서 학습되었습니다.

### 빠진 재료: 데이터

여러 층을 가진 심층 모델은 볼록 최적화(예: 선형 및 커널 방법)에 기반한
전통적인 방법을 크게 능가하는
영역에 진입하기 위해서는
많은 양의 데이터를 필요로 합니다.
그러나, 1990년대의 컴퓨터의 제한된 저장 용량,
(이미징) 센서의 상대적으로 높은 비용,
그리고 비교적 더 빠듯한 연구 예산을 고려할 때,
대부분의 연구는 작은 데이터셋에 의존했습니다.
수많은 논문이 UCI 데이터셋 컬렉션에 의존했는데,
그 중 많은 것은 저해상도로 캡처되고 종종 인위적으로 깨끗한 배경을 가진
수백 개 또는 (몇) 수천 개의 이미지만 포함했습니다.

2009년에 ImageNet 데이터셋이 출시되어 :cite:`Deng.Dong.Socher.ea.2009`,
연구자들에게 1000개의 서로 다른 객체 범주에서 각각 1000개씩,
100만 개의 예제로부터 모델을 학습하도록 도전했습니다. 범주 자체는
WordNet :cite:`Miller.1995`에서 가장 인기 있는 명사 노드를 기반으로 했습니다.
ImageNet 팀은 Google 이미지 검색을 사용하여 각 범주에 대한 대규모 후보 세트를 사전 필터링하고,
Amazon Mechanical Turk 크라우드소싱 파이프라인을
사용하여 각 이미지가 관련 범주에 속하는지 여부를 확인했습니다.
이 규모는 전례 없는 것으로, 다른 데이터셋(예: CIFAR-100은 60,000개의 이미지)을
한 자릿수 이상 초과했습니다. 또 다른 측면은 이미지가 80 million 크기의
TinyImages 데이터셋 :cite:`Torralba.Fergus.Freeman.2008`의 $32 \times 32$ 픽셀 썸네일과는 달리
비교적 높은 해상도인 $224 \times 224$ 픽셀이었다는 것입니다.
이는 더 높은 수준의 특성 형성을 가능하게 했습니다.
ImageNet Large Scale Visual Recognition
Challenge :cite:`russakovsky2015imagenet`라고 명명된 관련 대회는
컴퓨터 비전과 머신러닝 연구를 앞으로 밀어붙였고,
이전에 학자들이 고려했던 것보다 더 큰 규모에서
어떤 모델이 가장 잘 수행되는지 식별하도록 연구자들에게 도전했습니다. LAION-5B
:cite:`schuhmann2022laion`와 같은 가장 큰 비전 데이터셋은 추가 메타데이터와 함께 수십억 개의 이미지를 포함합니다.

### 빠진 재료: 하드웨어

딥러닝 모델은 계산 사이클의 탐욕스러운 소비자입니다.
학습은 수백 에포크가 걸릴 수 있고, 각 반복은
계산 비용이 많이 드는 선형 대수 연산의 많은 층을 통해
데이터를 전달해야 합니다.
이것이 1990년대와 2000년대 초반에
더 효율적으로 최적화된 볼록 목표에 기반한 단순한
알고리즘이 선호된 주요 이유 중 하나입니다.

*그래픽 처리 장치*(GPU)는 딥러닝을 실현 가능하게 만드는 데
판도를 바꾸는 것으로 입증되었습니다.
이러한 칩은 이전에 컴퓨터 게임에 도움이 되도록
그래픽 처리를 가속화하기 위해 개발되었습니다.
특히, 많은 컴퓨터 그래픽 작업에 필요한
높은 처리량의 $4 \times 4$ 행렬-벡터 곱에 최적화되어 있었습니다.
다행히도, 그 수학은 합성곱 층을 계산하는 데 필요한 것과
놀랍도록 유사합니다.
그 무렵, NVIDIA와 ATI는 GPU를 일반 계산 작업에 최적화하기 시작했고
:cite:`Fernando.2004`,
*범용 GPU*(GPGPU)로 마케팅할 정도까지 갔습니다.

직관을 제공하기 위해, 현대 마이크로프로세서
(CPU)의 코어를 고려해 보세요.
각 코어는 높은 클럭 주파수에서 실행되고
큰 캐시(최대 몇 메가바이트의 L3)를 갖춘 상당히 강력합니다.
각 코어는 분기 예측기, 깊은 파이프라인, 특수 실행 단위,
추측 실행, 그리고 다양한 부가 기능을 통해
복잡한 제어 흐름을 가진 다양한 프로그램을 실행할 수 있는
광범위한 명령어 세트를 실행하기에 적합합니다.
그러나 이 명백한 강점은 또한 그 아킬레스건이기도 합니다.
범용 코어는 만들기에 매우 비쌉니다. 이들은 많은 제어 흐름이 있는
범용 코드에서 뛰어납니다.
이는 계산이 일어나는 실제 ALU(산술 논리 단위)뿐만 아니라,
앞서 언급한 모든 부가 기능,
코어 간 메모리 인터페이스, 캐싱 로직,
고속 인터커넥트 등을 위한
많은 칩 면적을 필요로 합니다. CPU는
전용 하드웨어와 비교할 때 단일 작업에서는 비교적 좋지 않습니다.
현대 노트북은 4(8개의 코어)를 가지고 있으며,
하이엔드 서버조차도 단순히 비용 효율적이지 않기 때문에 소켓당 64개의 코어를 거의 초과하지 않습니다.

비교하자면, GPU는 수천 개의 작은 처리 요소(NIVIDA의 최신 Ampere 칩은 최대 6912개의 CUDA 코어를 가지고 있음)로 구성될 수 있으며, 종종 더 큰 그룹(NVIDIA는 워프라고 부름)으로 그룹화됩니다.
세부 사항은 NVIDIA, AMD, ARM 및 기타 칩 공급업체 간에 다소 다릅니다. 각 코어는 비교적 약하고,
약 1GHz의 클럭 주파수에서 실행되지만,
GPU를 CPU보다 몇 자릿수 더 빠르게 만드는 것은 이러한 코어의 총 수입니다.
예를 들어, NVIDIA의 최근 Ampere A100 GPU는 특수 16비트 정밀도(BFLOAT16) 행렬-행렬 곱셈을 위해 칩당 300 TFLOPs 이상을 제공하고, 더 범용 부동 소수점 연산(FP32)을 위해 최대 20 TFLOPs를 제공합니다.
동시에, CPU의 부동 소수점 성능은 1 TFLOPs를 거의 초과하지 않습니다. 예를 들어, Amazon의 Graviton 3는 16비트 정밀도 연산에 대해 최대 2 TFLOPs의 성능에 도달하며, 이는 Apple의 M1 프로세서의 GPU 성능과 유사한 수치입니다.

GPU가 FLOPs 측면에서 CPU보다 훨씬 빠른 데에는 많은 이유가 있습니다.
첫째, 전력 소비는 클럭 주파수에 따라 *이차적으로* 증가하는 경향이 있습니다.
따라서, 네 배 빠르게 실행되는 CPU 코어의 전력 예산(일반적인 수치)으로,
$\frac{1}{4}$의 속도로 16개의 GPU 코어를 사용할 수 있고,
이는 $16 \times \frac{1}{4} = 4$배의 성능을 산출합니다.
둘째, GPU 코어는 훨씬 더 단순하고
(실제로 오랫동안 범용 코드를 실행할 *수* 조차 없었습니다),
이는 이들을 더 에너지 효율적으로 만듭니다. 예를 들어, (i) 이들은 추측 실행을 지원하지 않는 경향이 있고, (ii) 일반적으로 각 처리 요소를 개별적으로 프로그래밍하는 것이 불가능하며, (iii) 코어당 캐시가 훨씬 더 작은 경향이 있습니다.
마지막으로, 딥러닝의 많은 연산은 높은 메모리 대역폭을 필요로 합니다.
다시 한번, GPU는 많은 CPU보다 최소 10배 넓은 버스로 빛을 발합니다.

2012년으로 돌아갑시다. Alex Krizhevsky와 Ilya Sutskever가
GPU에서 실행할 수 있는 심층 CNN을 구현했을 때
주요 돌파구가 찾아왔습니다.
그들은 CNN의 계산 병목인
합성곱과 행렬 곱셈이
모두 하드웨어에서 병렬화될 수 있는 연산이라는 것을 깨달았습니다.
3GB의 메모리를 가진 두 개의 NVIDIA GTX 580을 사용하여(둘 다 1.5 TFLOPs를 처리할 수 있음, 이는 10년 후에도 대부분의 CPU에서 여전히 도전적임),
그들은 빠른 합성곱을 구현했습니다.
[cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) 코드는
충분히 좋아서 몇 년 동안 업계 표준이었으며
딥러닝 붐의 첫 몇 년을 견인했습니다.

## AlexNet

8층 CNN을 채택한 AlexNet은,
2012 ImageNet Large Scale Visual Recognition Challenge에서
큰 차이로 우승했습니다 :cite:`Russakovsky.Deng.Huang.ea.2013`.
이 네트워크는 처음으로
학습을 통해 얻은 특성이 수동으로 설계된 특성을 초월할 수 있음을 보여주어, 컴퓨터 비전의 이전 패러다임을 깨뜨렸습니다.

AlexNet과 LeNet의 아키텍처는 놀랍도록 유사하며,
:numref:`fig_alexnet`이 보여줍니다.
2012년에 모델이 두 개의 작은 GPU에 맞도록 필요했던 일부 설계 특이점을 제거한
약간 간소화된 버전의 AlexNet을 제공하고 있음을 참고하세요.

![LeNet(왼쪽)에서 AlexNet(오른쪽)으로.](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet과 LeNet 사이에는 또한 상당한 차이가 있습니다.
첫째, AlexNet은 비교적 작은 LeNet-5보다 훨씬 깊습니다.
AlexNet은 8개의 층으로 구성됩니다: 5개의 합성곱 층,
2개의 완전 연결 은닉 층, 그리고 1개의 완전 연결 출력 층.
둘째, AlexNet은 활성화 함수로 시그모이드 대신 ReLU를 사용했습니다. 아래에서 세부 사항을 자세히 살펴보겠습니다.

### 아키텍처

AlexNet의 첫 번째 층에서, 합성곱 윈도우 모양은 $11\times11$입니다.
ImageNet의 이미지는 MNIST 이미지보다
8배 더 높고 넓기 때문에,
ImageNet 데이터의 객체는 더 많은 시각적 세부 사항으로 더 많은 픽셀을 차지하는 경향이 있습니다.
결과적으로, 객체를 캡처하기 위해 더 큰 합성곱 윈도우가 필요합니다.
두 번째 층의 합성곱 윈도우 모양은
$5\times5$로 축소되고, 그 다음에 $3\times3$이 옵니다.
또한, 첫 번째, 두 번째, 다섯 번째 합성곱 층 다음에,
네트워크는 윈도우 모양이 $3\times3$이고
스트라이드가 2인 최대 풀링 층을 추가합니다.
더욱이, AlexNet은 LeNet보다 10배 많은 합성곱 채널을 가지고 있습니다.

마지막 합성곱 층 뒤에는 4096개의 출력을 가진 두 개의 거대한
완전 연결 층이 있습니다.
이러한 층은 거의 1GB의 모델 파라미터를 필요로 합니다.
초기 GPU의 제한된 메모리 때문에,
원래 AlexNet은 듀얼 데이터 스트림 설계를 사용하여,
두 GPU가 각각 모델의 절반만 저장하고
계산할 수 있도록 했습니다.
다행히도, 현재 GPU 메모리는 비교적 풍부하므로,
요즘에는 GPU 간에 모델을 분할하는 경우가 거의 없습니다
(저희의 AlexNet 모델 버전은 이 측면에서
원래 논문에서 벗어납니다).

### 활성화 함수

더욱이, AlexNet은 시그모이드 활성화 함수를 더 단순한 ReLU 활성화 함수로 변경했습니다. 한편으로, ReLU 활성화 함수의 계산이 더 단순합니다. 예를 들어, 시그모이드 활성화 함수에서 발견되는 지수 연산이 없습니다.
다른 한편으로, ReLU 활성화 함수는 다른 파라미터 초기화 방법을 사용할 때 모델 학습을 더 쉽게 만듭니다. 이는 시그모이드 활성화 함수의 출력이 0 또는 1에 매우 가까울 때, 이러한 영역의 기울기가 거의 0이어서 역전파가 일부 모델 파라미터를 계속 업데이트할 수 없기 때문입니다. 대조적으로, 양의 구간에서 ReLU 활성화 함수의 기울기는 항상 1입니다 (:numref:`subsec_activation-functions`). 따라서 모델 파라미터가 적절하게 초기화되지 않으면, 시그모이드 함수는 양의 구간에서 거의 0의 기울기를 얻을 수 있으며, 이는 모델이 효과적으로 학습될 수 없음을 의미합니다.

### 용량 제어 및 전처리

AlexNet은 드롭아웃(:numref:`sec_dropout`)을 통해
완전 연결 층의 모델 복잡도를 제어하는 반면,
LeNet은 가중치 감쇠만 사용합니다.
데이터를 더욱 증강시키기 위해, AlexNet의 학습 루프는
뒤집기, 잘라내기, 색상 변경 등 많은 양의 이미지 증강을 추가했습니다.
이는 모델을 더 견고하게 만들고 더 큰 샘플 크기는 효과적으로 과적합을 줄입니다.
이러한 전처리 단계에 대한 심층 리뷰는 :citet:`Buslaev.Iglovikov.Khvedchenya.ea.2020`을 참조하세요.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class AlexNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=96, kernel_size=(11, 11), strides=4, padding=1),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=256, kernel_size=(5, 5)),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3)), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=self.num_classes)
        ])
```

(**각 층의 출력 모양을 관찰하기 위해**) 높이와 너비가 모두 224인 [**단일 채널 데이터 예제를 구성합니다**]. 이는 :numref:`fig_alexnet`의 AlexNet 아키텍처와 일치합니다.

```{.python .input  n=6}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
AlexNet(training=False).layer_summary((1, 224, 224, 1))
```

## 학습

AlexNet은 :citet:`Krizhevsky.Sutskever.Hinton.2012`에서 ImageNet으로 학습되었지만,
ImageNet 모델을 수렴까지 학습시키는 것은 현대 GPU에서도 몇 시간 또는 며칠이 걸릴 수 있으므로
여기서는 Fashion-MNIST를 사용합니다.
AlexNet을 [**Fashion-MNIST**]에 직접 적용할 때의 문제 중 하나는
그 (**이미지가 ImageNet 이미지보다 더 낮은 해상도($28 \times 28$ 픽셀)**)
(**를 가진다는 것입니다.**)
작동하게 하기 위해, (**저희는 이미지를 $224 \times 224$로 업샘플링합니다**).
이는 일반적으로 현명한 관행은 아니며, 정보를 추가하지 않고 계산
복잡도만 증가시키기 때문입니다. 그럼에도 불구하고, AlexNet 아키텍처에 충실하기 위해 여기서 그렇게 합니다.
저희는 `d2l.FashionMNIST` 생성자의 `resize` 인수로 이 크기 조정을 수행합니다.

이제, [**AlexNet 학습을 시작할 수 있습니다.**]
:numref:`sec_lenet`의 LeNet과 비교하여,
여기서 주요 변경 사항은 더 작은 학습률을 사용하는 것과
더 깊고 넓은 네트워크, 더 높은 이미지 해상도, 그리고 더 비싼 합성곱으로 인한
훨씬 더 느린 학습입니다.

```{.python .input  n=8}
%%tab pytorch, mxnet, jax
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## 논의

AlexNet의 구조는 정확도(드롭아웃)와 학습 용이성(ReLU) 모두를 위한 여러 중요한 개선과 함께 LeNet과 놀라울 정도로 유사합니다. 똑같이 놀라운 것은 딥러닝 도구 측면에서 이루어진 발전의 양입니다. 2012년에 몇 달의 작업이었던 것이 이제는 어떤 현대 프레임워크를 사용해서도 수십 줄의 코드로 달성될 수 있습니다.

아키텍처를 검토해 보면, AlexNet이 효율성 측면에서 아킬레스건을 가지고 있음을 알 수 있습니다: 마지막 두 은닉 층은 각각 $6400 \times 4096$ 및 $4096 \times 4096$ 크기의 행렬을 필요로 합니다. 이는 164 MB의 메모리와 81 MFLOPs의 계산에 해당하며, 둘 다 특히 휴대폰과 같은 작은 장치에서는 사소하지 않은 비용입니다. 이것이 AlexNet이 다음 절에서 다룰 훨씬 더 효과적인 아키텍처에 의해 능가된 이유 중 하나입니다. 그럼에도 불구하고, 이는 오늘날 사용되는 얕은 네트워크에서 심층 네트워크로의 중요한 단계입니다. 저희 실험에서 파라미터의 수가 학습 데이터의 양을 훨씬 초과하더라도(마지막 두 층은 4천만 개 이상의 파라미터를 가지며, 6만 개의 이미지로 구성된 데이터셋에서 학습됨), 과적합은 거의 없음에 주목하세요: 학습 손실과 검증 손실은 학습 전반에 걸쳐 사실상 동일합니다. 이는 현대 심층 네트워크 설계에 내재된 드롭아웃과 같은 향상된 정규화 덕분입니다.

AlexNet의 구현이 LeNet보다 몇 줄 더 많은 것처럼 보이지만, 학계가 이 개념적 변화를 수용하고 그 우수한 실험 결과를 활용하는 데는 수년이 걸렸습니다. 이는 또한 효율적인 계산 도구의 부족 때문이기도 했습니다. 당시에는 DistBelief :cite:`Dean.Corrado.Monga.ea.2012`도 Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`도 존재하지 않았고, Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`는 여전히 많은 구별되는 기능이 부족했습니다. 상황을 극적으로 변화시킨 것은 TensorFlow :cite:`Abadi.Barham.Chen.ea.2016`의 가용성이었습니다.

## 연습문제

1. 위의 논의를 이어가며, AlexNet의 계산 속성을 분석하세요.
    1. 합성곱과 완전 연결 층에 대한 메모리 사용량을 각각 계산하세요. 어느 것이 지배적인가요?
    1. 합성곱과 완전 연결 층의 계산 비용을 계산하세요.
    1. 메모리(읽기 및 쓰기 대역폭, 지연 시간, 크기)는 계산에 어떤 영향을 미치나요? 학습과 추론에 대해 그 영향에 차이가 있나요?
1. 여러분은 칩 설계자이며 계산과 메모리 대역폭 사이의 트레이드오프를 해야 합니다. 예를 들어, 더 빠른 칩은 더 많은 전력과 아마도 더 큰 칩 면적을 필요로 합니다. 더 많은 메모리 대역폭은 더 많은 핀과 제어 로직, 따라서 더 많은 면적을 필요로 합니다. 어떻게 최적화하시겠어요?
1. 엔지니어들이 더 이상 AlexNet에 대한 성능 벤치마크를 보고하지 않는 이유는 무엇인가요?
1. AlexNet을 학습할 때 에포크 수를 늘려보세요. LeNet과 비교하여 결과가 어떻게 다른가요? 왜 그런가요?
1. AlexNet은 Fashion-MNIST 데이터셋에 비해 너무 복잡할 수 있으며, 특히 초기 이미지의 낮은 해상도 때문입니다.
    1. 정확도가 크게 떨어지지 않도록 하면서 학습이 더 빨라지도록 모델을 단순화해 보세요.
    1. $28 \times 28$ 이미지에서 직접 작동하는 더 나은 모델을 설계하세요.
1. 배치 크기를 수정하고, 처리량(이미지/초), 정확도, 그리고 GPU 메모리의 변화를 관찰하세요.
1. LeNet-5에 드롭아웃과 ReLU를 적용하세요. 개선이 있나요? 이미지에 내재된 불변성을 활용하기 위한 전처리로 더 개선할 수 있나요?
1. AlexNet을 과적합시킬 수 있나요? 학습을 중단시키기 위해 제거하거나 변경해야 하는 특성은 무엇인가요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18001)
:end_tab:
