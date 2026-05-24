# 현대 합성곱 신경망
:label:`chap_modern_cnn`

이제 CNN을 함께 연결하는 기초를 이해했으니, 현대 CNN 아키텍처를 둘러보겠습니다. 흥미로운 새로운 설계가 너무 많이 추가되고 있어 이 둘러보기는 필연적으로 불완전할 수밖에 없습니다. 이들의 중요성은 비전 작업에 직접 사용될 수 있을 뿐만 아니라, 추적
:cite:`Zhang.Sun.Jiang.ea.2021`, 세그멘테이션 :cite:`Long.Shelhamer.Darrell.2015`, 객체
검출 :cite:`Redmon.Farhadi.2018`, 또는 스타일 변환
:cite:`Gatys.Ecker.Bethge.2016`과 같은 더 고급 작업을 위한 기본 특성 생성기로도 사용된다는 사실에서 비롯됩니다. 이 장에서, 대부분의 절은
어느 시점(또는 현재)에서 많은 연구 프로젝트와
배포된 시스템이 구축된 기반 모델이었던 중요한 CNN 아키텍처에 해당합니다. 이러한 각 네트워크는 한때
지배적인 아키텍처였으며 많은 네트워크는 2010년 이후 컴퓨터 비전 분야의 지도 학습 발전의 척도 역할을 해 온
[ImageNet 대회](https://www.image-net.org/challenges/LSVRC/)의
우승자 또는 준우승자였습니다. 트랜스포머가 CNN을 대체하기 시작한 것은 비교적 최근의 일로,
:citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`을 시작으로 Swin Transformer :cite:`liu2021swin`가 그 뒤를 이었습니다. 이러한 발전은 나중에
:numref:`chap_attention-and-transformers`에서 다루겠습니다.

*심층* 신경망의 아이디어는 매우 단순(여러 층을 쌓는 것)하지만,
성능은 아키텍처와 하이퍼파라미터 선택에 따라 크게 달라질 수 있습니다. 이 장에서 설명하는
신경망은 직관, 약간의 수학적 통찰, 그리고 많은 시행착오의 산물입니다. 저희는 이러한
모델을 시대순으로 제시하는데, 이는 부분적으로 역사적 감각을 전달하여
여러분이 이 분야가 어디로 향하고 있는지에 대한 자신만의 직관을 형성하고 어쩌면 자신만의 아키텍처를 개발할 수 있도록 하기 위함입니다.
예를 들어, 이 장에서 설명하는
배치 정규화와 잔차 연결은
심층 모델의 학습과 설계를 위한 두 가지 인기 있는 아이디어를 제공했으며,
이 두 가지는 모두 이후 컴퓨터 비전을 넘어선 아키텍처에도 적용되었습니다.

저희는 현대 CNN 둘러보기를 대규모 비전 챌린지에서 기존의 컴퓨터
비전 방법을 이긴 최초의 대규모 네트워크인 AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`로 시작합니다;
요소의 반복 블록을 여러 개 활용하는 VGG 네트워크
:cite:`Simonyan.Zisserman.2014`; 입력에 대해 패치 단위로 전체 신경망을 합성곱하는
network in network (NiN)
:cite:`Lin.Chen.Yan.2013`; 다중 분기 합성곱을 가진 네트워크를 사용하는
GoogLeNet :cite:`Szegedy.Liu.Jia.ea.2015`; 컴퓨터 비전에서
가장 인기 있는 기성 아키텍처 중 하나로 남아 있는 잔차
네트워크(ResNet) :cite:`He.Zhang.Ren.ea.2016`;
더 희소한 연결을 위한 ResNeXt 블록 :cite:`Xie.Girshick.Dollar.ea.2017`;
그리고 잔차 아키텍처의 일반화를 위한 DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017`을 다룹니다. 시간이 지나면서 효율적인
네트워크를 위한 많은 특수한 최적화 기법이 개발되었는데, 좌표 이동(ShiftNet) :cite:`wu2018shift`과 같은 것입니다. 이는
MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`와 같은 효율적인 아키텍처에 대한 자동 탐색에서 정점을 이루었습니다. 또한
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020`의 반자동 설계 탐색을 포함하는데,
이는 이 장의 뒷부분에서 다룰 RegNetX/Y로 이어졌습니다.
이 작업은 효율적인 설계 공간을 찾는 데 있어 무차별 대입 계산과
실험자의 독창성을 결합하는 길을 제시한다는 점에서 시사적입니다. 또한 주목할 만한 것은
:citet:`liu2022convnet`의 작업입니다. 이는 학습 기법(예: 옵티마이저, 데이터 증강, 정규화)이
정확도 향상에 핵심적인 역할을 한다는 것을 보여주기 때문입니다. 또한 합성곱 윈도우의 크기와 같은
오랫동안 유지되어 온 가정들이 계산 능력과 데이터 증가를 고려할 때
재검토되어야 할 수도 있음을 보여줍니다. 저희는 이 장 전반에 걸쳐 이러한 문제와 더 많은 질문을 적절한 시점에 다룰 것입니다.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```

