# 머리말

불과 몇 년 전만 해도, 주요 기업과 스타트업에서
지능형 제품과 서비스를 개발하는
딥러닝 연구자 군단은 존재하지 않았습니다.
저희가 이 분야에 들어왔을 때, 머신러닝은
일간지 헤드라인을 장식하는 주제가 아니었습니다.
부모님은 머신러닝이 무엇인지조차 모르셨고,
저희가 의학이나 법학 대신 왜 이 길을 택했는지는
더더욱 이해하지 못하셨습니다.
머신러닝은 음성 인식과 컴퓨터 비전을 비롯한
좁은 실제 응용 분야에서만
산업적 의미를 가지던
순수 학문 분야였습니다.
게다가 이러한 응용 분야의 상당수는
도메인 지식이 너무 많이 필요해서
머신러닝이 하나의 작은 구성 요소에 불과한
별개의 영역으로 여겨지곤 했습니다.
당시 신경망(이 책에서 다루는
딥러닝 기법의 전신)은
일반적으로 시대에 뒤떨어진 것으로
간주되었습니다.


그러나 불과 몇 년 만에 딥러닝은 세상을 놀라게 했고,
컴퓨터 비전, 자연어 처리,
자동 음성 인식, 강화 학습,
생의학 정보학 등 다양한 분야에서
급격한 발전을 이끌었습니다.
나아가 실용적으로 중요한
수많은 과제에서 거둔 딥러닝의 성공은
이론 머신러닝과 통계학의
발전까지 촉진했습니다.
이러한 진전 덕분에 우리는 이제
그 어느 때보다 자율적으로 주행하는 자동차
(다만 일부 기업이 주장하는 만큼 자율적이지는 않은),
명확화 질문을 던지며 코드를 디버깅하는 대화 시스템,
한때 수십 년은 더 걸릴 것으로 여겨졌던 바둑 같은 보드 게임에서 세계 최고의 인간 기사를 이긴 소프트웨어 에이전트까지 만들어낼 수 있게 되었습니다.
이러한 도구들은 이미 산업과 사회 전반에 점점 더 폭넓은 영향을 미치며,
영화 제작 방식과 질병 진단 방식을 바꾸고,
천체물리학, 기후 모델링, 일기 예보, 생의학에 이르는 기초 과학에서도 점점 더 큰 역할을 하고 있습니다.



## 이 책에 대하여

이 책은 *개념*과 *맥락*, *코드*를 함께 가르치며
딥러닝을 누구나 다가갈 수 있는 것으로 만들고자 하는 저희의 시도입니다.

### 코드, 수학, HTML을 통합한 하나의 매체

어떤 컴퓨팅 기술이든 그 영향력을 온전히 발휘하려면,
잘 이해되고, 잘 문서화되어 있어야 하며,
성숙하고 잘 유지되는 도구의 뒷받침이 있어야 합니다.
핵심 아이디어는 명확하게 정제되어,
새로운 사용자가 최신 내용을 따라잡는 데 필요한
진입 시간을 최소화해야 합니다.
성숙한 라이브러리는 흔한 작업을 자동화해야 하고,
예제 코드는 사용자가 자신의 필요에 맞게
일반적인 응용 사례를 수정하고, 적용하고, 확장하기 쉽도록 해 줘야 합니다.


예를 들어 동적 웹 애플리케이션을 살펴봅시다.
1990년대에 아마존을 비롯한 수많은 기업이
데이터베이스 기반의 성공적인 웹 애플리케이션을 개발했지만,
이 기술이 창의적인 기업가들을 돕는 잠재력은
강력하고 잘 문서화된 프레임워크들의 등장에 힘입어
지난 10년에 와서야 비로소 훨씬 더 큰 폭으로 실현되었습니다.


딥러닝의 잠재력을 시험하는 일은 독특한 도전 과제를 제시합니다.
단 하나의 응용 사례에도 여러 분야가 한데 모이기 때문입니다.
딥러닝을 적용하려면 다음을 동시에 이해해야 합니다.
(i) 특정한 방식으로 문제를 정식화하게 된 동기,
(ii) 주어진 모델의 수학적 형태,
(iii) 모델을 데이터에 맞추기 위한 최적화 알고리즘,
(iv) 모델이 본 적 없는 데이터에 대해
언제 일반화될 것이라 기대할 수 있는지를 알려주는
통계적 원리와, 모델이 실제로 일반화되었음을
입증하기 위한 실용적 방법,
(v) 수치 계산의 함정을 피하고
가용 하드웨어를 최대한 활용하면서
효율적으로 모델을 학습시키기 위한
엔지니어링 기법.
문제를 정식화하는 데 필요한 비판적 사고 능력,
그것을 풀어내는 데 필요한 수학,
그 해법을 구현하는 데 필요한 소프트웨어 도구를
한곳에서 가르치는 일은 만만치 않은 과제입니다.
이 책에서 저희의 목표는 미래의 사용자들이
빠르게 본궤도에 오를 수 있도록 통합된 자료를 제공하는 것입니다.

저희가 이 책 프로젝트를 시작했을 때,
다음을 동시에 충족하는 자료는 없었습니다.
(i) 최신성을 유지하면서,
(ii) 충분한 기술적 깊이로
현대 머신러닝 실무의 전반을 다루고,
(iii) 교과서에 기대할 법한 수준의 설명과
실습 튜토리얼에 기대할 법한 깔끔하고 실행 가능한 코드를
함께 엮어 제시하는 자료.
저희는 특정 딥러닝 프레임워크의 사용법을 보여 주거나
(예: 텐서플로에서 행렬로 기본적인 수치 계산을 하는 법),
특정 기법을 구현하는
(예: LeNet, AlexNet, ResNet 등의 코드 스니펫)
수많은 코드 예제가 여러 블로그 글과 깃허브 리포지토리에
흩어져 있는 것을 발견했습니다.
그러나 이러한 예제들은 대체로
주어진 접근법을 *어떻게* 구현하는지에 초점을 맞췄을 뿐,
*왜* 특정한 알고리즘적 결정이 내려졌는지에 대한
논의는 빠져 있었습니다.
특정 주제를 다루는 인터랙티브 자료들이
간간이 등장하기도 했습니다.
예를 들어 [Distill](http://distill.pub) 웹사이트에 게시된
흥미로운 블로그 글이나 개인 블로그가 그러한데,
이들은 딥러닝의 일부 선택된 주제만 다루었고,
관련 코드가 없는 경우도 많았습니다.
한편으로, 여러 딥러닝 교과서가 등장했고
(예를 들어 :citet:`Goodfellow.Bengio.Courville.2016`은
딥러닝의 기초에 대한 포괄적인 개관을 제공합니다)
이러한 자료들은 설명과 개념의 코드 구현을 결합하지 않아서
독자가 어떻게 구현해야 할지 막막하게 만드는 경우가 있었습니다.
게다가 너무 많은 자료가
상용 강의 제공업체의 유료 결제 장벽 뒤에 숨어 있습니다.

저희는 다음과 같은 자료를 만들고자 했습니다.
(i) 누구나 무료로 이용할 수 있고,
(ii) 실제로 응용 머신러닝 과학자가 되기 위한
출발점을 제공할 만큼의
충분한 기술적 깊이를 갖추며,
(iii) 실제로 문제를 *어떻게* 풀어야 하는지를 독자에게 보여주는
실행 가능한 코드를 포함하고,
(iv) 저희와 커뮤니티 전반이
빠르게 업데이트할 수 있고,
(v) 기술적 세부 사항을 상호 토론하고 질문에 답할 수 있는
[포럼](https://discuss.d2l.ai/c/5)으로 보완되는 자료.

이 목표들은 종종 서로 충돌했습니다.
수식, 정리, 인용은 LaTeX으로
관리하고 조판하는 것이 가장 좋습니다.
코드는 파이썬으로 기술하는 것이 가장 좋습니다.
웹 페이지는 HTML과 자바스크립트에 본래 기반을 둡니다.
나아가 저희는 콘텐츠가 실행 가능한 코드로,
인쇄된 책으로, 내려받을 수 있는 PDF로,
그리고 인터넷의 웹사이트로 모두 접근 가능하기를 원했습니다.
이러한 요구에 들어맞는 워크플로는 어디에도 없어 보였기에,
저희는 직접 워크플로를 구성하기로 결정했습니다(:numref:`sec_how_to_contribute`).
소스 공유와 커뮤니티 기여를 돕기 위해 깃허브를,
코드·수식·텍스트를 함께 담기 위해 주피터 노트북을,
렌더링 엔진으로는 Sphinx를,
토론 플랫폼으로는 Discourse를 채택했습니다.
저희 시스템이 완벽하지는 않지만,
이러한 선택은 서로 경쟁하는 요구사항들 사이에서 절충점을 만들어 줍니다.
저희는 *Dive into Deep Learning*이
이러한 통합 워크플로로 출판된
최초의 책일지도 모른다고 생각합니다.


### 직접 해보면서 배우기

많은 교과서는 개념을 차례차례 제시하며
각각을 빠짐없이 자세하게 다룹니다.
예를 들어
:citet:`Bishop.2006`의 훌륭한 교과서는
각 주제를 너무나 철저하게 다뤄서
선형 회귀 챕터에 도달하는 데에도
적지 않은 노력이 필요합니다.
전문가들은 바로 그 철저함 때문에 이 책을 좋아하지만,
진정한 초심자에게는 이러한 특성이
입문서로서의 유용성을 떨어뜨립니다.

이 책에서 저희는 대부분의 개념을 *적시에*(just in time) 가르칩니다.
다시 말해, 여러분은 어떤 실용적인 목적을 이루기 위해
개념이 필요해지는 바로 그 순간에 그것을 배우게 됩니다.
시작 부분에서 선형대수와 확률 같은
기본적인 사전 지식을 가르치는 데 어느 정도 시간을 들이긴 하지만,
저희는 여러분이 더 난해한 개념을 걱정하기 전에
자신의 첫 모델을 학습시켜 보는 성취감을 먼저 맛보기를 바랍니다.

기본적인 수학적 배경을 속성으로 다루는 몇 개의 준비용 노트북을 제외하면,
이후의 각 챕터는 적절한 수의 새로운 개념을 소개하면서
실제 데이터셋을 사용한 자기 완결적인 작동 예제 몇 개를 함께 제공합니다.
이는 구성상의 과제를 안겨주었습니다.
어떤 모델들은 논리적으로 하나의 노트북에 함께 묶일 수도 있고,
어떤 아이디어는 여러 모델을 차례로 실행해 보는 방식으로
가장 잘 가르칠 수 있을지도 모릅니다.
이와 반대로, *작동 예제 하나, 노트북 하나*라는 원칙을 고수하는 데에는
큰 장점이 있습니다.
이렇게 하면 여러분이 저희 코드를 활용해 자신의 연구 프로젝트를
시작하기가 최대한 쉬워집니다.
노트북을 복사해서 수정을 시작하기만 하면 됩니다.

책 전반에 걸쳐 저희는 실행 가능한 코드와 배경 지식을
필요에 따라 교차로 배치합니다.
일반적으로 저희는 도구를 충분히 설명하기 전에
먼저 사용할 수 있게 제공하는 쪽을 택하곤 합니다
(배경 지식은 나중에 채워 넣는 경우가 많습니다).
예를 들어 *확률적 경사 하강법*을 그것이 왜 유용한지 설명하거나
작동 원리에 대한 직관을 제공하기 전에 사용할 수도 있습니다.
이러한 방식은 사용자에게 문제를 빠르게 풀 수 있는
필요한 무기를 쥐어 주는 데 도움이 되지만,
독자가 저희의 몇몇 편집상의 결정을 믿고 따라야 한다는 대가가 따릅니다.

이 책은 딥러닝 개념을 밑바닥부터 가르칩니다.
때로는 현대 딥러닝 프레임워크가 보통은 사용자에게 감추는
모델의 세부 사항까지 깊이 파고듭니다.
특히 기본 튜토리얼에서 이런 일이 자주 일어나는데,
주어진 층(layer)이나 옵티마이저 안에서 일어나는 모든 일을
여러분이 이해하기를 원하기 때문입니다.
이러한 경우에 저희는 종종 예제를 두 가지 버전으로 제시합니다.
하나는 NumPy 류의 기능과 자동 미분에만 의존해
모든 것을 밑바닥부터 구현하는 버전이고,
다른 하나는 딥러닝 프레임워크의 고수준 API를 사용해
간결한 코드를 작성하는 더 실용적인 예제입니다.
어떤 구성 요소가 어떻게 동작하는지 설명한 뒤에는,
이어지는 튜토리얼에서 고수준 API에 의존합니다.


### 내용과 구성

이 책은 대략 세 부분으로 나눌 수 있으며,
각각 사전 준비 내용,
딥러닝 기법,
그리고 실제 시스템과 응용에 초점을 맞춘
고급 주제를 다룹니다(:numref:`fig_book_org`).

![책의 구성.](../img/book-org.svg)
:label:`fig_book_org`


* **1부: 기초와 사전 준비**.
:numref:`chap_introduction`은
딥러닝에 대한 소개입니다.
그다음 :numref:`chap_preliminaries`에서는
데이터를 저장하고 조작하는 법,
그리고 선형대수, 미적분, 확률의 기본 개념을 바탕으로
다양한 수치 연산을 적용하는 법 등
실습 중심의 딥러닝에 필요한 사전 지식을
빠르게 따라잡을 수 있게 안내합니다.
:numref:`chap_regression`과 :numref:`chap_perceptrons`은
회귀와 분류, 선형 모델, 다층 퍼셉트론,
과적합과 정규화 등 딥러닝에서 가장 기본이 되는
개념과 기법을 다룹니다.

* **2부: 현대 딥러닝 기법**.
:numref:`chap_computation`은
딥러닝 시스템의 핵심 계산 구성 요소를 설명하고,
이후에 더 복잡한 모델을 구현할 수 있는
토대를 마련합니다.
그다음 :numref:`chap_cnn`과 :numref:`chap_modern_cnn`은
대부분의 현대 컴퓨터 비전 시스템의 근간을 이루는 강력한 도구인
합성곱 신경망(CNN)을 소개합니다.
마찬가지로 :numref:`chap_rnn`과 :numref:`chap_modern_rnn`은
데이터의 순차적(예: 시간적) 구조를 활용하는 모델로
자연어 처리와 시계열 예측에 흔히 쓰이는
순환 신경망(RNN)을 소개합니다.
:numref:`chap_attention-and-transformers`에서는
이른바 *어텐션 메커니즘*에 기반한 상대적으로 새로운 부류의 모델을 다루는데,
이 모델은 대부분의 자연어 처리 과제에서
지배적인 아키텍처 자리를 RNN으로부터 넘겨받았습니다.
이 부분들은 딥러닝 사용자들이 폭넓게 활용하는
가장 강력하고 일반적인 도구들에
빠르게 익숙해질 수 있게 해 줍니다.

* **3부: 확장성, 효율성, 응용**([온라인](https://d2l.ai)에서 제공).
12장에서는
딥러닝 모델을 학습시키는 데 사용되는
몇 가지 흔한 최적화 알고리즘을 다룹니다.
그다음 13장에서는
딥러닝 코드의 계산 성능에
영향을 주는
몇 가지 핵심 요소를 살펴봅니다.
그리고 14장에서는
컴퓨터 비전에서의
딥러닝의 주요 응용 사례를 보여 줍니다.
마지막으로 15장과 16장에서는
언어 표현 모델을 사전 학습하고
자연어 처리 과제에 적용하는 방법을 보여 줍니다.


### 코드
:label:`sec_code`

이 책의 대부분의 절은 실행 가능한 코드를 포함합니다.
저희는 어떤 직관은 시행착오를 통해,
즉 코드를 조금씩 수정해 보고 결과를 관찰하면서
가장 잘 길러진다고 믿습니다.
이상적으로는 우아한 수학 이론이 원하는 결과를 얻기 위해
코드를 어떻게 수정해야 하는지를 정확히 알려줄 수도 있겠지만,
오늘날의 딥러닝 사용자들은 견고한 이론이 길잡이가 되지 못하는 곳을
자주 헤쳐 나가야 합니다.
저희의 최선의 시도에도 불구하고, 다양한 기법의 효과에 대한
형식적인 설명은 여전히 부족합니다.
그 이유는 여러 가지인데, 이러한 모델을 특징짓는 수학이 너무 어려울 수 있고,
그 설명은 아직 명확한 정의가 없는
데이터의 성질에 의존할 가능성이 높으며,
이러한 주제에 대한 본격적인 탐구가
최근에야 본격적으로 가속이 붙기 시작했기 때문입니다.
저희는 딥러닝 이론이 발전해 감에 따라
이 책의 후속 판들이 현재 가용한 것들을 능가하는
통찰을 제공해 주리라 기대합니다.

불필요한 반복을 피하기 위해, 저희는 가장 자주 임포트하고 사용하는
함수와 클래스 일부를 `d2l` 패키지에 담아 두었습니다.
책 전반에 걸쳐, 나중에 `d2l` 패키지를 통해 접근될 코드 블록
(함수, 클래스, 임포트 문 모음 등)에는
`#@save` 표시를 해 두어 이를 알립니다.
이러한 클래스와 함수에 대한 자세한 개요는
:numref:`sec_d2l`에서 제공합니다.
`d2l` 패키지는 가볍고 다음 의존성만 필요합니다.

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
이 책의 대부분의 코드는 Apache MXNet에 기반하고 있습니다.
MXNet은 오픈소스 딥러닝 프레임워크로,
AWS(Amazon Web Services)는 물론
많은 대학과 기업들이 선호하는 선택지입니다.
이 책의 모든 코드는 최신 MXNet 버전에서 테스트를 통과했습니다.
다만 딥러닝의 빠른 발전 속도 때문에
*인쇄본*의 일부 코드는
미래의 MXNet 버전에서는 제대로 동작하지 않을 수 있습니다.
저희는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참고해
코드와 런타임 환경을 업데이트해 주세요.
아래는 저희 MXNet 구현이 사용하는 의존성 목록입니다.
:end_tab:

:begin_tab:`pytorch`
이 책의 대부분의 코드는 PyTorch에 기반하고 있습니다.
PyTorch는 딥러닝 연구 커뮤니티가 열렬히 받아들인
인기 있는 오픈소스 프레임워크입니다.
이 책의 모든 코드는 PyTorch의 최신 안정 버전에서 테스트를 통과했습니다.
다만 딥러닝의 빠른 발전 속도 때문에
*인쇄본*의 일부 코드는
미래의 PyTorch 버전에서는 제대로 동작하지 않을 수 있습니다.
저희는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참고해
코드와 런타임 환경을 업데이트해 주세요.
아래는 저희 PyTorch 구현이 사용하는 의존성 목록입니다.
:end_tab:

:begin_tab:`tensorflow`
이 책의 대부분의 코드는 TensorFlow에 기반하고 있습니다.
TensorFlow는 산업계에서 폭넓게 채택되고
연구자들 사이에서도 인기 있는
오픈소스 딥러닝 프레임워크입니다.
이 책의 모든 코드는 TensorFlow의 최신 안정 버전에서 테스트를 통과했습니다.
다만 딥러닝의 빠른 발전 속도 때문에
*인쇄본*의 일부 코드는
미래의 TensorFlow 버전에서는 제대로 동작하지 않을 수 있습니다.
저희는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참고해
코드와 런타임 환경을 업데이트해 주세요.
아래는 저희 TensorFlow 구현이 사용하는 의존성 목록입니다.
:end_tab:

:begin_tab:`jax`
이 책의 대부분의 코드는 Jax에 기반하고 있습니다.
Jax는 임의의 Python·NumPy 함수의 미분과 같이
합성 가능한 함수 변환과
JIT 컴파일, 벡터화 등 다양한 기능을 가능케 하는 오픈소스 프레임워크입니다!
머신러닝 연구 분야에서 인기를 얻고 있으며,
익히기 쉬운 NumPy 유사 API를 제공합니다.
실제로 JAX는 NumPy와 1:1 호환을 목표로 하므로,
임포트 문 한 줄만 바꿔도 코드를 옮길 수 있을 정도입니다!
다만 딥러닝의 빠른 발전 속도 때문에
*인쇄본*의 일부 코드는
미래의 Jax 버전에서는 제대로 동작하지 않을 수 있습니다.
저희는 온라인 버전을 최신 상태로 유지할 계획입니다.
문제가 발생하면 :ref:`chap_installation`을 참고해
코드와 런타임 환경을 업데이트해 주세요.
아래는 저희 JAX 구현이 사용하는 의존성 목록입니다.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### 대상 독자

이 책은 딥러닝의 실용적인 기법을 탄탄히 다지고자 하는
학생(학부·대학원), 엔지니어, 연구자를 대상으로 합니다.
모든 개념을 밑바닥부터 설명하므로
딥러닝이나 머신러닝에 대한 사전 지식은 필요하지 않습니다.
딥러닝 기법을 온전히 설명하려면 어느 정도의 수학과 프로그래밍이 필요하지만,
저희는 여러분이 적당한 수준의 선형대수, 미적분, 확률,
그리고 파이썬 프로그래밍 같은 기초만 갖추고
들어왔다고 가정할 것입니다.
혹시 잊은 내용이 있을 경우를 대비해,
[온라인 부록](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html)에서
이 책에 등장하는 대부분의 수학에 대한
복습 자료를 제공합니다.
보통은 수학적 엄밀성보다
직관과 아이디어를 우선시할 것입니다.
만약 이러한 기초를 사전 지식 수준을 넘어
더 확장하고 싶다면,
다른 훌륭한 자료들을 기꺼이 추천합니다.
:citet:`Bollobas.1999`의 *Linear Analysis*는
선형대수와 함수해석을 매우 깊이 있게 다룹니다.
*All of Statistics* :cite:`Wasserman.2013`은
통계학에 대한 멋진 입문서입니다.
Joe Blitzstein의 확률과 추론에 관한 [책들](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918)과
[강의들](https://projects.iq.harvard.edu/stat110/home)은
교육적 가치가 보석 같은 자료들입니다.
그리고 파이썬을 사용해 본 적이 없다면,
이 [파이썬 튜토리얼](http://learnpython.org/)을 살펴보면 좋습니다.


### 노트북, 웹사이트, 깃허브, 포럼

저희의 모든 노트북은 [D2L.ai 웹사이트](https://d2l.ai)와
[깃허브](https://github.com/d2l-ai/d2l-en)에서 내려받을 수 있습니다.
이 책과 연계해 저희는 [discuss.d2l.ai](https://discuss.d2l.ai/c/5)에
토론 포럼을 개설했습니다.
책의 어떤 절에 대해서든 질문이 있다면,
각 노트북 끝에서 해당 토론 페이지로 가는 링크를
찾을 수 있습니다.



## 감사의 말

저희는 영어판과 중국어판 모두에 기여해 주신
수백 명의 기여자들에게 큰 빚을 지고 있습니다.
그분들은 내용을 개선하는 데 도움을 주고 귀중한 피드백을 보내 주셨습니다.
이 책은 처음에는 MXNet을 주요 프레임워크로 삼아 구현되었습니다.
초기 MXNet 코드의 상당 부분을 각각 PyTorch와 TensorFlow 구현으로 옮겨 주신 Anirudh Dagar와 Yuan Tang께 감사드립니다.
2021년 7월부터 저희는 이 책을 PyTorch, MXNet, TensorFlow로 다시 설계해 재구현했으며, PyTorch를 주요 프레임워크로 선택했습니다.
비교적 최근의 PyTorch 코드의 상당 부분을 JAX 구현으로 옮겨 주신 Anirudh Dagar께 감사드립니다.
중국어판에서 비교적 최근의 PyTorch 코드의 상당 부분을 PaddlePaddle 구현으로 옮겨 주신 Baidu의 Gaosheng Wu, Liujun Hu, Ge Zhang, Jiehang Xie께 감사드립니다.
출판사의 LaTeX 스타일을 PDF 빌드에 통합해 주신 Shuai Zhang께 감사드립니다.

깃허브에서 이 영어판 초안을 모두에게 더 좋게 만들어 주신
모든 기여자께 감사드립니다.
그들의 깃허브 ID 또는 이름은 다음과 같습니다(순서는 무관합니다):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

이 책의 집필에 아낌없는 지원을 보내 주신 Amazon Web Services,
특히 Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky,
Andrew Jassy께 감사드립니다.
시간과 자원, 동료들과의 토론, 끊임없는 격려가 없었다면
이 책은 세상에 나오지 못했을 것입니다.
출판 준비 과정에서
Cambridge University Press가 훌륭한 지원을 제공해 주었습니다.
도움과 전문성을 보여 주신 담당 편집자 David Tranah께
감사드립니다.


## 요약

딥러닝은 패턴 인식 분야에 혁명을 일으켰으며,
컴퓨터 비전, 자연어 처리,
자동 음성 인식 같은 다양한 분야에서
폭넓은 기술들을 뒷받침하는 기술을 도입했습니다.
딥러닝을 성공적으로 적용하려면,
문제를 정식화하는 법,
모델링의 기초 수학,
모델을 데이터에 맞추기 위한 알고리즘,
그리고 그 모든 것을 구현하기 위한 엔지니어링 기법을 이해해야 합니다.
이 책은 글, 그림, 수학, 코드를 한곳에 모은 종합 자료를 제공합니다.



## 연습 문제

1. 이 책의 토론 포럼 [discuss.d2l.ai](https://discuss.d2l.ai/)에 계정을 등록하세요.
1. 컴퓨터에 파이썬을 설치하세요.
1. 절 끝에 있는 포럼 링크를 따라가 보세요. 그곳에서 도움을 구하고 책에 대해 토론하며, 저자들과 더 넓은 커뮤니티와 교류하면서 질문에 대한 답을 찾을 수 있습니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17963)
:end_tab:
