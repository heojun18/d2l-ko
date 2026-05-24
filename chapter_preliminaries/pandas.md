```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 데이터 전처리
:label:`sec_pandas`

지금까지 저희는 이미 만들어진 텐서 형태로 도착한
합성 데이터를 다뤄왔습니다.
그러나 실제 환경에서 딥러닝을 적용하려면
임의의 형식으로 저장된 지저분한 데이터를 추출해
저희의 필요에 맞도록 전처리해야 합니다.
다행스럽게도 *pandas* [라이브러리](https://pandas.pydata.org/)가
이러한 힘든 작업의 상당 부분을 해 줄 수 있습니다.
이 절은 제대로 된 *pandas* [튜토리얼](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)을
대체하지는 못하지만,
가장 흔히 사용되는 일부 루틴에 대한
속성 강좌를 제공해 드릴 것입니다.

## 데이터셋 읽어오기

쉼표로 구분된 값(CSV) 파일은 표 형식
(스프레드시트와 유사한) 데이터를 저장하는 데 널리 사용됩니다.
이 파일에서 각 줄은 하나의 레코드에 해당하며
여러 개의 (쉼표로 구분된) 필드로 구성됩니다. 예를 들어,
"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics"와 같습니다.
`pandas`로 CSV 파일을 로드하는 방법을 보여주기 위해,
저희는 (**아래에 CSV 파일을 생성**)합니다 `../data/house_tiny.csv`.
이 파일은 주택 데이터셋을 나타내며,
각 행은 서로 다른 주택에 해당하고
열은 방의 수(`NumRooms`), 지붕 유형(`RoofType`),
가격(`Price`)에 해당합니다.

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

이제 `pandas`를 임포트하고 `read_csv`를 사용해 데이터셋을 로드해 봅시다.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 데이터 준비

지도 학습에서, 저희는 어떤 *입력* 값의 집합이 주어졌을 때
지정된 *타깃* 값을 예측하도록 모델을 훈련시킵니다.
데이터셋을 처리하는 첫 단계는
입력 값에 해당하는 열과 타깃 값에 해당하는 열을
분리하는 것입니다.
열은 이름으로 선택하거나
정수 위치 기반 인덱싱(`iloc`)을 통해 선택할 수 있습니다.

`pandas`가 값이 `NA`인 모든 CSV 항목을
특수한 `NaN`(*not a number*) 값으로
대체했음을 눈치챘을 것입니다.
이는 항목이 비어 있을 때마다 발생할 수도 있습니다.
예를 들어, "3,,,270000"과 같은 경우입니다.
이들은 *결측치*라고 불리며
데이터 과학의 "빈대"와 같은 존재로,
여러분이 경력 내내 마주하게 될
끈질긴 골칫거리입니다.
상황에 따라, 결측치는
*대치* 또는 *삭제*를 통해 처리될 수 있습니다.
대치는 결측치를 그 값의 추정치로 대체하는 반면
삭제는 결측치가 포함된 행이나 열을
단순히 버립니다.

다음은 몇 가지 일반적인 대치 휴리스틱입니다.
[**범주형 입력 필드의 경우,
`NaN`을 하나의 범주로 처리할 수 있습니다.**]
`RoofType` 열은 `Slate`와 `NaN` 값을 가지므로,
`pandas`는 이 열을 `RoofType_Slate`와 `RoofType_nan`이라는
두 개의 열로 변환할 수 있습니다.
지붕 유형이 `Slate`인 행은 `RoofType_Slate`와
`RoofType_nan`의 값을 각각 1과 0으로 설정합니다.
`RoofType` 값이 결측된 행에 대해서는 그 반대가 적용됩니다.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

결측된 수치 값에 대한 일반적인 휴리스틱 하나는
[**`NaN` 항목을 해당 열의 평균값으로 대체하는 것**]입니다.

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## 텐서 형식으로의 변환

이제 [**`inputs`와 `targets`의 모든 항목이 수치이므로,
이를 텐서로 로드할 수 있습니다**] (:numref:`sec_ndarray`를 떠올려 보세요).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## 논의

이제 여러분은 데이터 열을 분할하고,
결측 변수를 대치하고,
`pandas` 데이터를 텐서로 로드하는 방법을 알게 되었습니다.
:numref:`sec_kaggle_house`에서 여러분은
몇 가지 추가적인 데이터 처리 기술을 익히게 될 것입니다.
이 속성 강좌에서는 내용을 단순하게 유지했지만,
데이터 처리는 까다로워질 수 있습니다.
예를 들어, 단일 CSV 파일로 도착하는 대신,
저희의 데이터셋은 관계형 데이터베이스에서 추출된
여러 파일에 흩어져 있을 수 있습니다.
예를 들어, 전자상거래 응용 분야에서
고객 주소는 한 테이블에 있고
구매 데이터는 다른 테이블에 있을 수 있습니다.
또한, 실무자들은 범주형과 수치형을 넘어
수많은 데이터 유형을 마주합니다. 예를 들어,
텍스트 문자열, 이미지,
오디오 데이터, 그리고 점 구름과 같은 것들입니다.
데이터 처리가 머신러닝 파이프라인에서 가장 큰 병목이 되지 않도록 하기 위해
종종 고급 도구와 효율적인 알고리즘이 필요합니다.
이러한 문제들은 컴퓨터 비전과 자연어 처리에 이르면
발생하게 될 것입니다.
마지막으로, 저희는 데이터 품질에 주의를 기울여야 합니다.
실제 데이터셋은 종종 이상치, 센서의 결함 측정값, 기록 오류로 인해
시달리며, 이러한 문제들은 데이터를 모델에
입력하기 전에 반드시 해결되어야 합니다.
[seaborn](https://seaborn.pydata.org/),
[Bokeh](https://docs.bokeh.org/), [matplotlib](https://matplotlib.org/) 같은
데이터 시각화 도구는 데이터를 수동으로 검사하고
여러분이 해결해야 할 문제 유형에 대한
직관을 키우는 데 도움을 줄 수 있습니다.


## 연습문제

1. 예를 들어 [UCI 머신러닝 저장소](https://archive.ics.uci.edu/ml/datasets)에서 Abalone 같은 데이터셋을 로드해 보고 그 속성을 검사해 보세요. 그중 몇 분의 몇이 결측치를 가지고 있나요? 변수 중 몇 분의 몇이 수치형, 범주형, 또는 텍스트인가요?
1. 열 번호가 아닌 이름으로 데이터 열을 인덱싱하고 선택해 보세요. pandas의 [인덱싱](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) 문서에 이를 수행하는 방법에 대한 더 자세한 내용이 있습니다.
1. 이 방식으로 얼마나 큰 데이터셋을 로드할 수 있다고 생각하시나요? 한계는 무엇일까요? 힌트: 데이터를 읽는 시간, 표현, 처리, 메모리 사용량을 고려해 보세요. 노트북에서 이를 시도해 보세요. 서버에서 시도하면 어떻게 되나요?
1. 매우 많은 수의 범주를 가진 데이터를 어떻게 다루겠습니까? 범주 레이블이 모두 고유하다면 어떨까요? 후자를 포함해야 할까요?
1. pandas에 대한 어떤 대안이 떠오르나요? [파일에서 NumPy 텐서를 로드하는 것](https://numpy.org/doc/stable/reference/generated/numpy.load.html)은 어떤가요? Python 이미징 라이브러리인 [Pillow](https://python-pillow.org/)를 살펴보세요.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17967)
:end_tab:
