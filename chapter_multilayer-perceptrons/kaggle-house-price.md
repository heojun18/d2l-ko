```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Kaggle에서 주택 가격 예측하기
:label:`sec_kaggle_house`

이제 저희가 심층 네트워크를 만들고 훈련하고
가중치 감쇠와 드롭아웃을 포함한 기법으로 정규화하는
기본 도구들을 소개했으므로,
이 모든 지식을 Kaggle 대회에 참여하여
실전에 적용할 준비가 되었습니다.
주택 가격 예측 대회는
시작하기에 좋은 곳입니다.
데이터는 상당히 일반적이며 ((오디오나 비디오처럼)) 특수한 모델이 필요할 만한
이국적인 구조를 보이지 않습니다.
:citet:`De-Cock.2011`이 수집한 이 데이터셋은,
2006년부터 2010년까지의 기간 동안 아이오와주 에임스의 주택 가격을 다룹니다.
이는 Harrison과 Rubinfeld (1978)의 유명한 [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)보다
상당히 더 크며,
더 많은 예제와 더 많은 특징을 자랑합니다.


이 절에서, 저희는 데이터 전처리, 모델 설계,
하이퍼파라미터 선택의 세부 사항을 안내해 드릴 것입니다.
실습 접근법을 통해, 여러분이
데이터 과학자로서의 경력에서 길잡이가 될
직관을 얻으시길 바랍니다.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## 데이터 다운로드

이 책 전반에 걸쳐, 저희는 다양한 다운로드된 데이터셋에서
모델을 훈련하고 테스트할 것입니다.
여기서 저희는 zip 또는 tar 파일을 다운로드하고
추출하기 위한 (**두 가지 유틸리티 함수를 구현합니다**).
다시 한번, 저희는 이러한 유틸리티 함수의 구현 세부 사항은
건너뜁니다.

```{.python .input  n=2}
%%tab all
def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""

def extract(filename, folder):
    """Extract a zip/tar file into folder."""
```

## Kaggle

[Kaggle](https://www.kaggle.com)은 머신러닝 대회를 주최하는
인기 있는 플랫폼입니다.
각 대회는 데이터셋을 중심으로 하며 많은 대회가
우승 솔루션에 상금을 제공하는 이해관계자들이 후원합니다.
이 플랫폼은 사용자들이 포럼과 공유 코드를 통해
상호작용할 수 있도록 돕고,
협력과 경쟁을 모두 촉진합니다.
리더보드 추격은 종종 통제 불능으로 치닫고,
연구자들이 근본적인 질문을 던지기보다는
근시안적으로 전처리 단계에 집중하기도 하지만,
경쟁 접근법들 사이의 직접적인 정량적 비교를 용이하게 하고
모두가 무엇이 효과가 있었고 없었는지 배울 수 있도록 코드 공유를 가능하게 하는
플랫폼의 객관성에는 엄청난 가치가 있기도 합니다.
Kaggle 대회에 참여하고 싶다면,
먼저 계정을 등록해야 합니다
(:numref:`fig_kaggle` 참조).

![Kaggle 웹사이트.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

:numref:`fig_house_pricing`에 나타난 것처럼
주택 가격 예측 대회 페이지에서,
여러분은 ("Data" 탭 아래에서) 데이터셋을 찾고,
예측을 제출하고, 순위를 볼 수 있습니다.
URL은 바로 여기입니다.

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![주택 가격 예측 대회 페이지.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## 데이터셋 접근 및 읽기

대회 데이터는
훈련 집합과 테스트 집합으로 분리되어 있다는 점에 유의하세요.
각 레코드는 주택의 재산 가치와
거리 유형, 건축 연도,
지붕 유형, 지하실 상태 등의 속성을 포함합니다.
특징은 다양한 데이터 유형으로 구성됩니다.
예를 들어, 건축 연도는
정수로 표현되고,
지붕 유형은 이산 범주형 할당으로,
다른 특징은 부동소수점 수로 표현됩니다.
그리고 여기서 현실이 일을 복잡하게 만듭니다.
일부 예제의 경우, 일부 데이터가 단순히 "na"로 표시된
누락 값으로 완전히 누락되어 있습니다.
각 주택의 가격은
훈련 집합에 대해서만 포함되어 있습니다
((어쨌든 대회니까요)).
저희는 검증 집합을 만들기 위해 훈련 집합을
분할하길 원하지만,
저희 모델을 공식 테스트 집합에서 평가하는 것은
예측을 Kaggle에 업로드한 후에만 가능합니다.
:numref:`fig_house_pricing`의 대회 탭에 있는 "Data" 탭에는
데이터를 다운로드하기 위한 링크가 있습니다.

시작하기 위해, 저희는 :numref:`sec_pandas`에서 소개한 `pandas`를 사용하여
[**데이터를 읽고 처리할 것입니다**].
편의를 위해, 저희는 Kaggle 주택 데이터셋을
다운로드하고 캐시할 수 있습니다.
이 데이터셋에 해당하는 파일이 이미 캐시 디렉터리에 존재하고 그 SHA-1이 `sha1_hash`와 일치한다면, 저희 코드는 중복 다운로드로 인터넷을 막히게 하지 않도록 캐시된 파일을 사용할 것입니다.

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

훈련 데이터셋은 1460개의 예제, 80개의 특징, 그리고 하나의 레이블을 포함하고,
검증 데이터는 1459개의 예제와 80개의 특징을
포함합니다.

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## 데이터 전처리

처음 네 개의 예제로부터 [**처음 네 개와 마지막 두 개의 특징,
그리고 레이블 (SalePrice)을 살펴봅시다**].

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

각 예제에서, 첫 번째 특징은 식별자라는 것을 볼 수 있습니다.
이는 모델이 각 훈련 예제를 결정하는 데 도움이 됩니다.
편리하긴 하지만, 예측 목적으로는
어떤 정보도 담고 있지 않습니다.
따라서, 저희는 데이터를 모델에 입력하기 전에
데이터셋에서 이를 제거할 것입니다.
또한, 다양한 데이터 유형이 주어진 상태에서,
저희는 모델링을 시작하기 전에 데이터를 전처리해야 합니다.


수치형 특징부터 시작합시다.
먼저, 저희는 휴리스틱을 적용하여,
[**모든 누락 값을 해당 특징의 평균으로
대체합니다.**]
그런 다음, 모든 특징을 공통 척도에 두기 위해, 저희는
(**특징을 0 평균과 단위 분산으로 다시 스케일링하여 데이터를 *표준화*합니다**).

$$x \leftarrow \frac{x - \mu}{\sigma},$$

여기서 $\mu$와 $\sigma$는 각각 평균과 표준 편차를 나타냅니다.
이것이 정말로 저희 특징 ((변수))을 0 평균과 단위 분산을 갖도록
변환한다는 것을 확인하기 위해,
$E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$이고
$E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$임에 주목하세요.
직관적으로, 저희는 두 가지 이유로
데이터를 표준화합니다.
첫째, 이는 최적화에 편리한 것으로 입증됩니다.
둘째, 어떤 특징이 관련성이 있을지
*사전에* 알지 못하기 때문에,
저희는 한 특징에 할당된 계수에
다른 어떤 것보다 더 페널티를 주고 싶지 않습니다.

[**다음으로 저희는 이산 값을 다룹니다.**]
이들은 "MSZoning"과 같은 특징을 포함합니다.
(**저희는 이전에 다중 클래스 레이블을 벡터로 변환했던 것
((:numref:`subsec_classification-problem` 참조))과 동일한 방식으로
이들을 원-핫 인코딩으로 대체합니다**).
예를 들어, "MSZoning"은 "RL"과 "RM" 값을 가정합니다.
"MSZoning" 특징을 드롭하고,
두 개의 새로운 지시 특징
"MSZoning_RL"과 "MSZoning_RM"이 0 또는 1의 값으로 생성됩니다.
원-핫 인코딩에 따르면,
"MSZoning"의 원래 값이 "RL"이라면,
"MSZoning_RL"은 1이고 "MSZoning_RM"은 0입니다.
`pandas` 패키지는 이를 저희를 위해 자동으로 해줍니다.

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes!='object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

이 변환이 특징 수를
79에서 331로 늘리는 것을 (((ID 및 레이블 열 제외)) 볼 수 있습니다.

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## 오차 척도

시작하기 위해 저희는 제곱 손실로 선형 모델을 훈련할 것입니다. 놀랍지 않게도, 저희의 선형 모델은 대회에서 우승할 제출물로 이어지지는 않을 것이지만, 데이터에 의미 있는 정보가 있는지 확인하는 정상성 검사를 제공합니다. 여기서 무작위 추측보다 더 잘하지 못한다면, 저희가 데이터 처리 버그를 가지고 있을 가능성이 높습니다. 그리고 잘 작동한다면, 선형 모델은 단순한 모델이 가장 잘 보고된 모델에 얼마나 가까이 가는지에 대한 약간의 직관을 제공하는 기준선 역할을 할 것이고, 더 화려한 모델로부터 저희가 얼마나 큰 이득을 기대해야 하는지에 대한 감각을 줄 것입니다.

주택 가격은, 주식 가격과 마찬가지로,
저희가 절대적인 양보다는
상대적인 양에 더 신경 씁니다.
따라서 [**저희는 절대 오차 $y - \hat{y}$보다는
상대 오차 $\frac{y - \hat{y}}{y}$에 더 신경 쓰는 경향이 있습니다**].
예를 들어, 일반적인 주택 가치가 \$125,000인
오하이오 시골의 주택 가격을 추정할 때
저희 예측이 \$100,000 빗나간다면,
저희는 아마 끔찍한 일을 하고 있는 것입니다.
반면에, 캘리포니아주 로스 알토스 힐스에서 이 정도로
빗나간다면, 이는 놀랍도록 정확한 예측을
나타낼 수도 있습니다
((그곳에서는 주택 가격 중앙값이 \$400만을 초과합니다)).

(**이 문제를 다루는 한 가지 방법은
가격 추정치의 로그에서 불일치를 측정하는 것입니다.**)
사실, 이는 또한 제출물의 품질을 평가하기 위해
대회에서 사용하는 공식 오차 척도이기도 합니다.
어쨌든, $|\log y - \log \hat{y}| \leq \delta$에 대한 작은 값 $\delta$는
$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$로 번역됩니다.
이는 예측 가격의 로그와 레이블 가격의 로그 사이의 다음과 같은 평균 제곱근 오차로 이어집니다.

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # Logarithm of prices 
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$-겹 교차 검증

저희가 모델 선택을 다루는 방법을 논의했던 :numref:`subsec_generalization-model-selection`에서
[**교차 검증**]을 소개했다는 것을
기억하실 것입니다.
저희는 모델 설계를 선택하고
하이퍼파라미터를 조정하기 위해 이를 잘 활용할 것입니다.
저희는 먼저 $K$-겹 교차 검증 절차에서 데이터의
$i^\textrm{th}$ 겹을 반환하는 함수가 필요합니다.
이는 $i^\textrm{th}$ 세그먼트를 검증 데이터로 잘라내고
나머지를 훈련 데이터로 반환하여 진행됩니다.
이는 데이터를 다루는 가장 효율적인 방법은 아니며
데이터셋이 상당히 더 크다면 분명히 훨씬 더 영리한 방법을
사용할 것이라는 점에 유의하세요.
하지만 이러한 추가 복잡성은 코드를 불필요하게 모호하게 만들 수 있으므로
문제의 단순성 덕분에 여기서는 안전하게 생략할 수 있습니다.

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

$K$-겹 교차 검증에서 $K$번 훈련할 때 [**평균 검증 오차가 반환됩니다**].

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**모델 선택**]

이 예제에서, 저희는 조정되지 않은 하이퍼파라미터 집합을 선택하고
모델을 개선하는 것은 독자에게 맡깁니다.
좋은 선택을 찾는 것은 얼마나 많은 변수에 대해 최적화하는지에 따라
시간이 걸릴 수 있습니다.
충분히 큰 데이터셋과,
일반적인 종류의 하이퍼파라미터로,
$K$-겹 교차 검증은
다중 테스트에 합리적으로 견고한 경향이 있습니다.
하지만 비합리적으로 많은 수의 옵션을 시도하면
저희는 검증 성능이
더 이상 실제 오차를 대표하지 않게 될 수도 있다는 것을 발견할 수도 있습니다.

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

때로는 한 하이퍼파라미터 집합에 대한 훈련 오차의 수가
매우 낮을 수 있는데, $K$-겹 교차 검증에서의 오차 수가
훨씬 더 높아지는 경우에도 그렇다는 점에 유의하세요.
이는 저희가 과적합되고 있음을 나타냅니다.
훈련 내내 여러분은 두 숫자를 모두 모니터링해야 할 것입니다.
과적합이 적다는 것은 저희 데이터가 더 강력한 모델을 지원할 수 있음을 나타낼 수도 있습니다.
대규모 과적합은 정규화 기법을 도입함으로써
이득을 얻을 수 있음을 시사할 수도 있습니다.

##  [**Kaggle에 예측 제출**]

이제 좋은 하이퍼파라미터 선택이 무엇이어야 하는지 알았으므로,
저희는 모든 $K$개의 모델로
테스트 집합에 대한 평균 예측을
계산할 수도 있습니다.
예측을 csv 파일로 저장하면
결과를 Kaggle에 업로드하는 것이 단순해질 것입니다.
다음 코드는 `submission.csv`라는 파일을 생성할 것입니다.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
             d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

다음으로, :numref:`fig_kaggle_submit2`에 나타난 것처럼,
저희는 예측을 Kaggle에 제출하고
테스트 집합의 실제 주택 가격 ((레이블))과
어떻게 비교되는지 볼 수 있습니다.
단계는 매우 간단합니다.

* Kaggle 웹사이트에 로그인하고 주택 가격 예측 대회 페이지를 방문합니다.
* "Submit Predictions" 또는 "Late Submission" 버튼을 클릭합니다.
* 페이지 하단의 점선 상자에 있는 "Upload Submission File" 버튼을 클릭하고 업로드하려는 예측 파일을 선택합니다.
* 결과를 보려면 페이지 하단의 "Make Submission" 버튼을 클릭합니다.

![Kaggle에 데이터 제출하기.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## 요약 및 논의

실제 데이터는 종종 서로 다른 데이터 유형의 혼합을 포함하며 전처리되어야 합니다.
실숫값 데이터를 0 평균과 단위 분산으로 다시 스케일링하는 것이 좋은 기본값입니다. 누락 값을 평균으로 대체하는 것도 그렇습니다.
또한, 범주형 특징을 지시 특징으로 변환하면 이를 원-핫 벡터처럼 다룰 수 있습니다.
저희가 절대 오차보다 상대 오차에 더 신경 쓰는 경향이 있을 때,
저희는 예측의 로그에서
불일치를 측정할 수 있습니다.
모델을 선택하고 하이퍼파라미터를 조정하기 위해,
저희는 $K$-겹 교차 검증을 사용할 수 있습니다.



## 연습문제

1. 이 절의 예측을 Kaggle에 제출하세요. 그 결과는 얼마나 좋은가요?
1. 누락 값을 평균으로 대체하는 것이 항상 좋은 생각인가요? 힌트: 값이 무작위로 누락되지 않는 상황을 구성할 수 있나요?
1. $K$-겹 교차 검증을 통해 하이퍼파라미터를 조정하여 점수를 개선하세요.
1. 모델 ((예: 층, 가중치 감쇠, 드롭아웃))을 개선하여 점수를 개선하세요.
1. 이 절에서 한 것처럼 연속형 수치 특징을 표준화하지 않으면 어떤 일이 일어나나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17988)
:end_tab:
