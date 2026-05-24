# 풍부한 특징을 가진 추천 시스템

상호작용 데이터는 사용자의 선호도와 관심사를 나타내는 가장 기본적인 지표입니다. 이는 이전에 소개한 모델들에서 중요한 역할을 합니다. 그러나 상호작용 데이터는 보통 매우 희소하고 때로는 잡음이 많을 수 있습니다. 이 문제를 해결하기 위해, 저희는 아이템의 특징, 사용자 프로필, 심지어 상호작용이 발생한 컨텍스트와 같은 부가 정보를 추천 모델에 통합할 수 있습니다. 이러한 특징들을 활용하는 것은 특히 상호작용 데이터가 부족할 때 사용자 관심사의 효과적인 예측 변수가 될 수 있다는 점에서 추천을 만드는 데 도움이 됩니다. 따라서 추천 모델도 이러한 특징을 처리하고 모델에 일부 콘텐츠/컨텍스트 인식을 부여하는 능력을 갖는 것이 필수적입니다. 이러한 유형의 추천 모델을 설명하기 위해, 저희는 온라인 광고 추천을 위한 클릭률(CTR)에 대한 또 다른 작업을 소개하고 :cite:`McMahan.Holt.Sculley.ea.2013` 익명 광고 데이터셋을 제시합니다. 타겟 광고 서비스는 광범위한 관심을 끌었으며 종종 추천 엔진으로 구성됩니다. 사용자의 개인 취향과 관심사에 맞는 광고를 추천하는 것은 클릭률 향상에 중요합니다.


디지털 마케터들은 고객에게 광고를 표시하기 위해 온라인 광고를 사용합니다. 클릭률은 광고주가 광고의 노출 수당 받는 클릭 수를 측정하는 지표이며, 다음 공식으로 계산된 백분율로 표현됩니다.

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

클릭률은 예측 알고리즘의 효과성을 나타내는 중요한 신호입니다. 클릭률 예측은 웹사이트의 무언가가 클릭될 가능성을 예측하는 작업입니다. CTR 예측 모델은 타겟 광고 시스템뿐만 아니라 일반 아이템(예: 영화, 뉴스, 제품) 추천 시스템, 이메일 캠페인, 심지어 검색 엔진에도 활용될 수 있습니다. 또한 사용자 만족도, 전환율과 밀접하게 관련되어 있으며, 광고주가 현실적인 기대치를 설정하는 데 도움을 줄 수 있으므로 캠페인 목표 설정에도 도움이 될 수 있습니다.

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## 온라인 광고 데이터셋

인터넷과 모바일 기술의 상당한 발전과 함께, 온라인 광고는 중요한 수입원이 되었으며 인터넷 산업에서 대부분의 수익을 창출합니다. 일상적인 방문자가 유료 고객으로 전환될 수 있도록 관련 광고나 사용자의 관심을 끄는 광고를 표시하는 것이 중요합니다. 저희가 소개한 데이터셋은 온라인 광고 데이터셋입니다. 이는 34개의 필드로 구성되어 있으며, 첫 번째 열은 광고가 클릭되었는지(1) 아닌지(0)를 나타내는 대상 변수를 나타냅니다. 다른 모든 열은 범주형 특징입니다. 열은 광고 ID, 사이트 또는 애플리케이션 ID, 장치 ID, 시간, 사용자 프로필 등을 나타낼 수 있습니다. 특징의 실제 의미는 익명화와 개인정보 보호 문제로 인해 공개되지 않습니다.

다음 코드는 저희 서버에서 데이터셋을 다운로드하고 로컬 데이터 폴더에 저장합니다.

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

각각 15000개와 3000개의 샘플/라인으로 구성된 훈련 셋과 테스트 셋이 있습니다.

## 데이터셋 래퍼

데이터 로딩의 편의를 위해, 저희는 CSV 파일에서 광고 데이터셋을 로드하고 `DataLoader`에서 사용할 수 있는 `CTRDataset`를 구현합니다.

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

다음 예제는 훈련 데이터를 로드하고 첫 번째 레코드를 출력합니다.

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

보시는 바와 같이, 34개의 모든 필드는 범주형 특징입니다. 각 값은 해당 항목의 원-핫 인덱스를 나타냅니다. 레이블 $0$은 클릭되지 않았음을 의미합니다. 이 `CTRDataset`은 Criteo 디스플레이 광고 챌린지 [데이터셋](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)과 Avazu 클릭률 예측 [데이터셋](https://www.kaggle.com/c/avazu-ctr-prediction)과 같은 다른 데이터셋을 로드하는 데에도 사용될 수 있습니다.

## 요약
* 클릭률은 광고 시스템과 추천 시스템의 효과성을 측정하는 데 사용되는 중요한 지표입니다.
* 클릭률 예측은 보통 이진 분류 문제로 변환됩니다. 목표는 주어진 특징을 기반으로 광고/아이템이 클릭될지 여부를 예측하는 것입니다.

## 연습문제

* 제공된 `CTRDataset`로 Criteo와 Avazu 데이터셋을 로드할 수 있습니까? Criteo 데이터셋은 실수값 특징으로 구성되어 있다는 점에 유의해야 하므로, 코드를 약간 수정해야 할 수 있습니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/405)
:end_tab:
