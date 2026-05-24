#  MovieLens 데이터셋

추천 연구에 사용할 수 있는 데이터셋이 여러 가지 있습니다. 그중에서도 [MovieLens](https://movielens.org/) 데이터셋은 아마도 가장 인기 있는 것들 중 하나일 것입니다. MovieLens는 비상업적 웹 기반 영화 추천 시스템입니다. 1997년에 만들어져 미네소타 대학교의 연구소인 GroupLens에 의해 운영되며, 연구 목적으로 영화 평점 데이터를 수집하기 위해 만들어졌습니다. MovieLens 데이터는 개인화 추천과 사회 심리학을 포함한 여러 연구에 중요한 역할을 해 왔습니다.


## 데이터 가져오기


MovieLens 데이터셋은 [GroupLens](https://grouplens.org/datasets/movielens/) 웹사이트에서 호스팅됩니다. 여러 버전을 사용할 수 있습니다. 저희는 MovieLens 100K 데이터셋을 사용합니다 :cite:`Herlocker.Konstan.Borchers.ea.1999`. 이 데이터셋은 943명의 사용자가 1682편의 영화에 매긴 1점에서 5점까지의 $100,000$개의 평점으로 구성되어 있습니다. 각 사용자가 최소 20편의 영화에 평점을 매기도록 정리되었습니다. 사용자와 아이템에 대한 나이, 성별, 장르와 같은 간단한 인구통계학적 정보도 제공됩니다. 저희는 [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip)을 다운로드하고 csv 형식의 모든 $100,000$개의 평점을 포함하는 `u.data` 파일을 추출할 수 있습니다. 폴더에는 다른 많은 파일이 있으며, 각 파일에 대한 자세한 설명은 데이터셋의 [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) 파일에서 확인할 수 있습니다.

먼저 이 절의 실험을 실행하는 데 필요한 패키지를 임포트합시다.

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

그런 다음 MovieLens 100k 데이터셋을 다운로드하여 상호작용을 `DataFrame`으로 로드합니다.

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## 데이터셋 통계

데이터를 로드하고 처음 다섯 개의 레코드를 수동으로 살펴봅시다. 이는 데이터 구조를 학습하고 데이터가 올바르게 로드되었는지 확인하는 효과적인 방법입니다.

```{.python .input  n=3}
#@tab mxnet
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

각 줄이 "user id" 1-943, "item id" 1-1682, "rating" 1-5, "timestamp"의 네 열로 구성되어 있는 것을 볼 수 있습니다. 저희는 크기 $n \times m$의 상호작용 행렬을 구성할 수 있는데, 여기서 $n$과 $m$은 각각 사용자 수와 아이템 수입니다. 이 데이터셋은 기존 평점만 기록하므로 평점 행렬이라고도 부를 수 있으며, 이 행렬의 값이 정확한 평점을 나타내는 경우에는 상호작용 행렬과 평점 행렬을 같은 의미로 사용하겠습니다. 평점 행렬에 있는 대부분의 값은 사용자가 대부분의 영화에 평점을 매기지 않았기 때문에 알 수 없는 상태입니다. 저희는 이 데이터셋의 희소성도 보여드립니다. 희소성은 `1 - 0이 아닌 항목의 수 / (사용자 수 * 아이템 수)`로 정의됩니다. 분명히 상호작용 행렬은 매우 희소합니다(즉, 희소성 = 93.695%). 실세계 데이터셋은 더 큰 정도의 희소성을 겪을 수 있으며, 이는 추천 시스템 구축에서 오랫동안 지속되어 온 도전 과제입니다. 실행 가능한 해결책은 희소성을 완화하기 위해 사용자/아이템 특징과 같은 추가적인 부가 정보를 사용하는 것입니다.

그런 다음 다양한 평점 개수의 분포를 그립니다. 예상대로 대부분의 평점이 3-4에 집중된 정규분포처럼 보입니다.

```{.python .input  n=4}
#@tab mxnet
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## 데이터셋 분할

데이터셋을 훈련 셋과 테스트 셋으로 분할합니다. 다음 함수는 `random`과 `seq-aware`를 포함한 두 가지 분할 모드를 제공합니다. `random` 모드에서는 함수가 타임스탬프를 고려하지 않고 100k 상호작용을 무작위로 분할하고 기본적으로 데이터의 90%를 훈련 샘플로, 나머지 10%를 테스트 샘플로 사용합니다. `seq-aware` 모드에서는 사용자가 가장 최근에 평점을 매긴 아이템을 테스트용으로 남겨두고, 사용자의 과거 상호작용을 훈련 셋으로 사용합니다. 사용자의 과거 상호작용은 타임스탬프를 기준으로 가장 오래된 것부터 가장 최신 것까지 정렬됩니다. 이 모드는 시퀀스 인식 추천 절에서 사용될 것입니다.

```{.python .input  n=5}
#@tab mxnet
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

실제로는 테스트 셋만이 아닌 검증 셋을 사용하는 것이 좋은 관행이라는 점에 유의하십시오. 그러나 간결성을 위해 그 부분은 생략합니다. 이 경우 저희의 테스트 셋은 보류된 검증 셋으로 간주될 수 있습니다.

## 데이터 로딩

데이터셋 분할 후 저희는 편의를 위해 훈련 셋과 테스트 셋을 리스트와 딕셔너리/행렬로 변환할 것입니다. 다음 함수는 데이터프레임을 한 줄씩 읽고 사용자/아이템의 인덱스를 0부터 열거합니다. 그런 다음 함수는 사용자, 아이템, 평점의 리스트와 상호작용을 기록하는 딕셔너리/행렬을 반환합니다. 저희는 피드백 유형을 `explicit` 또는 `implicit`로 지정할 수 있습니다.

```{.python .input  n=6}
#@tab mxnet
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

이후 저희는 위의 단계들을 모아서 다음 절에서 사용할 것입니다. 결과는 `Dataset`과 `DataLoader`로 감싸집니다. 훈련 데이터를 위한 `DataLoader`의 `last_batch`는 `rollover` 모드로 설정되어 있으며(남은 샘플들은 다음 에포크로 이월됩니다) 순서가 셔플된다는 점에 유의하십시오.

```{.python .input  n=7}
#@tab mxnet
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## 요약

* MovieLens 데이터셋은 추천 연구에 널리 사용됩니다. 공개적으로 사용 가능하며 무료로 이용할 수 있습니다.
* 저희는 이후 절에서의 추가 사용을 위해 MovieLens 100k 데이터셋을 다운로드하고 전처리하는 함수를 정의합니다.


## 연습문제

* 다른 어떤 유사한 추천 데이터셋을 찾을 수 있습니까?
* MovieLens에 대한 더 많은 정보를 얻으려면 [https://movielens.org/](https://movielens.org/) 사이트를 살펴보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/399)
:end_tab:
