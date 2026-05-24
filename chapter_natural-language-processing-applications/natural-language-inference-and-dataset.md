# 자연어 추론과 데이터셋
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment`에서, 저희는 감성 분석 문제를 논의했습니다.
이 작업은 단일 텍스트 시퀀스를 감정 극성 집합과 같은
미리 정의된 범주로 분류하는 것을 목표로 합니다.
그러나 한 문장이 다른 문장으로부터 추론될 수 있는지 결정하거나,
의미적으로 동등한 문장을 식별하여 중복을 제거해야 할 필요가 있을 때,
하나의 텍스트 시퀀스를 분류하는 방법을 아는 것만으로는 충분하지 않습니다.
대신, 저희는 텍스트 시퀀스 쌍에 대해 추론할 수 있어야 합니다.


## 자연어 추론

*자연어 추론*은 *가설*이 *전제*로부터 추론될 수 있는지를 연구하며,
둘 다 텍스트 시퀀스입니다.
다시 말해, 자연어 추론은 텍스트 시퀀스 쌍 간의 논리적 관계를 결정합니다.
이러한 관계는 일반적으로 세 가지 유형으로 구분됩니다:

* *함의(Entailment)*: 가설이 전제로부터 추론될 수 있습니다.
* *모순(Contradiction)*: 가설의 부정이 전제로부터 추론될 수 있습니다.
* *중립(Neutral)*: 그 외의 모든 경우.

자연어 추론은 텍스트 함의 인식 작업으로도 알려져 있습니다.
예를 들어, 다음 쌍은 *함의*로 레이블링되는데, 이는 가설의 "애정을 표현"이 전제의 "서로 안고 있다"로부터 추론될 수 있기 때문입니다.

> 전제: 두 여성이 서로 안고 있다.

> 가설: 두 여성이 애정을 표현하고 있다.

다음은 *모순*의 예로, "코딩 예제를 실행하고 있다"는 "잠자고 있다"가 아니라 "잠자지 않고 있다"를 나타내기 때문입니다.

> 전제: 한 남자가 Dive into Deep Learning의 코딩 예제를 실행하고 있다.

> 가설: 그 남자는 잠자고 있다.

세 번째 예는 *중립* 관계를 보여주는데, 이는 "우리를 위해 공연하고 있다"는 사실로부터 "유명하다"도 "유명하지 않다"도 추론할 수 없기 때문입니다.

> 전제: 음악가들이 우리를 위해 공연하고 있다.

> 가설: 음악가들은 유명하다.

자연어 추론은 자연어 이해를 위한 중심 주제로 여겨져 왔습니다.
이는 정보 검색에서 오픈 도메인 질의응답에 이르기까지
폭넓게 활용되고 있습니다.
이 문제를 연구하기 위해, 인기 있는 자연어 추론 벤치마크 데이터셋을 조사하는 것부터 시작하겠습니다.


## 스탠퍼드 자연어 추론(SNLI) 데이터셋

[**스탠퍼드 자연어 추론(SNLI) 코퍼스**]는 500000개 이상의 레이블링된 영어 문장 쌍의 모음입니다 :cite:`Bowman.Angeli.Potts.ea.2015`.
저희는 압축이 풀린 SNLI 데이터셋을 다운로드하여 `../data/snli_1.0` 경로에 저장합니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**데이터셋 읽기**]

원본 SNLI 데이터셋은 저희가 실험에서 실제로 필요로 하는 것보다 훨씬 풍부한 정보를 담고 있습니다. 따라서 데이터셋의 일부만을 추출한 다음, 전제, 가설, 그리고 그 레이블의 리스트를 반환하는 `read_snli` 함수를 정의합니다.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

이제 전제와 가설의 [**처음 3쌍을 출력**]해 보겠습니다. 그 레이블("0", "1", "2"는 각각 "함의", "모순", "중립"에 해당)도 함께 출력합니다.

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

학습 세트는 약 550000쌍을 가지고 있고,
테스트 세트는 약 10000쌍을 가지고 있습니다.
다음은
학습 세트와 테스트 세트 모두에서
[**"함의", "모순", "중립" 세 가지 레이블이 균형을 이루고 있음**]을
보여줍니다.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**데이터셋 로딩을 위한 클래스 정의하기**]

아래에서 저희는 Gluon의 `Dataset` 클래스를 상속하여 SNLI 데이터셋을 로드하기 위한 클래스를 정의합니다. 클래스 생성자의 `num_steps` 인자는 텍스트 시퀀스의 길이를 지정하여 시퀀스의 각 미니배치가 같은 형상을 갖도록 합니다.
다시 말해,
긴 시퀀스에서 처음 `num_steps`개 이후의 토큰은 잘라내며, 짧은 시퀀스에는 그 길이가 `num_steps`가 될 때까지 특수 토큰 “&lt;pad&gt;”가 추가됩니다.
`__getitem__` 함수를 구현함으로써, 인덱스 `idx`로 전제, 가설, 레이블에 임의로 접근할 수 있습니다.

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**모두 통합하기**]

이제 SNLI 데이터셋을 다운로드하고 학습 및 테스트 세트 모두에 대한 `DataLoader` 인스턴스를 학습 세트의 어휘와 함께 반환하기 위해 `read_snli` 함수와 `SNLIDataset` 클래스를 호출할 수 있습니다.
주목할 점은 저희가 학습 세트로부터 구성된 어휘를 테스트 세트의 어휘로도 반드시 사용해야 한다는 것입니다.
결과적으로, 테스트 세트의 어떤 새로운 토큰도 학습 세트로 학습된 모델에게는 미지의 것이 됩니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

여기서 저희는 배치 크기를 128, 시퀀스 길이를 50으로 설정하고,
데이터 이터레이터와 어휘를 얻기 위해 `load_data_snli` 함수를 호출합니다.
그런 다음 어휘 크기를 출력합니다.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

이제 첫 번째 미니배치의 형상을 출력합니다.
감성 분석과는 달리,
전제와 가설의 쌍을 나타내는 두 입력 `X[0]`과 `X[1]`이 있습니다.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## 요약

* 자연어 추론은 가설이 전제로부터 추론될 수 있는지를 연구하며, 둘 다 텍스트 시퀀스입니다.
* 자연어 추론에서 전제와 가설 간의 관계는 함의, 모순, 중립을 포함합니다.
* 스탠퍼드 자연어 추론(SNLI) 코퍼스는 자연어 추론의 인기 있는 벤치마크 데이터셋입니다.


## 연습문제

1. 기계 번역은 오랫동안 출력 번역과 정답 번역 간의 표면적인 $n$-그램 일치에 기반하여 평가되어 왔습니다. 자연어 추론을 사용하여 기계 번역 결과를 평가하기 위한 측정 지표를 설계할 수 있겠습니까?
1. 어휘 크기를 줄이기 위해 하이퍼파라미터를 어떻게 변경할 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1388)
:end_tab:
