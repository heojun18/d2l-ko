# 단어 임베딩 사전 학습을 위한 데이터셋
:label:`sec_word2vec_data`

이제 저희는 word2vec 모델과 근사 학습 방법의 기술적 세부 사항을 알고 있으므로,
구현을 살펴봅시다.
구체적으로,
:numref:`sec_word2vec`의 스킵그램 모델과
:numref:`sec_approx_train`의 네거티브 샘플링을
예로 들겠습니다.
이 절에서는,
단어 임베딩 모델을 사전 학습하기 위한 데이터셋부터 시작합니다.
데이터의 원본 형식을
학습 중에 반복 처리할 수 있는
미니배치로 변환할 것입니다.

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## 데이터셋 읽기

저희가 여기서 사용하는 데이터셋은
[Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42)입니다.
이 말뭉치는 월스트리트 저널 기사에서 샘플링되었으며,
학습, 검증, 테스트 세트로 분할되어 있습니다.
원본 형식에서,
텍스트 파일의 각 줄은
공백으로 구분된 단어들로 이루어진 한 문장을 나타냅니다.
여기서 저희는 각 단어를 토큰으로 다룹니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

학습 세트를 읽은 후,
저희는 말뭉치에 대한 어휘를 만듭니다.
여기서 10번 미만 등장하는 단어는
"&lt;unk&gt;" 토큰으로 대체됩니다.
원본 데이터셋에도
드문(알 수 없는) 단어를 나타내는 "&lt;unk&gt;" 토큰이
포함되어 있음에 유의하십시오.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## 서브샘플링

텍스트 데이터는
일반적으로 "the", "a", "in" 같은
고빈도 단어를 가지고 있습니다.
이들은 매우 큰 말뭉치에서는 수십억 번 등장할 수도 있습니다.
그러나
이러한 단어들은 문맥 윈도우에서
서로 다른 많은 단어들과 함께 등장하는 경우가 많아,
유용한 신호를 거의 제공하지 않습니다.
예를 들어,
문맥 윈도우에서 "chip"이라는 단어를 생각해 보십시오.
직관적으로
저빈도 단어 "intel"과 함께 등장하는 것이
고빈도 단어 "a"와 함께 등장하는 것보다
학습에 더 유용합니다.
게다가, 방대한 양의 (고빈도) 단어로 학습하는 것은
느립니다.
따라서, 단어 임베딩 모델을 학습할 때
고빈도 단어를 *서브샘플링(subsample)* 할 수 있습니다 :cite:`Mikolov.Sutskever.Chen.ea.2013`.
구체적으로,
데이터셋에서 인덱스를 가진 각 단어 $w_i$는
다음의 확률로 버려집니다.


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

여기서 $f(w_i)$는
데이터셋의 전체 단어 수에 대한
$w_i$의 개수 비율이고,
상수 $t$는 하이퍼파라미터입니다
(실험에서는 $10^{-4}$).
상대 빈도
$f(w_i) > t$일 때만
(고빈도) 단어 $w_i$가 버려질 수 있으며,
단어의 상대 빈도가 높을수록
버려질 확률이 더 크다는 것을 알 수 있습니다.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

다음 코드 조각은
서브샘플링 전과 후의
문장당 토큰 수의 히스토그램을
그립니다.
예상대로,
서브샘플링은 고빈도 단어를 버림으로써
문장을 크게 짧게 만들며,
이는 학습 속도 향상으로 이어집니다.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

개별 토큰의 경우, 고빈도 단어 "the"의 샘플링 비율은 1/20보다 작습니다.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

반면,
저빈도 단어 "join"은 완전히 유지됩니다.

```{.python .input}
#@tab all
compare_counts('join')
```

서브샘플링 후, 저희는 말뭉치에 대해 토큰을 그 인덱스에 매핑합니다.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 중심 단어와 문맥 단어 추출


다음의 `get_centers_and_contexts`
함수는 `corpus`에서
모든 중심 단어와 그 문맥 단어를
추출합니다.
1과 `max_window_size` 사이의 정수를
문맥 윈도우 크기로 무작위 균등 샘플링합니다.
임의의 중심 단어에 대해,
샘플링된 문맥 윈도우 크기를
초과하지 않는 거리의 단어들이
그것의 문맥 단어입니다.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

다음으로, 각각 7개와 3개 단어로 이루어진 두 문장을 포함하는 인공 데이터셋을 만듭니다.
최대 문맥 윈도우 크기를 2로 두고
모든 중심 단어와 그 문맥 단어를 출력합시다.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

PTB 데이터셋으로 학습할 때,
저희는 최대 문맥 윈도우 크기를 5로 설정합니다.
다음은 데이터셋의 모든 중심 단어와 그 문맥 단어를 추출합니다.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## 네거티브 샘플링

근사 학습을 위해 저희는 네거티브 샘플링을 사용합니다.
미리 정의된 분포에 따라
노이즈 단어를 샘플링하기 위해,
다음의 `RandomGenerator` 클래스를 정의합니다.
여기서 (정규화되지 않을 수 있는) 샘플링 분포는
`sampling_weights` 인수를 통해 전달됩니다.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

예를 들어,
샘플링 확률 $P(X=1)=2/9, P(X=2)=3/9$, $P(X=3)=4/9$로
인덱스 1, 2, 3 중에서
10개의 확률 변수 $X$를 다음과 같이 추출할 수 있습니다.

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

한 쌍의 중심 단어와 문맥 단어에 대해,
저희는 `K`개(실험에서는 5)의 노이즈 단어를 무작위로 샘플링합니다. word2vec 논문의 제안에 따라,
노이즈 단어 $w$의
샘플링 확률 $P(w)$는
사전에서의 상대 빈도를
0.75 제곱한 값으로 설정됩니다 :cite:`Mikolov.Sutskever.Chen.ea.2013`.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## 미니배치로 학습 예제 불러오기
:label:`subsec_word2vec-minibatch-loading`

모든 중심 단어와 그 문맥 단어, 그리고 샘플링된 노이즈 단어가 추출된 후,
이들은
학습 중에 반복적으로 불러올 수 있는
예제의 미니배치로
변환될 것입니다.



미니배치에서,
$i^\textrm{번째}$ 예제는 하나의 중심 단어와
그것의 $n_i$개 문맥 단어 및 $m_i$개 노이즈 단어를 포함합니다.
문맥 윈도우 크기가 변화하므로,
$n_i+m_i$는 서로 다른 $i$에 대해 달라집니다.
따라서,
각 예제에 대해
저희는 문맥 단어와 노이즈 단어를
`contexts_negatives` 변수에 연결하고,
연결 길이가 $\max_i n_i+m_i$ (`max_len`)에 도달할 때까지
0을 패딩합니다.
손실 계산에서 패딩을 제외하기 위해,
저희는 마스크 변수 `masks`를 정의합니다.
`masks`의 원소와 `contexts_negatives`의 원소 사이에는
일대일 대응이 있으며,
`masks`의 0(나머지는 1)은 `contexts_negatives`의 패딩에 대응합니다.


양성 예제와 음성 예제를 구분하기 위해,
저희는 `labels` 변수를 통해 `contexts_negatives`에서 문맥 단어를 노이즈 단어와 분리합니다.
`masks`와 유사하게,
`labels`의 원소와 `contexts_negatives`의 원소 사이에도
일대일 대응이 있으며,
`labels`의 1(나머지는 0)은 `contexts_negatives`의 문맥 단어(양성 예제)에 대응합니다.


위의 아이디어는 다음의 `batchify` 함수에서 구현됩니다.
입력 `data`는 길이가
배치 크기와 같은 리스트이며,
각 원소는
중심 단어 `center`, 그것의 문맥 단어 `context`, 그리고 그것의 노이즈 단어 `negative`로
구성된 예제입니다.
이 함수는 학습 중에 계산을 위해 불러올 수 있는
미니배치를 반환합니다.
예를 들어 마스크 변수를 포함합니다.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

두 예제로 이루어진 미니배치를 사용해 이 함수를 테스트해 봅시다.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## 모두 합치기

마지막으로, PTB 데이터셋을 읽고 데이터 반복자와 어휘를 반환하는 `load_data_ptb` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

데이터 반복자의 첫 번째 미니배치를 출력해 봅시다.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## 요약

* 고빈도 단어는 학습에 그다지 유용하지 않을 수 있습니다. 학습 속도 향상을 위해 이들을 서브샘플링할 수 있습니다.
* 계산 효율을 위해, 저희는 예제를 미니배치로 불러옵니다. 패딩과 비패딩, 양성 예제와 음성 예제를 구분하기 위해 다른 변수를 정의할 수 있습니다.



## 연습문제

1. 서브샘플링을 사용하지 않으면 이 절의 코드의 실행 시간은 어떻게 변합니까?
1. `RandomGenerator` 클래스는 `k`개의 무작위 샘플링 결과를 캐시합니다. `k`를 다른 값으로 설정하여 데이터 로딩 속도에 어떻게 영향을 미치는지 보십시오.
1. 이 절의 코드에서 데이터 로딩 속도에 영향을 미칠 수 있는 다른 하이퍼파라미터는 무엇입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:
