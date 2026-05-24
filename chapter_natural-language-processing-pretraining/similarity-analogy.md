# 단어 유사도와 유추
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining`에서,
저희는 작은 데이터셋에서 word2vec 모델을 학습하고,
입력 단어에 대해 의미적으로 유사한 단어를
찾기 위해 이를 적용했습니다.
실제로는,
큰 말뭉치에서 사전 학습된 단어 벡터는
다운스트림 자연어 처리 작업에
적용될 수 있는데,
이는 추후 :numref:`chap_nlp_app`에서 다룹니다.
큰 말뭉치에서 사전 학습된
단어 벡터의 의미를 직관적으로
보여주기 위해,
단어 유사도와 유추 작업에 이들을
적용해 봅시다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## 사전 학습된 단어 벡터 불러오기

아래는 [GloVe 웹사이트](https://nlp.stanford.edu/projects/glove/)에서 다운로드할 수 있는 차원 50, 100, 300의 사전 학습된 GloVe 임베딩을 나열합니다.
사전 학습된 fastText 임베딩은 여러 언어로 제공됩니다.
여기서 저희는 [fastText 웹사이트](https://fasttext.cc/)에서 다운로드할 수 있는
한 영어 버전(300차원 "wiki.en")을 고려합니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

이러한 사전 학습된 GloVe 및 fastText 임베딩을 불러오기 위해, 다음의 `TokenEmbedding` 클래스를 정의합니다.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

아래에서 저희는 (위키피디아 부분 집합에서 사전 학습된)
50차원 GloVe 임베딩을
불러옵니다.
`TokenEmbedding` 인스턴스를 생성할 때,
지정된 임베딩 파일이
아직 다운로드되지 않았다면 다운로드되어야 합니다.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

어휘 크기를 출력합니다. 어휘는 400000개의 단어(토큰)와 하나의 특수 알 수 없음 토큰을 포함합니다.

```{.python .input}
#@tab all
len(glove_6b50d)
```

저희는 어휘 내 단어의 인덱스를 얻을 수 있고, 그 반대도 가능합니다.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 사전 학습된 단어 벡터 적용

불러온 GloVe 벡터를 사용해,
저희는 다음의 단어 유사도와 유추 작업에 적용함으로써
그 의미를 보여줄 것입니다.


### 단어 유사도

:numref:`subsec_apply-word-embed`와 유사하게,
단어 벡터 사이의 코사인 유사도에 기반하여
입력 단어에 대한 의미적으로 유사한 단어를
찾기 위해,
저희는 다음의 `knn`
($k$-최근접 이웃) 함수를 구현합니다.

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

그런 다음, 저희는
`TokenEmbedding` 인스턴스 `embed`의
사전 학습된 단어 벡터를 사용해
유사한 단어를 검색합니다.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d`의 사전 학습된 단어 벡터의 어휘는
400000개의 단어와 하나의 특수 알 수 없음 토큰을 포함합니다.
입력 단어와 알 수 없음 토큰을 제외하면,
이 어휘에서
"chip"이라는 단어와
의미적으로 가장 유사한
세 단어를 찾아봅시다.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

아래는 "baby"와 "beautiful"과
유사한 단어를 출력합니다.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 단어 유추

유사한 단어를 찾는 것 외에도,
저희는 단어 유추 작업에도
단어 벡터를 적용할 수 있습니다.
예를 들어,
"man":"woman"::"son":"daughter"는
단어 유추의 형태입니다.
"man"과 "woman"의 관계는 "son"과 "daughter"의 관계와 같습니다.
구체적으로,
단어 유추 완성 작업은 다음과 같이 정의될 수 있습니다.
단어 유추 $a : b :: c : d$에 대해, 처음 세 단어 $a$, $b$, $c$가 주어졌을 때, $d$를 찾습니다.
단어 $w$의 벡터를 $\textrm{vec}(w)$로 표시합니다.
유추를 완성하기 위해,
$\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$의 결과와
벡터가 가장 유사한 단어를
찾을 것입니다.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

불러온 단어 벡터를 사용해 "남성-여성" 유추를 검증해 봅시다.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

아래는
"수도-국가" 유추를 완성합니다.
"beijing":"china"::"tokyo":"japan".
이는 사전 학습된 단어 벡터의
의미를 보여줍니다.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

"bad":"worst"::"big":"biggest"와 같은
"형용사-최상급 형용사" 유추의 경우,
저희는 사전 학습된 단어 벡터가
구문 정보를 포착할 수 있음을 알 수 있습니다.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

사전 학습된 단어 벡터에서 포착된
과거 시제의 개념을 보여주기 위해,
저희는 "현재 시제-과거 시제" 유추를 사용해 구문을 테스트할 수 있습니다. "do":"did"::"go":"went".

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## 요약

* 실제로는, 큰 말뭉치에서 사전 학습된 단어 벡터는 다운스트림 자연어 처리 작업에 적용될 수 있습니다.
* 사전 학습된 단어 벡터는 단어 유사도와 유추 작업에 적용될 수 있습니다.


## 연습문제

1. `TokenEmbedding('wiki.en')`을 사용해 fastText 결과를 테스트하십시오.
1. 어휘가 매우 클 때, 유사한 단어를 찾거나 단어 유추를 완성하는 것을 어떻게 더 빨리 할 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
