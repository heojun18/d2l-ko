# 원시 텍스트를 시퀀스 데이터로 변환하기
:label:`sec_text-sequence`

이 책 전반에 걸쳐,
저희는 단어, 문자, 또는 워드피스의 시퀀스로 표현된
텍스트 데이터를 가지고 자주 작업할 것입니다.
시작하려면 원시 텍스트를 적절한 형태의 시퀀스로
변환하기 위한 몇 가지 기본 도구가 필요합니다.
일반적인 전처리 파이프라인은 다음 단계들을 실행합니다.

1. 텍스트를 문자열로 메모리에 로드합니다.
1. 문자열을 토큰(예: 단어 또는 문자)으로 분할합니다.
1. 각 어휘 요소를 수치 인덱스와 연결시키는 어휘 사전을 만듭니다.
1. 텍스트를 수치 인덱스의 시퀀스로 변환합니다.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## 데이터셋 읽기

여기서 저희는 H. G. 웰스(Wells)의
[The Time Machine](http://www.gutenberg.org/ebooks/35),
약 30,000개가 조금 넘는 단어를 포함하는 책을 가지고 작업할 것입니다.
실제 응용에서는 일반적으로
훨씬 더 큰 데이터셋이 관여하지만,
이것은 전처리 파이프라인을 시연하기에 충분합니다.
다음 `_download` 메서드는
(**원시 텍스트를 문자열로 읽습니다**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

간단함을 위해, 저희는 원시 텍스트를 전처리할 때 구두점과 대소문자를 무시합니다.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## 토큰화

*토큰(Tokens)* 은 텍스트의 원자(분할 불가능한) 단위입니다.
각 타임스텝은 1개의 토큰에 대응되지만,
무엇이 정확히 토큰을 구성하는지는 설계상의 선택입니다.
예를 들어, 저희는 문장
"Baby needs a new pair of shoes"를
7개 단어의 시퀀스로 표현할 수 있는데,
여기서 모든 단어의 집합은 큰 어휘(일반적으로 수만 개 또는
수십만 개의 단어)를 이룹니다.
또는 저희는 같은 문장을 훨씬 더 작은 어휘
(고유한 ASCII 문자는 256개밖에 없습니다)를 사용하여
30개 문자의 훨씬 더 긴 시퀀스로 표현할 수도 있습니다.
아래에서 저희는 전처리된 텍스트를
문자의 시퀀스로 토큰화합니다.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## 어휘 (Vocabulary)

이 토큰들은 여전히 문자열입니다.
그러나 저희 모델로의 입력은
궁극적으로 수치 입력으로 구성되어야 합니다.
[**다음으로, 저희는 *어휘(vocabularies)*,
즉 각 고유한 토큰 값을
고유한 인덱스와 연결시키는 객체를
구성하기 위한 클래스를 소개합니다.**]
먼저, 저희는 학습 *코퍼스(corpus)* 에서 고유한 토큰의 집합을 결정합니다.
그 다음 각 고유한 토큰에 수치 인덱스를 할당합니다.
드문 어휘 요소는 편의를 위해 종종 버려집니다.
학습 또는 테스트 시점에 이전에 보지 못했거나
어휘에서 버려진 토큰을 마주칠 때마다,
저희는 그것을 특수한 "&lt;unk&gt;" 토큰으로 표현하여,
이것이 *알 수 없는(unknown)* 값임을 나타냅니다.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

이제 저희는 데이터셋에 대한 [**어휘를 구성**]하여,
문자열의 시퀀스를 수치 인덱스의 리스트로 변환합니다.
저희가 어떤 정보도 잃지 않았으며
저희 데이터셋을 원래의 (문자열) 표현으로
쉽게 되돌릴 수 있음에 유의하세요.

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## 모두 합치기

위의 클래스와 메서드를 사용하여,
저희는 [**모든 것을 다음의 `TimeMachine` 클래스의
`build` 메서드로 패키징**]하는데,
이 메서드는 토큰 인덱스의 리스트인 `corpus`와
*The Time Machine* 코퍼스의 어휘인 `vocab`을 반환합니다.
저희가 여기서 한 수정 사항은 다음과 같습니다.
(i) 이후 절에서의 학습을 단순화하기 위해,
텍스트를 단어가 아니라 문자로 토큰화합니다.
(ii) `corpus`는 토큰 리스트들의 리스트가 아니라 단일 리스트인데,
이는 *The Time Machine* 데이터셋의 각 텍스트 줄이
반드시 문장이나 단락이 아니기 때문입니다.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## 탐색적 언어 통계
:label:`subsec_natural-lang-stat`

실제 코퍼스와 단어에 대해 정의된 `Vocab` 클래스를 사용하여,
저희는 코퍼스 내 단어 사용에 관한 기본 통계를 살펴볼 수 있습니다.
아래에서 저희는 *The Time Machine*에서 사용된 단어로부터
어휘를 구성하고 가장 자주 발생하는 단어 10개를 출력합니다.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

(**가장 빈도가 높은 10개 단어**)가
그렇게 서술적이지는 않다는 점에 유의하세요.
저희가 어떤 책을 무작위로 골랐더라도
매우 유사한 리스트를 봤을 것이라고
상상하실 수도 있을 것입니다.
"the"와 "a" 같은 관사,
"i"와 "my" 같은 대명사,
"of", "to", "in" 같은 전치사는
흔한 구문적 역할을 수행하기 때문에 자주 발생합니다.
흔하지만 특별히 서술적이지는 않은 그러한 단어를
종종 (***불용어(stop words)***)라고 부르고,
이른바 단어 가방(bag-of-words) 표현을 기반으로 한
이전 세대의 텍스트 분류기에서는,
그것들이 가장 자주 걸러졌습니다.
그러나 그것들은 의미를 담고 있으며,
현대의 RNN 및 Transformer 기반 신경 모델로 작업할 때는
그것들을 걸러낼 필요가 없습니다.
리스트를 더 아래로 내려다보면,
단어 빈도가 빠르게 감쇠한다는 점을
알아채실 것입니다.
$10^{\textrm{th}}$번째로 빈도가 높은 단어는
가장 인기 있는 단어의 $1/5$도 안 됩니다.
순위가 내려갈수록 단어 빈도는
거듭제곱 법칙 분포(구체적으로는 지프 분포(Zipfian))를 따르는 경향이 있습니다.
더 잘 이해하기 위해, 저희는 [**단어 빈도의 도표를 그립니다**].

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

처음 몇 단어를 예외로 다룬 후,
나머지 모든 단어는 로그-로그 도표에서 대체로 직선을 따릅니다.
이 현상은 *지프의 법칙(Zipf's law)* 에 의해 포착되는데,
이는 $i^\textrm{th}$번째로 빈도가 높은 단어의 빈도 $n_i$가:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

라고 하며, 이는 다음과 동등합니다.

$$\log n_i = -\alpha \log i + c,$$

여기서 $\alpha$는 분포를 특징짓는 지수이고
$c$는 상수입니다.
이것은 저희가 통계를 세는 방식으로 단어를 모델링하고자 한다면
이미 멈춰 생각해 봐야 할 점을 시사합니다.
결국 저희는 꼬리, 즉 빈도가 낮은 단어의 빈도를 상당히 과대평가할 것입니다. 그러나 [**다른 단어 조합들, 예를 들어 두 개의 연속된 단어(바이그램), 세 개의 연속된 단어(트라이그램)**], 그리고 그 이상은 어떨까요?
바이그램 빈도가 단일 단어(유니그램) 빈도와 같은 방식으로 동작하는지 봅시다.

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

여기서 한 가지가 주목할 만합니다. 가장 빈도가 높은 10개의 단어 쌍 중, 9개는 모두 불용어로 구성되어 있고 단 하나만이 실제 책과 관련이 있습니다("the time"). 더 나아가, 트라이그램 빈도가 같은 방식으로 동작하는지 봅시다.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

이제, 이 세 가지 모델(유니그램, 바이그램, 트라이그램) 사이에서 [**토큰 빈도를 시각화**]해 봅시다.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

이 그림은 꽤 흥미진진합니다.
첫째, 유니그램 단어를 넘어, 단어 시퀀스도
시퀀스 길이에 따라
:eqref:`eq_zipf_law`에서 더 작은 지수 $\alpha$로,
지프의 법칙을 따르는 것으로 보입니다.
둘째, 고유한 $n$-그램의 개수는 그렇게 크지 않습니다.
이는 언어에 꽤 많은 구조가 있다는 희망을 줍니다.
셋째, 많은 $n$-그램이 매우 드물게 발생합니다.
이는 특정 방법을 언어 모델링에 부적합하게 만들고
딥러닝 모델의 사용에 동기를 부여합니다.
저희는 다음 절에서 이를 논의할 것입니다.


## 요약

텍스트는 딥러닝에서 마주치는 가장 흔한 형태의 시퀀스 데이터 중 하나입니다.
무엇이 토큰을 구성하는지에 대한 흔한 선택은 문자, 단어, 워드피스입니다.
텍스트를 전처리하기 위해, 저희는 보통 (i) 텍스트를 토큰으로 분할하고, (ii) 토큰 문자열을 수치 인덱스로 매핑하는 어휘를 구성하며, (iii) 모델이 다룰 수 있도록 텍스트 데이터를 토큰 인덱스로 변환합니다.
실제로 단어의 빈도는 지프의 법칙을 따르는 경향이 있습니다. 이는 개별 단어(유니그램)뿐만 아니라 $n$-그램에도 마찬가지로 적용됩니다.


## 연습문제

1. 이 절의 실험에서, 텍스트를 단어로 토큰화하고 `Vocab` 인스턴스의 `min_freq` 인자 값을 변화시켜 보세요. `min_freq`의 변화가 결과 어휘의 크기에 어떻게 영향을 미치는지 정성적으로 특징지어 보세요.
1. 이 코퍼스에서 유니그램, 바이그램, 트라이그램에 대한 지프 분포의 지수를 추정하세요.
1. 다른 데이터 소스를 찾으세요(표준 머신러닝 데이터셋을 다운로드하거나, 다른 퍼블릭 도메인 책을 고르거나,
   웹사이트를 스크래핑하는 등). 각각에 대해, 단어 수준과 문자 수준 모두에서 데이터를 토큰화하세요. 동등한 `min_freq` 값에서 어휘 크기가 *The Time Machine* 코퍼스와 어떻게 비교되나요? 이 코퍼스들에 대한 유니그램과 바이그램 분포에 해당하는 지프 분포의 지수를 추정하세요. 그것들이 *The Time Machine* 코퍼스에 대해 관찰한 값과 어떻게 비교되나요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18011)
:end_tab:
