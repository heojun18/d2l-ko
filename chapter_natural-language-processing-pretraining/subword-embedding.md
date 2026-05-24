# 서브워드 임베딩
:label:`sec_fasttext`

영어에서,
"helps", "helped", "helping" 같은 단어는
같은 단어 "help"의 굴절된 형태입니다.
"dog"와 "dogs"의 관계는
"cat"과 "cats"의 관계와 같으며,
"boy"와 "boyfriend"의 관계는
"girl"과 "girlfriend"의 관계와 같습니다.
프랑스어와 스페인어 같은 다른 언어에서는,
많은 동사가 40개 이상의 굴절 형태를 가지며,
핀란드어에서는,
명사가 최대 15개의 격(case)을 가질 수 있습니다.
언어학에서,
형태론(morphology)은 단어 형성과 단어 관계를 연구합니다.
그러나,
단어의 내부 구조는
word2vec과 GloVe 모두에서 탐구되지 않았습니다.

## fastText 모델

word2vec에서 단어가 어떻게 표현되는지 떠올려 보십시오.
스킵그램 모델과 연속 단어 묶음 모델 모두에서,
같은 단어의 서로 다른 굴절 형태는
공유 파라미터 없이 서로 다른 벡터로 직접 표현됩니다.
형태론적 정보를 사용하기 위해,
*fastText* 모델은
*서브워드 임베딩(subword embedding)* 접근법을 제안했으며,
여기서 서브워드는 문자 $n$-그램입니다 :cite:`Bojanowski.Grave.Joulin.ea.2017`.
단어 수준의 벡터 표현을 학습하는 대신,
fastText는 서브워드 수준의 스킵그램으로 간주될 수 있으며,
여기서 각 *중심 단어* 는
그 서브워드 벡터들의 합으로 표현됩니다.

"where"라는 단어를 사용해
fastText에서 각 중심 단어에 대해 서브워드를
어떻게 얻는지 설명해 봅시다.
먼저, 단어의 시작과 끝에 특수 문자 "&lt;"와 "&gt;"를 추가하여
접두사와 접미사를 다른 서브워드와 구별합니다.
그런 다음, 단어에서 문자 $n$-그램을 추출합니다.
예를 들어, $n=3$일 때,
저희는 길이 3의 모든 서브워드를 얻습니다. "&lt;wh", "whe", "her", "ere", "re&gt;"와 특수 서브워드 "&lt;where&gt;"입니다.


fastText에서, 임의의 단어 $w$에 대해,
길이가 3과 6 사이인 모든 서브워드와 특수 서브워드의
합집합을 $\mathcal{G}_w$로 표시합니다.
어휘는
모든 단어의 서브워드의 합집합입니다.
사전에서 서브워드 $g$의 벡터를
$\mathbf{z}_g$라 하면,
스킵그램 모델에서 중심 단어로서 단어 $w$의
벡터 $\mathbf{v}_w$는
그 서브워드 벡터들의 합입니다.

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

fastText의 나머지는 스킵그램 모델과 동일합니다. 스킵그램 모델과 비교하여,
fastText의 어휘는 더 크고,
이는 더 많은 모델 파라미터로 이어집니다.
게다가,
단어의 표현을 계산하기 위해,
그것의 모든 서브워드 벡터를
합산해야 하며,
이는 더 높은 계산 복잡도로 이어집니다.
그러나,
유사한 구조를 가진 단어들 사이에서 서브워드로부터의 공유 파라미터 덕분에,
드문 단어와 심지어 어휘 외(out-of-vocabulary) 단어도
fastText에서 더 나은 벡터 표현을 얻을 수 있습니다.



## 바이트 페어 인코딩
:label:`subsec_Byte_Pair_Encoding`

fastText에서, 추출된 모든 서브워드는 $3$부터 $6$ 같은 지정된 길이를 가져야 하므로, 어휘 크기는 미리 정의될 수 없습니다.
고정 크기 어휘에서 가변 길이 서브워드를 허용하기 위해,
저희는 서브워드를 추출하는 데
*바이트 페어 인코딩(byte pair encoding, BPE)* 이라는 압축 알고리즘을 적용할 수 있습니다 :cite:`Sennrich.Haddow.Birch.2015`.

바이트 페어 인코딩은 임의 길이의 연속된 문자 같이 단어 내의 공통 심볼을 발견하기 위해 학습 데이터셋의 통계적 분석을 수행합니다.
길이 1의 심볼부터 시작하여,
바이트 페어 인코딩은 가장 빈번한 연속된 심볼 쌍을 반복적으로 병합하여 새로운 더 긴 심볼을 생성합니다.
효율을 위해, 단어 경계를 가로지르는 쌍은 고려되지 않음에 유의하십시오.
결국, 저희는 이러한 심볼을 서브워드로 사용해 단어를 분할할 수 있습니다.
바이트 페어 인코딩과 그 변종은 GPT-2 :cite:`Radford.Wu.Child.ea.2019`와 RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` 같이 인기 있는 자연어 처리 사전 학습 모델의 입력 표현에 사용되어 왔습니다.
다음에서는, 바이트 페어 인코딩이 어떻게 작동하는지 설명하겠습니다.

먼저, 모든 영어 소문자 글자, 특수 단어 종료 심볼 `'_'`, 그리고 특수 알 수 없음 심볼 `'[UNK]'`로 심볼 어휘를 초기화합니다.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

단어 경계를 가로지르는 심볼 쌍은 고려하지 않으므로,
데이터셋에서 단어를 그 빈도(등장 횟수)로 매핑하는
`raw_token_freqs` 사전만 필요합니다.
출력 심볼 시퀀스(예: "a_ tall er_ man")에서
단어 시퀀스(예: "a taller man")를 쉽게 복원할 수 있도록
각 단어에 특수 심볼 `'_'`가 추가됨에 유의하십시오.
저희는 오직 단일 문자와 특수 심볼의 어휘에서 병합 과정을 시작하므로, 각 단어(사전 `token_freqs`의 키) 내 모든 연속된 문자 쌍 사이에 공백이 삽입됩니다.
다시 말해, 공백은 단어 내 심볼 사이의 구분자입니다.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

저희는 다음의 `get_max_freq_pair` 함수를 정의하는데,
이는 입력 사전 `token_freqs`의 키에서 나오는 단어 내에서
가장 빈번한 연속된 심볼 쌍을 반환합니다.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

연속된 심볼의 빈도에 기반한 탐욕적 접근법으로서,
바이트 페어 인코딩은 가장 빈번한 연속된 심볼 쌍을 병합하여 새로운 심볼을 생성하기 위해 다음의 `merge_symbols` 함수를 사용할 것입니다.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

이제 사전 `token_freqs`의 키에 대해 바이트 페어 인코딩 알고리즘을 반복적으로 수행합니다. 첫 번째 반복에서, 가장 빈번한 연속된 심볼 쌍은 `'t'`와 `'a'`이므로, 바이트 페어 인코딩은 이들을 병합하여 새로운 심볼 `'ta'`를 생성합니다. 두 번째 반복에서, 바이트 페어 인코딩은 계속해서 `'ta'`와 `'l'`을 병합하여 또 다른 새로운 심볼 `'tal'`을 결과로 얻습니다.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

바이트 페어 인코딩의 10회 반복 후, 리스트 `symbols`가 이제 다른 심볼들로부터 반복적으로 병합된 10개의 더 많은 심볼을 포함하는 것을 볼 수 있습니다.

```{.python .input}
#@tab all
print(symbols)
```

사전 `raw_token_freqs`의 키에 지정된 같은 데이터셋에 대해,
바이트 페어 인코딩 알고리즘의 결과로
데이터셋의 각 단어는 이제 서브워드 "fast_", "fast", "er_", "tall_", "tall"로 분할됩니다.
예를 들어, 단어 "faster_"와 "taller_"는 각각 "fast er_"와 "tall er_"로 분할됩니다.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

바이트 페어 인코딩의 결과는 사용되는 데이터셋에 의존함에 유의하십시오.
저희는 또한 한 데이터셋에서 학습한 서브워드를 사용해
다른 데이터셋의 단어를 분할할 수도 있습니다.
탐욕적 접근법으로서, 다음의 `segment_BPE` 함수는 입력 인수 `symbols`에서 가능한 가장 긴 서브워드로 단어를 분할하려고 시도합니다.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

다음에서, 저희는 앞서 언급한 데이터셋에서 학습한 리스트 `symbols`의 서브워드를 사용해,
또 다른 데이터셋을 표현하는 `tokens`을 분할합니다.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## 요약

* fastText 모델은 서브워드 임베딩 접근법을 제안합니다. word2vec의 스킵그램 모델을 기반으로, 중심 단어를 그 서브워드 벡터들의 합으로 표현합니다.
* 바이트 페어 인코딩은 단어 내의 공통 심볼을 발견하기 위해 학습 데이터셋의 통계적 분석을 수행합니다. 탐욕적 접근법으로서, 바이트 페어 인코딩은 가장 빈번한 연속된 심볼 쌍을 반복적으로 병합합니다.
* 서브워드 임베딩은 드문 단어와 사전 외 단어의 표현 품질을 개선할 수 있습니다.

## 연습문제

1. 예로, 영어에는 약 $3\times 10^8$개의 가능한 $6$-그램이 있습니다. 서브워드가 너무 많을 때 문제는 무엇입니까? 그 문제를 어떻게 해결합니까? 힌트: fastText 논문 :cite:`Bojanowski.Grave.Joulin.ea.2017`의 3.2절 끝부분을 참조하십시오.
1. 연속 단어 묶음 모델에 기반하여 서브워드 임베딩 모델을 어떻게 설계합니까?
1. 크기 $m$의 어휘를 얻기 위해, 초기 심볼 어휘 크기가 $n$일 때 몇 번의 병합 연산이 필요합니까?
1. 구절을 추출하기 위해 바이트 페어 인코딩의 아이디어를 어떻게 확장합니까?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4587)
:end_tab:
