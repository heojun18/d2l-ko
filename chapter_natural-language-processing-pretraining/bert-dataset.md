# BERT 사전 학습을 위한 데이터셋
:label:`sec_bert-dataset`

:numref:`sec_bert`에서 구현된 BERT 모델을 사전 학습하기 위해,
저희는 두 사전 학습 작업, 즉 마스킹된 언어 모델링과 다음 문장 예측을 수월하게 하기 위해
이상적인 형식으로 데이터셋을 생성해야 합니다.
한편으로는,
원래의 BERT 모델은 두 거대한 말뭉치 BookCorpus와 영어 위키피디아의 연결로 사전 학습되며(:numref:`subsec_bert_pretraining_tasks` 참조),
이 책의 대부분 독자가 실행하기 어렵게 만듭니다.
다른 한편으로는,
기성 사전 학습된 BERT 모델은
의학 같은 특정 도메인의 응용에 맞지 않을 수 있습니다.
따라서, 맞춤형 데이터셋에서 BERT를 사전 학습하는 것이 인기를 얻고 있습니다.
BERT 사전 학습의 시연을 수월하게 하기 위해,
저희는 더 작은 말뭉치 WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`를 사용합니다.

:numref:`sec_word2vec_data`에서 word2vec 사전 학습에 사용된 PTB 데이터셋과 비교하면,
WikiText-2는 (i) 원래의 구두점을 유지하여 다음 문장 예측에 적합하게 하고, (ii) 원래의 대소문자와 숫자를 유지하며, (iii) 두 배 이상 큽니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

[**WikiText-2 데이터셋**]에서,
각 줄은 문단을 나타내며
구두점과 그 앞 토큰 사이에 공백이 삽입되어 있습니다.
적어도 두 문장 이상인 문단이 유지됩니다.
문장을 분리하기 위해, 저희는 단순화를 위해 마침표만을 구분자로 사용합니다.
더 복잡한 문장 분리 기법에 대한 논의는 이 절 끝의 연습문제에 남겨둡니다.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## 사전 학습 작업을 위한 헬퍼 함수 정의

다음에서,
저희는 두 BERT 사전 학습 작업을 위한 헬퍼 함수를 구현하는 것부터 시작합니다.
다음 문장 예측과 마스킹된 언어 모델링입니다.
이러한 헬퍼 함수는
나중에 원본 텍스트 말뭉치를
BERT를 사전 학습하기 위한 이상적인 형식의 데이터셋으로 변환할 때
호출될 것입니다.

### [**다음 문장 예측 작업 생성**]

:numref:`subsec_nsp`의 설명에 따라,
`_get_next_sentence` 함수는
이진 분류 작업을 위한 학습 예제를 생성합니다.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

다음 함수는 `_get_next_sentence` 함수를 호출하여
입력 `paragraph`로부터 다음 문장 예측을 위한 학습 예제를 생성합니다.
여기서 `paragraph`는 문장의 리스트이며, 각 문장은 토큰의 리스트입니다.
인수 `max_len`은 사전 학습 동안 BERT 입력 시퀀스의 최대 길이를 지정합니다.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### [**마스킹된 언어 모델링 작업 생성**]
:label:`subsec_prepare_mlm_data`

BERT 입력 시퀀스에서
마스킹된 언어 모델링 작업을 위한 학습 예제를 생성하기 위해,
저희는 다음의 `_replace_mlm_tokens` 함수를 정의합니다.
그 입력에서, `tokens`는 BERT 입력 시퀀스를 나타내는 토큰의 리스트이고,
`candidate_pred_positions`는 특수 토큰의 인덱스를 제외한 BERT 입력 시퀀스의 토큰 인덱스 리스트이며(특수 토큰은 마스킹된 언어 모델링 작업에서 예측되지 않습니다),
`num_mlm_preds`는 예측의 수(15% 무작위 토큰을 예측함을 떠올리십시오)를 가리킵니다.
:numref:`subsec_mlm`의 마스킹된 언어 모델링 작업의 정의에 따라,
각 예측 위치에서, 입력은
특수 "&lt;mask&gt;" 토큰 또는 무작위 토큰으로 대체되거나, 변경되지 않을 수 있습니다.
결국, 함수는 가능한 대체 후의 입력 토큰,
예측이 발생하는 토큰 인덱스, 그리고 이러한 예측에 대한 레이블을 반환합니다.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

앞서 언급한 `_replace_mlm_tokens` 함수를 호출함으로써,
다음 함수는 BERT 입력 시퀀스(`tokens`)를 입력으로 받아
입력 토큰의 인덱스(:numref:`subsec_mlm`에서 설명된 가능한 토큰 대체 후),
예측이 발생하는 토큰 인덱스,
그리고 이러한 예측에 대한 레이블 인덱스를 반환합니다.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## 텍스트를 사전 학습 데이터셋으로 변환

이제 저희는 BERT를 사전 학습하기 위한 `Dataset` 클래스를 맞춤 제작할 준비가 거의 되었습니다.
그 전에,
[**입력에 특수 "&lt;pad&gt;" 토큰을 추가하기**] 위해
헬퍼 함수 `_pad_bert_inputs`를 여전히 정의해야 합니다.
그 인수 `examples`는 두 사전 학습 작업을 위한 헬퍼 함수 `_get_nsp_data_from_paragraph`와 `_get_mlm_data_from_tokens`로부터의 출력을 포함합니다.

```{.python .input}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

두 사전 학습 작업의 학습 예제를 생성하기 위한 헬퍼 함수와
입력을 패딩하기 위한 헬퍼 함수를 함께 묶어서,
저희는 다음의 `_WikiTextDataset` 클래스를 [**BERT 사전 학습을 위한 WikiText-2 데이터셋**]으로 맞춤 제작합니다.
`__getitem__` 함수를 구현함으로써,
저희는 WikiText-2 말뭉치에서 한 쌍의 문장으로부터 생성된 사전 학습(마스킹된 언어 모델링과 다음 문장 예측) 예제에
임의로 접근할 수 있습니다.

원래의 BERT 모델은 어휘 크기가 30000인 WordPiece 임베딩을 사용합니다 :cite:`Wu.Schuster.Chen.ea.2016`.
WordPiece의 토큰화 방법은
:numref:`subsec_Byte_Pair_Encoding`의 원래 바이트 페어 인코딩 알고리즘의 약간의 수정입니다.
단순화를 위해, 저희는 토큰화를 위해 `d2l.tokenize` 함수를 사용합니다.
다섯 번 미만 등장하는 드문 토큰은 필터링됩니다.

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

`_read_wiki` 함수와 `_WikiTextDataset` 클래스를 사용함으로써,
저희는 [**WikiText-2 데이터셋을 다운로드하고 그로부터 사전 학습 예제를 생성하기**] 위해
다음의 `load_data_wiki`를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

배치 크기를 512로 설정하고 BERT 입력 시퀀스의 최대 길이를 64로 두고,
저희는 [**BERT 사전 학습 예제의 미니배치 형상을 출력합니다**].
각 BERT 입력 시퀀스에서,
$10$개 ($64 \times 0.15$) 위치가 마스킹된 언어 모델링 작업을 위해 예측됨에 유의하십시오.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

마지막으로, 어휘 크기를 살펴봅시다.
드문 토큰을 필터링한 후에도,
PTB 데이터셋의 그것보다 여전히 두 배 이상 큽니다.

```{.python .input}
#@tab all
len(vocab)
```

## 요약

* PTB 데이터셋과 비교하여, WikiText-2 데이터셋은 원래의 구두점, 대소문자, 숫자를 유지하며, 두 배 이상 큽니다.
* 저희는 WikiText-2 말뭉치에서 한 쌍의 문장으로부터 생성된 사전 학습(마스킹된 언어 모델링과 다음 문장 예측) 예제에 임의로 접근할 수 있습니다.


## 연습문제

1. 단순화를 위해, 마침표가 문장을 분리하는 유일한 구분자로 사용됩니다. spaCy와 NLTK 같은 다른 문장 분리 기법을 시도해 보십시오. NLTK를 예로 들어 봅시다. 먼저 NLTK를 설치해야 합니다. `pip install nltk`. 코드에서, 먼저 `import nltk`. 그런 다음, Punkt 문장 토크나이저를 다운로드합니다. `nltk.download('punkt')`. `sentences = 'This is great ! Why not ?'`와 같은 문장을 분리하려면, `nltk.tokenize.sent_tokenize(sentences)`를 호출하면 두 문장 문자열 리스트 `['This is great !', 'Why not ?']`를 반환할 것입니다.
1. 드문 토큰을 필터링하지 않으면 어휘 크기는 얼마입니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:
