# 자연어 추론: BERT 파인튜닝
:label:`sec_natural-language-inference-bert`

이 장의 앞선 절들에서,
저희는 SNLI 데이터셋(:numref:`sec_natural-language-inference-and-dataset`에서 설명됨)의
자연어 추론 작업을 위한
어텐션 기반 아키텍처를(:numref:`sec_natural-language-inference-attention`에서)
설계했습니다.
이제 BERT를 파인튜닝하여 이 작업을 다시 살펴봅니다.
:numref:`sec_finetuning-bert`에서 논의한 것처럼,
자연어 추론은 시퀀스 수준 텍스트 쌍 분류 문제이며,
BERT 파인튜닝은 :numref:`fig_nlp-map-nli-bert`에 나타난 것처럼
추가적인 MLP 기반 아키텍처만을 요구합니다.

![이 절에서는 자연어 추론을 위해 사전 학습된 BERT를 MLP 기반 아키텍처에 입력합니다.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

이 절에서는,
저희가 사전 학습된 작은 버전의 BERT를 다운로드한 다음,
SNLI 데이터셋에서의 자연어 추론을 위해
이를 파인튜닝할 것입니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## [**사전 학습된 BERT 로딩하기**]

저희는 :numref:`sec_bert-dataset`과 :numref:`sec_bert-pretraining`에서
WikiText-2 데이터셋으로 BERT를 사전 학습하는 방법을 설명했습니다
(원본 BERT 모델은 훨씬 더 큰 말뭉치에서 사전 학습됨에 유의하세요).
:numref:`sec_bert-pretraining`에서 논의한 것처럼,
원본 BERT 모델은 수억 개의 파라미터를 가지고 있습니다.
다음에서,
저희는 두 가지 버전의 사전 학습된 BERT를 제공합니다:
"bert.base"는 파인튜닝에 많은 계산 자원을 요구하는 원본 BERT base 모델과 거의 같은 크기이며,
"bert.small"은 시연을 용이하게 하기 위한 작은 버전입니다.

```{.python .input}
#@tab mxnet
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

두 사전 학습된 BERT 모델 모두 어휘 집합을 정의하는 "vocab.json" 파일과
사전 학습된 파라미터의 "pretrained.params" 파일을 포함합니다.
저희는 [**사전 학습된 BERT 파라미터를 로드**]하기 위해 다음 `load_pretrained_model` 함수를 구현합니다.

```{.python .input}
#@tab mxnet
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_blks, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

대부분의 머신에서 시연을 용이하게 하기 위해,
저희는 이 절에서 사전 학습된 BERT의 작은 버전("bert.small")을 로드하고 파인튜닝할 것입니다.
연습문제에서는, 테스트 정확도를 상당히 향상시키기 위해 훨씬 큰 "bert.base"를 파인튜닝하는 방법을 보여드릴 것입니다.

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices)
```

## [**BERT 파인튜닝을 위한 데이터셋**]

SNLI 데이터셋의 다운스트림 작업 자연어 추론을 위해,
저희는 커스터마이즈된 데이터셋 클래스 `SNLIBERTDataset`을 정의합니다.
각 예제에서,
전제와 가설은 텍스트 시퀀스 쌍을 형성하고
:numref:`fig_bert-two-seqs`에 나타난 것처럼 하나의 BERT 입력 시퀀스로 패킹됩니다.
세그먼트 ID가 BERT 입력 시퀀스에서 전제와 가설을 구분하는 데 사용됨을
:numref:`subsec_bert_input_rep`에서 상기하세요.
BERT 입력 시퀀스의 미리 정의된 최대 길이(`max_len`)와 함께,
입력 텍스트 쌍 중 더 긴 것의 마지막 토큰이
`max_len`을 만족할 때까지 계속 제거됩니다.
BERT 파인튜닝을 위한
SNLI 데이터셋의 생성을 가속화하기 위해,
저희는 학습 또는 테스트 예제를 병렬로 생성하기 위해 4개의 워커 프로세스를 사용합니다.

```{.python .input}
#@tab mxnet
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

SNLI 데이터셋을 다운로드한 후,
저희는 `SNLIBERTDataset` 클래스를 인스턴스화하여
[**학습 및 테스트 예제를 생성**]합니다.
이러한 예제는 자연어 추론의 학습과 테스트 동안
미니배치로 읽힐 것입니다.

```{.python .input}
#@tab mxnet
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## BERT 파인튜닝

:numref:`fig_bert-two-seqs`가 나타내는 것처럼,
자연어 추론을 위한 BERT 파인튜닝은
두 개의 완전 연결 레이어로 구성된 추가적인 MLP만을 요구합니다
(다음 `BERTClassifier` 클래스의 `self.hidden`과 `self.output`을 참조하세요).
[**이 MLP는
특수 “&lt;cls&gt;” 토큰의 BERT 표현을**] 변환하여,
이 토큰은 전제와 가설 모두의 정보를 인코딩하며,
(**자연어 추론의 세 가지 출력으로 변환합니다**):
함의, 모순, 그리고 중립입니다.

```{.python .input}
#@tab mxnet
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

다음에서,
사전 학습된 BERT 모델 `bert`는
다운스트림 응용을 위해 `BERTClassifier` 인스턴스 `net`에 입력됩니다.
BERT 파인튜닝의 일반적인 구현에서,
추가적인 MLP의 출력 레이어(`net.output`)의 파라미터만이 처음부터 학습될 것입니다.
사전 학습된 BERT 인코더(`net.encoder`)와 추가적인 MLP의 은닉 레이어(`net.hidden`)의 모든 파라미터는 파인튜닝될 것입니다.

```{.python .input}
#@tab mxnet
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

:numref:`sec_bert`에서
`MaskLM` 클래스와 `NextSentencePred` 클래스
모두 사용된 MLP에 파라미터가 있다는 것을 상기하세요.
이러한 파라미터는 사전 학습된 BERT 모델
`bert`의 파라미터의 일부이며, 따라서 `net`의 파라미터의 일부입니다.
그러나, 그러한 파라미터는 사전 학습 동안 마스크된 언어 모델링 손실과
다음 문장 예측 손실을 계산하는 데에만
사용됩니다.
이 두 손실 함수는 다운스트림 응용의 파인튜닝과는 무관하므로,
BERT가 파인튜닝될 때
`MaskLM`과 `NextSentencePred`에서 사용된 MLP의 파라미터는 업데이트되지 않습니다(stale).

오래된(stale) 그래디언트를 가진 파라미터를 허용하기 위해,
`d2l.train_batch_ch13`의 `step` 함수에서 플래그 `ignore_stale_grad=True`가 설정됩니다.
저희는 SNLI의 학습 세트(`train_iter`)와 테스트 세트(`test_iter`)를 사용하여
모델 `net`을 학습하고 평가하기 위해 이 함수를 사용합니다.
제한된 계산 자원으로 인해, [**학습**] 및 테스트 정확도는
더 향상될 수 있습니다: 그 논의는 연습문제에 남겨둡니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
net(next(iter(train_iter))[0])
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 요약

* 저희는 SNLI 데이터셋의 자연어 추론과 같은 다운스트림 응용을 위해 사전 학습된 BERT 모델을 파인튜닝할 수 있습니다.
* 파인튜닝 동안, BERT 모델은 다운스트림 응용을 위한 모델의 일부가 됩니다. 사전 학습 손실에만 관련된 파라미터는 파인튜닝 동안 업데이트되지 않을 것입니다.



## 연습문제

1. 계산 자원이 허용한다면 원본 BERT base 모델과 거의 같은 크기인 훨씬 더 큰 사전 학습된 BERT 모델을 파인튜닝하세요. `load_pretrained_model` 함수의 인자를 다음과 같이 설정하세요: 'bert.small'을 'bert.base'로 대체하고, `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4`, `num_blks=2`의 값을 각각 768, 3072, 12, 12로 증가시키세요. 파인튜닝 에포크를 증가시키면(그리고 다른 하이퍼파라미터를 조정하면), 0.86보다 높은 테스트 정확도를 얻을 수 있습니까?
1. 시퀀스 쌍을 그 길이의 비율에 따라 어떻게 잘라낼 수 있습니까? 이 쌍 자르기 방법과 `SNLIBERTDataset` 클래스에서 사용된 방법을 비교해 보세요. 각각의 장단점은 무엇입니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1526)
:end_tab:
