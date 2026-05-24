# 시퀀스 인식 추천 시스템

이전 절에서 저희는 사용자의 단기 행동을 고려하지 않고 추천 작업을 행렬 완성 문제로 추상화했습니다. 이 절에서는 순차적으로 정렬된 사용자 상호작용 로그를 고려하는 추천 모델을 소개할 것입니다. 이는 시퀀스 인식 추천기 :cite:`Quadrana.Cremonesi.Jannach.2018`로, 입력은 정렬되고 종종 타임스탬프가 있는 과거 사용자 행동 목록입니다. 여러 최근 문헌은 사용자의 시간적 행동 패턴을 모델링하고 그들의 관심 변화를 발견하는 데 있어 이러한 정보를 통합하는 것의 유용성을 입증했습니다.

저희가 소개할 모델인 Caser :cite:`Tang.Wang.2018`는 합성곱 시퀀스 임베딩 추천 모델(convolutional sequence embedding recommendation model)의 줄임말로, 사용자의 최근 활동의 동적 패턴 영향을 포착하기 위해 합성곱 신경망을 채택합니다. Caser의 주요 구성 요소는 수평 합성곱 네트워크와 수직 합성곱 네트워크로 구성되며, 각각 결합 수준과 점 수준의 시퀀스 패턴을 발견하는 것을 목표로 합니다. 점 수준 패턴은 과거 시퀀스의 단일 아이템이 대상 아이템에 미치는 영향을 나타내며, 결합 수준 패턴은 이전 몇 가지 행동이 후속 대상에 미치는 영향을 의미합니다. 예를 들어, 우유와 버터를 함께 구매하는 것은 둘 중 하나만 구매하는 것보다 밀가루를 구매할 확률이 더 높아집니다. 또한, 사용자의 일반적인 관심사 또는 장기 선호도도 마지막 완전 연결 계층에서 모델링되어, 사용자 관심사를 더 포괄적으로 모델링합니다. 모델의 세부 사항은 다음과 같습니다.

## 모델 구조

시퀀스 인식 추천 시스템에서 각 사용자는 아이템 집합에서 일부 아이템의 시퀀스와 연관됩니다. $S^u = (S_1^u, ... S_{|S_u|}^u)$가 정렬된 시퀀스를 나타낸다고 합시다. Caser의 목표는 사용자의 일반적인 취향과 단기적인 의도를 모두 고려하여 아이템을 추천하는 것입니다. 이전 $L$개의 아이템을 고려한다고 가정하면, 시간 단계 $t$에 대한 이전 상호작용을 나타내는 임베딩 행렬을 구성할 수 있습니다.

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

여기서 $\mathbf{Q} \in \mathbb{R}^{n \times k}$는 아이템 임베딩을 나타내며 $\mathbf{q}_i$는 $i^\textrm{th}$ 행을 나타냅니다. $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$는 시간 단계 $t$에서 사용자 $u$의 일시적인 관심을 추론하는 데 사용될 수 있습니다. 저희는 입력 행렬 $\mathbf{E}^{(u, t)}$를 후속 두 합성곱 구성 요소의 입력인 이미지로 볼 수 있습니다.

수평 합성곱 계층은 $d$개의 수평 필터 $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$를 가지며, 수직 합성곱 계층은 $d'$개의 수직 필터 $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$를 갖습니다. 일련의 합성곱과 풀 연산 후에, 저희는 두 출력을 얻습니다.

$$
\mathbf{o} = \textrm{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \textrm{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

여기서 $\mathbf{o} \in \mathbb{R}^d$는 수평 합성곱 네트워크의 출력이며 $\mathbf{o}' \in \mathbb{R}^{kd'}$는 수직 합성곱 네트워크의 출력입니다. 간결성을 위해 저희는 합성곱과 풀 연산의 세부 사항을 생략합니다. 이들은 연결되어 완전 연결 신경망 계층으로 공급되어 더 높은 수준의 표현을 얻습니다.

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

여기서 $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$는 가중치 행렬이며 $\mathbf{b} \in \mathbb{R}^k$는 편향입니다. 학습된 벡터 $\mathbf{z} \in \mathbb{R}^k$는 사용자의 단기 의도의 표현입니다.

마지막으로 예측 함수는 사용자의 단기 및 일반적 취향을 결합하며, 다음과 같이 정의됩니다.

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

여기서 $\mathbf{V} \in \mathbb{R}^{n \times 2k}$는 또 다른 아이템 임베딩 행렬입니다. $\mathbf{b}' \in \mathbb{R}^n$는 아이템별 편향입니다. $\mathbf{P} \in \mathbb{R}^{m \times k}$는 사용자의 일반적 취향을 위한 사용자 임베딩 행렬입니다. $\mathbf{p}_u \in \mathbb{R}^{ k}$는 $P$의 $u^\textrm{th}$ 행이며 $\mathbf{v}_i \in \mathbb{R}^{2k}$는 $\mathbf{V}$의 $i^\textrm{th}$ 행입니다.

이 모델은 BPR 또는 힌지 손실로 학습될 수 있습니다. Caser의 구조는 아래와 같습니다.

![Caser 모델 예시](../img/rec-caser.svg)

먼저 필요한 라이브러리를 임포트합니다.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## 모델 구현
다음 코드는 Caser 모델을 구현합니다. 이는 수직 합성곱 계층, 수평 합성곱 계층, 그리고 완전 연결 계층으로 구성됩니다.

```{.python .input  n=4}
#@tab mxnet
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## 부정 샘플링을 사용한 순차 데이터셋
순차 상호작용 데이터를 처리하기 위해, 저희는 `Dataset` 클래스를 재구현해야 합니다. 다음 코드는 `SeqDataset`라는 새로운 데이터셋 클래스를 생성합니다. 각 샘플에서 사용자 ID, 시퀀스로서의 이전 $L$개 상호작용 아이템, 그리고 다음으로 상호작용하는 아이템을 대상으로 출력합니다. 다음 그림은 한 사용자에 대한 데이터 로딩 프로세스를 보여줍니다. 이 사용자가 9편의 영화를 좋아했다고 가정하면, 저희는 이 9편의 영화를 시간순으로 정리합니다. 가장 최근의 영화는 테스트 아이템으로 남겨집니다. 나머지 8편의 영화에 대해, 저희는 각 샘플이 5개($L=5$)의 영화 시퀀스와 그 후속 아이템을 대상 아이템으로 포함하는 3개의 훈련 샘플을 얻을 수 있습니다. 부정 샘플도 사용자 정의 데이터셋에 포함됩니다.

![데이터 생성 프로세스 예시](../img/rec-seq-data.svg)

```{.python .input  n=5}
#@tab mxnet
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## MovieLens 100K 데이터셋 로드

이후 저희는 MovieLens 100K 데이터셋을 시퀀스 인식 모드로 읽고 분할하며, 위에서 구현한 순차 데이터로더로 훈련 데이터를 로드합니다.

```{.python .input  n=6}
#@tab mxnet
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

훈련 데이터 구조가 위에 표시되어 있습니다. 첫 번째 요소는 사용자 ID이고, 다음 리스트는 이 사용자가 좋아한 마지막 5개의 아이템을 나타내며, 마지막 요소는 이 사용자가 그 5개 아이템 다음으로 좋아한 아이템입니다.

## 모델 훈련
이제 모델을 훈련합시다. 저희는 결과를 비교할 수 있도록 학습률, 최적화기, $k$를 포함하여 지난 절의 NeuMF와 동일한 설정을 사용합니다.

```{.python .input  n=7}
#@tab mxnet
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# Running takes > 1h (pending fix from MXNet)
# d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter, num_users, num_items, num_epochs, devices, d2l.evaluate_ranking, candidates, eval_step=1)
```

## 요약
* 사용자의 단기 및 장기 관심을 추론하면 사용자가 선호한 다음 아이템의 예측을 더 효과적으로 만들 수 있습니다.
* 합성곱 신경망은 순차적 상호작용에서 사용자의 단기 관심을 포착하는 데 활용될 수 있습니다.

## 연습문제

* 수평과 수직 합성곱 네트워크 중 하나를 제거하여 어블레이션 연구를 수행하십시오. 어느 구성 요소가 더 중요합니까?
* 하이퍼파라미터 $L$을 변화시켜 보십시오. 더 긴 과거 상호작용이 더 높은 정확도를 가져옵니까?
* 위에서 소개한 시퀀스 인식 추천 작업 외에도, 세션 기반 추천이라고 불리는 또 다른 유형의 시퀀스 인식 추천 작업이 있습니다 :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`. 이 두 작업 간의 차이점을 설명할 수 있습니까?


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/404)
:end_tab:
