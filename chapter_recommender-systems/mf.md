# 행렬 분해

행렬 분해 :cite:`Koren.Bell.Volinsky.2009`는 추천 시스템 문헌에서 잘 확립된 알고리즘입니다. 행렬 분해 모델의 첫 번째 버전은 Simon Funk가 유명한 [블로그 게시물](https://sifter.org/%7Esimon/journal/20061211.html)에서 제안했으며, 그는 그 게시물에서 상호작용 행렬을 분해하는 아이디어를 설명했습니다. 그 후 2006년에 개최된 Netflix 콘테스트로 인해 널리 알려지게 되었습니다. 당시 미디어 스트리밍과 비디오 대여 회사인 Netflix는 자사의 추천 시스템 성능을 향상시키기 위한 콘테스트를 발표했습니다. Netflix의 기준선(즉, Cinematch)을 10% 개선할 수 있는 최고의 팀에게 100만 달러의 상금을 수여한다는 내용이었습니다. 이렇게, 이 콘테스트는 추천 시스템 연구 분야에 많은 관심을 끌었습니다. 결과적으로 대상은 BellKor, Pragmatic Theory, BigChaos의 합동 팀인 BellKor's Pragmatic Chaos 팀이 수상했습니다(이러한 알고리즘에 대해 지금 걱정할 필요는 없습니다). 비록 최종 점수는 앙상블 솔루션(즉, 많은 알고리즘의 조합)의 결과였지만, 행렬 분해 알고리즘은 최종 조합에서 중요한 역할을 했습니다. Netflix 대상 솔루션의 기술 보고서 :cite:`Toscher.Jahrer.Bell.2009`는 채택된 모델에 대한 상세한 소개를 제공합니다. 이 절에서는 행렬 분해 모델의 세부 사항과 그 구현에 대해 자세히 살펴보겠습니다.


## 행렬 분해 모델

행렬 분해는 협업 필터링 모델의 한 부류입니다. 구체적으로, 이 모델은 사용자-아이템 상호작용 행렬(예: 평점 행렬)을 두 개의 낮은 랭크의 행렬의 곱으로 분해하여 사용자-아이템 상호작용의 낮은 랭크 구조를 포착합니다.

$\mathbf{R} \in \mathbb{R}^{m \times n}$이 $m$명의 사용자와 $n$개의 아이템을 가진 상호작용 행렬을 나타내며, $\mathbf{R}$의 값들이 명시적 평점을 나타낸다고 합시다. 사용자-아이템 상호작용은 사용자 잠재 행렬 $\mathbf{P} \in \mathbb{R}^{m \times k}$와 아이템 잠재 행렬 $\mathbf{Q} \in \mathbb{R}^{n \times k}$로 분해될 것이며, 여기서 $k \ll m, n$은 잠재 요인 크기입니다. $\mathbf{p}_u$가 $\mathbf{P}$의 $u^\textrm{th}$ 행을 나타내고 $\mathbf{q}_i$가 $\mathbf{Q}$의 $i^\textrm{th}$ 행을 나타낸다고 합시다. 주어진 아이템 $i$에 대해, $\mathbf{q}_i$의 요소들은 그 아이템이 영화의 장르나 언어와 같은 특성을 어느 정도 갖고 있는지를 측정합니다. 주어진 사용자 $u$에 대해, $\mathbf{p}_u$의 요소들은 사용자가 아이템의 해당 특성에 얼마나 관심이 있는지를 측정합니다. 이러한 잠재 요인은 그러한 예시에서 언급된 명백한 차원을 측정할 수도 있고 완전히 해석 불가능할 수도 있습니다. 예측된 평점은 다음과 같이 추정될 수 있습니다.

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

여기서 $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$은 $\mathbf{R}$과 동일한 형태를 가진 예측된 평점 행렬입니다. 이 예측 규칙의 한 가지 주요 문제는 사용자/아이템 편향을 모델링할 수 없다는 점입니다. 예를 들어 일부 사용자는 더 높은 평점을 주는 경향이 있거나, 일부 아이템은 품질이 떨어져서 항상 더 낮은 평점을 받습니다. 이러한 편향은 실세계 응용 분야에서 흔히 발생합니다. 이러한 편향을 포착하기 위해 사용자별 그리고 아이템별 편향 항이 도입됩니다. 구체적으로 사용자 $u$가 아이템 $i$에게 부여하는 예측 평점은 다음과 같이 계산됩니다.

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

그런 다음 저희는 예측 평점 점수와 실제 평점 점수 간의 평균 제곱 오차를 최소화하여 행렬 분해 모델을 훈련합니다. 목적 함수는 다음과 같이 정의됩니다.

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

여기서 $\lambda$는 정규화율을 나타냅니다. 정규화 항 $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$는 파라미터의 크기에 페널티를 주어 과적합을 방지하는 데 사용됩니다. $\mathbf{R}_{ui}$가 알려진 $(u, i)$ 쌍은 집합 $\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \textrm{ is known}\}$에 저장됩니다. 모델 파라미터는 확률적 경사 하강법과 Adam과 같은 최적화 알고리즘으로 학습될 수 있습니다.

행렬 분해 모델의 직관적인 그림은 아래에 표시되어 있습니다.

![행렬 분해 모델의 예시](../img/rec-mf.svg)

이 절의 나머지 부분에서는 행렬 분해의 구현을 설명하고 MovieLens 데이터셋에서 모델을 훈련할 것입니다.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## 모델 구현

먼저, 저희는 위에서 설명한 행렬 분해 모델을 구현합니다. 사용자와 아이템 잠재 요인은 `nn.Embedding`으로 생성될 수 있습니다. `input_dim`은 아이템/사용자의 수이며 `output_dim`은 잠재 요인 $k$의 차원입니다. 저희는 또한 `output_dim`을 1로 설정하여 사용자/아이템 편향을 생성하기 위해 `nn.Embedding`을 사용할 수 있습니다. `forward` 함수에서 사용자와 아이템 ID는 임베딩을 조회하는 데 사용됩니다.

```{.python .input  n=4}
#@tab mxnet
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## 평가 측정 방법

그런 다음 저희는 RMSE(Root-Mean-Square Error, 평균 제곱근 오차) 측도를 구현하며, 이는 모델이 예측한 평점 점수와 실제로 관찰된 평점(정답) 간의 차이를 측정하는 데 일반적으로 사용됩니다 :cite:`Gunawardana.Shani.2015`. RMSE는 다음과 같이 정의됩니다.

$$
\textrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

여기서 $\mathcal{T}$는 평가하고자 하는 사용자와 아이템 쌍으로 구성된 집합입니다. $|\mathcal{T}|$는 이 집합의 크기입니다. 저희는 `mx.metric`에서 제공하는 RMSE 함수를 사용할 수 있습니다.

```{.python .input  n=3}
#@tab mxnet
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## 모델 훈련 및 평가


훈련 함수에서 저희는 가중치 감쇠를 사용한 $\ell_2$ 손실을 채택합니다. 가중치 감쇠 메커니즘은 $\ell_2$ 정규화와 동일한 효과를 갖습니다.

```{.python .input  n=4}
#@tab mxnet
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

마지막으로 모든 것을 모아 모델을 훈련합시다. 여기서는 잠재 요인 차원을 30으로 설정합니다.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

아래에서 저희는 훈련된 모델을 사용하여 사용자(ID 20)가 아이템(ID 30)에 부여할 수 있는 평점을 예측합니다.

```{.python .input  n=6}
#@tab mxnet
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## 요약

* 행렬 분해 모델은 추천 시스템에서 널리 사용됩니다. 사용자가 아이템에 부여할 수 있는 평점을 예측하는 데 사용될 수 있습니다.
* 저희는 추천 시스템을 위한 행렬 분해를 구현하고 훈련할 수 있습니다.


## 연습문제

* 잠재 요인의 크기를 변화시켜 보십시오. 잠재 요인의 크기는 모델 성능에 어떻게 영향을 미칩니까?
* 서로 다른 최적화기, 학습률, 가중치 감쇠율을 시도해 보십시오.
* 특정 영화에 대한 다른 사용자의 예측 평점 점수를 확인해 보십시오.


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/400)
:end_tab:
