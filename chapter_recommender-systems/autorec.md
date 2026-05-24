# AutoRec: 오토인코더를 사용한 평점 예측

비록 행렬 분해 모델이 평점 예측 작업에서 적절한 성능을 달성하지만, 그것은 본질적으로 선형 모델입니다. 따라서 이러한 모델은 사용자의 선호도를 예측할 수 있는 복잡한 비선형적이고 정교한 관계를 포착할 수 없습니다. 이 절에서는 비선형 신경망 협업 필터링 모델인 AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`을 소개합니다. 이 모델은 협업 필터링(CF)을 오토인코더 구조로 식별하고 명시적 피드백을 기반으로 CF에 비선형 변환을 통합하는 것을 목표로 합니다. 신경망은 어떤 연속 함수도 근사할 수 있음이 입증되었으며, 이는 행렬 분해의 한계를 해결하고 행렬 분해의 표현력을 풍부하게 하는 데 적합하게 만듭니다.

한편으로, AutoRec은 입력 계층, 은닉 계층, 그리고 재구성(출력) 계층으로 구성된 오토인코더와 동일한 구조를 갖습니다. 오토인코더는 입력을 은닉(보통 저차원) 표현으로 인코딩하기 위해 자신의 입력을 출력으로 복사하는 것을 학습하는 신경망입니다. AutoRec에서는 사용자/아이템을 저차원 공간에 명시적으로 임베딩하는 대신, 상호작용 행렬의 열/행을 입력으로 사용하고 출력 계층에서 상호작용 행렬을 재구성합니다.

다른 한편으로, AutoRec은 전통적인 오토인코더와 다릅니다. 은닉 표현을 학습하는 대신, AutoRec은 출력 계층을 학습/재구성하는 데 중점을 둡니다. 부분적으로 관찰된 상호작용 행렬을 입력으로 사용하여 완성된 평점 행렬을 재구성하는 것을 목표로 합니다. 그동안 입력의 누락된 항목들은 추천을 목적으로 재구성을 통해 출력 계층에 채워집니다.

AutoRec에는 사용자 기반과 아이템 기반의 두 가지 변형이 있습니다. 간결성을 위해 여기서는 아이템 기반 AutoRec만 소개합니다. 사용자 기반 AutoRec은 그에 따라 유도될 수 있습니다.


## 모델

$\mathbf{R}_{*i}$가 평점 행렬의 $i^\textrm{th}$ 열을 나타내며, 알 수 없는 평점은 기본적으로 0으로 설정된다고 합시다. 신경망 구조는 다음과 같이 정의됩니다.

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

여기서 $f(\cdot)$와 $g(\cdot)$는 활성화 함수를 나타내고, $\mathbf{W}$와 $\mathbf{V}$는 가중치 행렬이며, $\mu$와 $b$는 편향입니다. $h( \cdot )$가 AutoRec의 전체 네트워크를 나타낸다고 합시다. 출력 $h(\mathbf{R}_{*i})$는 평점 행렬의 $i^\textrm{th}$ 열의 재구성입니다.

다음 목적 함수는 재구성 오차를 최소화하는 것을 목표로 합니다.

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

여기서 $\| \cdot \|_{\mathcal{O}}$는 관찰된 평점의 기여만 고려된다는 것을 의미합니다. 즉, 역전파 동안 관찰된 입력과 연관된 가중치만 업데이트됩니다.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## 모델 구현

전형적인 오토인코더는 인코더와 디코더로 구성됩니다. 인코더는 입력을 은닉 표현으로 투영하고 디코더는 은닉 계층을 재구성 계층으로 매핑합니다. 저희는 이 관행을 따르고 완전 연결 계층으로 인코더와 디코더를 생성합니다. 인코더의 활성화는 기본적으로 `sigmoid`로 설정되고 디코더에는 활성화가 적용되지 않습니다. 과적합을 줄이기 위해 인코딩 변환 후에 드롭아웃이 포함됩니다. 관찰되지 않은 입력의 그래디언트는 마스킹되어 관찰된 평점만 모델 학습 과정에 기여하도록 보장합니다.

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 평가자 재구현

입력과 출력이 변경되었으므로, 저희는 여전히 RMSE를 정확도 측도로 사용하면서도 평가 함수를 재구현해야 합니다.

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## 모델 훈련 및 평가

이제 MovieLens 데이터셋에서 AutoRec을 훈련하고 평가합시다. 테스트 RMSE가 행렬 분해 모델보다 낮은 것을 확인할 수 있으며, 이는 평점 예측 작업에서 신경망의 효과성을 입증합니다.

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## 요약

* 저희는 비선형 계층과 드롭아웃 정규화를 통합하면서 행렬 분해 알고리즘을 오토인코더로 구성할 수 있습니다.
* MovieLens 100K 데이터셋에서의 실험은 AutoRec이 행렬 분해보다 우수한 성능을 달성한다는 것을 보여줍니다.



## 연습문제

* AutoRec의 은닉 차원을 변화시켜 모델 성능에 미치는 영향을 살펴보십시오.
* 더 많은 은닉 계층을 추가해 보십시오. 모델 성능을 향상시키는 데 도움이 됩니까?
* 디코더와 인코더 활성화 함수의 더 나은 조합을 찾을 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/401)
:end_tab:
