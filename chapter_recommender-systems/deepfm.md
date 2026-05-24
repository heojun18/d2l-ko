# 딥 인수분해 머신

효과적인 특징 조합을 학습하는 것은 클릭률 예측 작업의 성공에 매우 중요합니다. 인수분해 머신은 선형 패러다임(예: 쌍선형 상호작용)으로 특징 상호작용을 모델링합니다. 이는 실세계 데이터에서 본질적인 특징 교차 구조가 보통 매우 복잡하고 비선형이기 때문에 종종 불충분합니다. 더 나쁜 것은, 실제로 인수분해 머신에서는 일반적으로 2차 특징 상호작용이 사용된다는 점입니다. 인수분해 머신으로 더 높은 차수의 특징 조합을 모델링하는 것은 이론적으로 가능하지만, 수치적 불안정성과 높은 계산 복잡도로 인해 보통 채택되지 않습니다.

한 가지 효과적인 해결책은 심층 신경망을 사용하는 것입니다. 심층 신경망은 특징 표현 학습에 강력하며 정교한 특징 상호작용을 학습할 잠재력이 있습니다. 따라서 심층 신경망을 인수분해 머신에 통합하는 것은 자연스럽습니다. 인수분해 머신에 비선형 변환 계층을 추가하면 저차 특징 조합과 고차 특징 조합을 모두 모델링할 수 있는 능력이 부여됩니다. 또한, 입력에서 비선형 본질적 구조도 심층 신경망으로 포착될 수 있습니다. 이 절에서는 FM과 심층 신경망을 결합한 대표적 모델인 딥 인수분해 머신(deep factorization machines, DeepFM) :cite:`Guo.Tang.Ye.ea.2017`를 소개합니다.


## 모델 구조

DeepFM은 병렬 구조로 통합된 FM 구성 요소와 딥 구성 요소로 구성됩니다. FM 구성 요소는 저차 특징 상호작용을 모델링하는 데 사용되는 2차 인수분해 머신과 동일합니다. 딥 구성 요소는 고차 특징 상호작용과 비선형성을 포착하는 데 사용되는 MLP입니다. 이 두 구성 요소는 동일한 입력/임베딩을 공유하며 그 출력은 최종 예측으로 합산됩니다. DeepFM의 정신은 암기와 일반화를 모두 포착할 수 있는 Wide \& Deep 구조의 정신과 유사하다는 점을 지적할 가치가 있습니다. Wide \& Deep 모델에 비한 DeepFM의 장점은 특징 조합을 자동으로 식별함으로써 수작업 특징 공학의 노력을 줄인다는 것입니다.

저희는 간결성을 위해 FM 구성 요소의 설명을 생략하고 출력을 $\hat{y}^{(FM)}$로 표기합니다. 독자는 자세한 내용을 위해 지난 절을 참조하십시오. $\mathbf{e}_i \in \mathbb{R}^{k}$가 $i^\textrm{th}$ 필드의 잠재 특징 벡터를 나타낸다고 합시다. 딥 구성 요소의 입력은 희소 범주형 특징 입력으로 조회된 모든 필드의 밀집 임베딩의 연결이며, 다음과 같이 표기됩니다.

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

여기서 $f$는 필드의 수입니다. 그런 다음 다음 신경망에 공급됩니다.

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

여기서 $\alpha$는 활성화 함수입니다. $\mathbf{W}_{l}$과 $\mathbf{b}_{l}$은 $l^\textrm{th}$ 계층의 가중치와 편향입니다. $y_{DNN}$이 예측의 출력을 나타낸다고 합시다. DeepFM의 궁극적인 예측은 FM과 DNN 양쪽의 출력의 합입니다. 따라서 저희는 다음을 얻습니다.

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

여기서 $\sigma$는 시그모이드 함수입니다. DeepFM의 구조는 아래에 그림으로 표시되어 있습니다.
![DeepFM 모델 예시](../img/rec-deepfm.svg)

DeepFM이 심층 신경망과 FM을 결합하는 유일한 방법은 아니라는 점에 유의할 가치가 있습니다. 저희는 또한 특징 상호작용 위에 비선형 계층을 추가할 수도 있습니다 :cite:`He.Chua.2017`.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## DeepFM의 구현
DeepFM의 구현은 FM의 구현과 유사합니다. 저희는 FM 부분을 변경하지 않고 활성화 함수로 `relu`를 사용하는 MLP 블록을 사용합니다. 모델을 정규화하기 위해 드롭아웃도 사용됩니다. MLP의 뉴런 수는 `mlp_dims` 하이퍼파라미터로 조정될 수 있습니다.

```{.python .input  n=2}
#@tab mxnet
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## 모델 훈련 및 평가
데이터 로딩 프로세스는 FM의 것과 동일합니다. 저희는 DeepFM의 MLP 구성 요소를 피라미드 구조(30-20-10)를 가진 3계층 밀집 네트워크로 설정합니다. 다른 모든 하이퍼파라미터는 FM과 동일하게 유지됩니다.

```{.python .input  n=4}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

FM과 비교하여, DeepFM은 더 빠르게 수렴하고 더 나은 성능을 달성합니다.

## 요약

* 신경망을 FM에 통합하면 복잡하고 고차 상호작용을 모델링할 수 있습니다.
* DeepFM은 광고 데이터셋에서 원본 FM을 능가합니다.

## 연습문제

* MLP의 구조를 변경하여 모델 성능에 미치는 영향을 확인하십시오.
* 데이터셋을 Criteo로 변경하고 원본 FM 모델과 비교하십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/407)
:end_tab:
