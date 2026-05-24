# 추천 시스템을 위한 개인화 랭킹

이전 절에서는 명시적 피드백만 고려되었고 모델은 관찰된 평점에 대해 훈련되고 테스트되었습니다. 이러한 방법에는 두 가지 단점이 있습니다. 첫째, 실세계 시나리오에서 대부분의 피드백은 명시적이지 않고 암묵적이며, 명시적 피드백은 수집하는 데 더 많은 비용이 들 수 있습니다. 둘째, 사용자의 관심사를 예측할 수 있는 관찰되지 않은 사용자-아이템 쌍은 완전히 무시되어, 이러한 방법들이 평점이 무작위로 누락되는 것이 아니라 사용자의 선호도 때문에 누락되는 경우에는 적합하지 않게 만듭니다. 관찰되지 않은 사용자-아이템 쌍은 실제 부정적 피드백(사용자가 아이템에 관심이 없음)과 누락된 값(사용자가 미래에 아이템과 상호작용할 수 있음)의 혼합입니다. 저희는 행렬 분해와 AutoRec에서 관찰되지 않은 쌍을 단순히 무시합니다. 분명히 이러한 모델들은 관찰된 쌍과 관찰되지 않은 쌍을 구별할 수 없으며 보통 개인화 랭킹 작업에 적합하지 않습니다.

이를 위해, 암묵적 피드백으로부터 랭킹된 추천 목록을 생성하는 것을 목표로 하는 일군의 추천 모델들이 인기를 얻고 있습니다. 일반적으로 개인화 랭킹 모델은 포인트와이즈(pointwise), 페어와이즈(pairwise), 또는 리스트와이즈(listwise) 접근법으로 최적화될 수 있습니다. 포인트와이즈 접근법은 한 번에 하나의 상호작용을 고려하고 개별 선호도를 예측하기 위해 분류기나 회귀기를 훈련합니다. 행렬 분해와 AutoRec은 포인트와이즈 목적으로 최적화됩니다. 페어와이즈 접근법은 각 사용자에 대한 한 쌍의 아이템을 고려하고 그 쌍에 대한 최적의 순서를 근사하는 것을 목표로 합니다. 일반적으로 페어와이즈 접근법은 상대적 순서를 예측하는 것이 랭킹의 본질을 연상시키므로 랭킹 작업에 더 적합합니다. 리스트와이즈 접근법은 전체 아이템 리스트의 순서를 근사하는데, 예를 들어 정규화된 할인 누적 이득(Normalized Discounted Cumulative Gain, [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain))와 같은 랭킹 측도를 직접 최적화합니다. 그러나 리스트와이즈 접근법은 포인트와이즈나 페어와이즈 접근법보다 더 복잡하고 계산 집약적입니다. 이 절에서는 두 가지 페어와이즈 목적/손실인 베이지안 개인화 랭킹 손실과 힌지 손실, 그리고 각각의 구현을 소개합니다.

## 베이지안 개인화 랭킹 손실과 그 구현

베이지안 개인화 랭킹(Bayesian Personalized Ranking, BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009`은 최대 사후 추정자에서 유도된 페어와이즈 개인화 랭킹 손실입니다. 이는 많은 기존 추천 모델에서 널리 사용되어 왔습니다. BPR의 훈련 데이터는 긍정적 쌍과 부정적 쌍(누락된 값)으로 구성됩니다. 사용자가 다른 모든 관찰되지 않은 아이템보다 긍정적 아이템을 선호한다고 가정합니다.

공식적으로 훈련 데이터는 사용자 $u$가 아이템 $j$보다 아이템 $i$를 선호한다는 것을 나타내는 $(u, i, j)$ 형태의 튜플로 구성됩니다. 사후 확률을 최대화하는 것을 목표로 하는 BPR의 베이지안 공식화는 아래와 같습니다.

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

여기서 $\Theta$는 임의의 추천 모델의 파라미터를 나타내며, $>_u$는 사용자 $u$에 대한 모든 아이템의 원하는 개인화된 전체 랭킹을 나타냅니다. 저희는 개인화 랭킹 작업에 대한 일반적인 최적화 기준을 유도하기 위해 최대 사후 추정자를 공식화할 수 있습니다.

$$
\begin{aligned}
\textrm{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$


여기서 $D \stackrel{\textrm{def}}{=} \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$는 훈련 셋이며, $I^+_u$는 사용자 $u$가 좋아한 아이템을, $I$는 모든 아이템을, $I \backslash I^+_u$는 사용자가 좋아한 아이템을 제외한 다른 모든 아이템을 나타냅니다. $\hat{y}_{ui}$와 $\hat{y}_{uj}$는 각각 사용자 $u$가 아이템 $i$와 $j$에 대해 예측한 점수입니다. 사전 분포 $p(\Theta)$는 평균이 0이고 분산-공분산 행렬이 $\Sigma_\Theta$인 정규분포입니다. 여기서 저희는 $\Sigma_\Theta = \lambda_\Theta I$로 둡니다.

![베이지안 개인화 랭킹 예시](../img/rec-ranking.svg)
저희는 기본 클래스 `mxnet.gluon.loss.Loss`를 구현하고 `forward` 메서드를 오버라이드하여 베이지안 개인화 랭킹 손실을 구성할 것입니다. Loss 클래스와 np 모듈을 임포트하는 것으로 시작합니다.

```{.python .input  n=5}
#@tab mxnet
from mxnet import gluon, np, npx
npx.set_np()
```

BPR 손실의 구현은 다음과 같습니다.

```{.python .input  n=2}
#@tab mxnet
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## 힌지 손실과 그 구현

랭킹용 힌지 손실은 SVM과 같은 분류기에서 자주 사용되는 gluon 라이브러리 내에서 제공되는 [힌지 손실](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss)과 다른 형태를 갖습니다. 추천 시스템에서 랭킹에 사용되는 손실은 다음과 같은 형태를 가집니다.

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

여기서 $m$은 안전 마진 크기입니다. 이 손실은 부정적 아이템을 긍정적 아이템으로부터 밀어내는 것을 목표로 합니다. BPR과 유사하게, 절대적인 출력 대신 긍정 샘플과 부정 샘플 간의 관련 거리를 최적화하는 것을 목표로 하여 추천 시스템에 잘 적합합니다.

```{.python .input  n=3}
#@tab mxnet
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

이 두 손실은 추천에서의 개인화 랭킹에 서로 교환 가능합니다.

## 요약

- 추천 시스템에서 개인화 랭킹 작업에 사용할 수 있는 랭킹 손실에는 세 가지 유형이 있습니다, 즉 포인트와이즈, 페어와이즈, 리스트와이즈 방법입니다.
- 두 가지 페어와이즈 손실인 베이지안 개인화 랭킹 손실과 힌지 손실은 서로 교환하여 사용할 수 있습니다.

## 연습문제

- BPR과 힌지 손실의 변형이 있습니까?
- BPR이나 힌지 손실을 사용하는 추천 모델을 찾을 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/402)
:end_tab:
