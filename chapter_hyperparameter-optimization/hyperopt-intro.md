```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# 하이퍼파라미터 최적화란 무엇인가?
:label:`sec_what_is_hpo`

이전 장들에서 보았듯이, 심층 신경망은 학습 과정에서 학습되는
다수의 파라미터(혹은 가중치)를 가지고 있습니다. 이뿐만 아니라,
모든 신경망에는 사용자가 별도로 설정해야 하는 추가적인
*하이퍼파라미터*가 존재합니다. 예를 들어, 확률적 경사 하강법이
학습 손실의 국소 최적점으로 수렴하도록 하기 위해서
(:numref:`chap_optimization` 참고), 학습률과 배치 크기를 조정해야
합니다. 학습 데이터셋에 대한 과적합을 피하기 위해서는, 가중치
감쇠(:numref:`sec_weight_decay` 참고)나 드롭아웃(:numref:`sec_dropout`
참고)과 같은 정규화 파라미터를 설정해야 할 수도 있습니다. 또한,
층의 수와 층마다의 유닛 수 혹은 필터 수(즉, 가중치의 실효 개수)를
지정함으로써 모델의 용량과 귀납적 편향을 정의할 수 있습니다.

안타깝게도, 학습 손실을 최소화하는 방식으로 이 하이퍼파라미터들을
단순히 조정할 수는 없습니다. 그렇게 하면 학습 데이터에 대한 과적합으로
이어지기 때문입니다. 예를 들어, 드롭아웃이나 가중치 감쇠와 같은
정규화 파라미터를 0으로 설정하면 학습 손실은 작아지지만, 일반화 성능에
악영향을 줄 수 있습니다.

![서로 다른 하이퍼파라미터로 모델을 여러 번 학습시키는 과정으로 이루어진 머신러닝의 전형적인 워크플로.](../img/ml_workflow.svg)
:label:`ml_workflow`

자동화의 다른 형태가 없다면, 하이퍼파라미터는 수작업으로
시행착오를 거치며 설정해야 하는데, 이는 머신러닝 워크플로에서
시간이 많이 들고 어려운 부분에 해당합니다. 예를 들어, Amazon
Elastic Cloud Compute (EC2) `g4dn.xlarge` 인스턴스에서 CIFAR-10에
대해 ResNet(:numref:`sec_resnet` 참고)을 학습시키려면 2시간 이상이
걸립니다. 단지 10개의 하이퍼파라미터 구성을 순차적으로 시도해
보더라도 이미 하루 정도가 소요됩니다. 더 큰 문제는, 하이퍼파라미터가
보통 아키텍처나 데이터셋 간에 직접적으로 이전될 수 없으며
:cite:`feurer-arxiv22,wistuba-ml18,bardenet-icml13a`, 새로운
태스크마다 다시 최적화해야 한다는 점입니다. 또한 대부분의
하이퍼파라미터에는 경험 법칙이 없어, 합리적인 값을 찾기 위해서는
전문가의 지식이 필요합니다.

*하이퍼파라미터 최적화(hyperparameter optimization, HPO)* 알고리즘은
이 문제를 전역 최적화 문제로 정식화함으로써, 원칙적이고 자동화된
방식으로 해결하도록 설계되었습니다 :cite:`feurer-automlbook18a`. 기본
목적 함수는 보류된(hold-out) 검증 데이터셋에 대한 오차이지만,
원칙적으로는 다른 어떤 비즈니스 지표가 될 수도 있습니다. 학습 시간,
추론 시간, 모델 복잡도와 같은 부차적인 목적과 결합되거나 그에 의해
제약을 받을 수도 있습니다.

최근에는 하이퍼파라미터 최적화가 *신경망 아키텍처 탐색(neural
architecture search, NAS)* :cite:`elsken-arxiv18a,wistuba-arxiv19`으로
확장되었으며, 그 목표는 완전히 새로운 신경망 아키텍처를 찾는
것입니다. 고전적인 HPO와 비교하면, NAS는 계산 측면에서 더욱 비싸며,
실용적으로 실행 가능하기 위해서는 추가적인 노력이 필요합니다. HPO와
NAS는 모두 전체 ML 파이프라인을 자동화하는 것을 목표로 하는 AutoML
:cite:`hutter-book19a`의 하위 분야로 볼 수 있습니다.

이 절에서는 HPO를 소개하고, :numref:`sec_softmax_concise`에서 소개된
로지스틱 회귀 예제의 최적 하이퍼파라미터를 자동으로 찾는 방법을
보여줍니다.

##  최적화 문제
:label:`sec_definition_hpo`

저희는 간단한 토이 문제로 시작합니다. :numref:`sec_softmax_concise`의
다중 클래스 로지스틱 회귀 모델 `SoftmaxRegression`에 대해 Fashion
MNIST 데이터셋에서 검증 오차를 최소화하는 학습률을 탐색하는
문제입니다. 배치 크기나 에폭 수와 같은 다른 하이퍼파라미터들도
조정할 만한 가치가 있지만, 단순화를 위해 학습률 하나에만 집중합니다.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from scipy import stats
```

HPO를 실행하기 전에, 먼저 두 가지 요소를 정의해야 합니다. 목적 함수와
구성 공간입니다.

### 목적 함수

학습 알고리즘의 성능은 하이퍼파라미터 공간
$\mathbf{x} \in \mathcal{X}$에서 검증 손실로 매핑되는 함수
$f: \mathcal{X} \rightarrow \mathbb{R}$로 볼 수 있습니다. $f(\mathbf{x})$의
매 평가마다, 저희는 머신러닝 모델을 학습하고 검증해야
하는데, 큰 데이터셋에서 학습되는 심층 신경망의 경우 이는 시간과
계산 자원이 많이 드는 작업일 수 있습니다. 기준 $f(\mathbf{x})$가
주어졌을 때, 저희의 목표는
$\mathbf{x}_{\star} \in \mathrm{argmin}_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$를
찾는 것입니다.

$f$의 $\mathbf{x}$에 대한 그래디언트를 계산하는 간단한 방법은 없는데,
이는 전체 학습 과정을 통해 그래디언트를 전파해야 하기 때문입니다.
근사 "하이퍼그래디언트(hypergradients)"를 통해 HPO를 수행하는 최근의
연구 :cite:`maclaurin-icml15,franceschi-icml17a`도 있지만, 기존의
어떤 접근도 아직 최신 기법들과 경쟁할 만한 수준은 아니어서 여기서는
다루지 않습니다. 더욱이, $f$의 산정에 따른 계산 부담 때문에 HPO 알고리즘은
가능한 한 적은 표본으로 전역 최적해에 접근해야 합니다.

신경망의 학습은 확률적이므로 (예: 가중치가 무작위로 초기화되고,
미니배치가 무작위로 표집됨) 저희의 관측값은 노이즈를 포함합니다.
즉, $y \sim f(\mathbf{x}) + \epsilon$이며, 여기서 일반적으로 관측
노이즈 $\epsilon \sim N(0, \sigma)$가 가우시안 분포를 따른다고
가정합니다.

이러한 모든 어려움 때문에, 저희는 보통 전역 최적해를 정확히 맞히기보다
좋은 성능을 보이는 소수의 하이퍼파라미터 구성을 빠르게 찾고자
시도합니다. 그러나 대부분의 신경망 모델은 막대한 계산을 요구하기
때문에, 이마저도 며칠 혹은 몇 주의 계산이 걸릴 수 있습니다.
:numref:`sec_mf_hpo`에서는 탐색을 분산시키거나 목적 함수의 산정 비용이
저렴한 근사를 사용하여 최적화 과정을 가속화하는 방법을 살펴봅니다.

먼저 모델의 검증 오차를 계산하는 방법부터 시작합니다.

```{.python .input  n=8}
%%tab pytorch
class HPOTrainer(d2l.Trainer):  #@save
    def validation_error(self):
        self.model.eval()
        accuracy = 0
        val_batch_idx = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                x, y = self.prepare_batch(batch)
                y_hat = self.model(x)
                accuracy += self.model.accuracy(y_hat, y)
            val_batch_idx += 1
        return 1 -  accuracy / val_batch_idx
```

저희는 `learning_rate`로 구성된 하이퍼파라미터 구성 `config`에 대해
검증 오차를 최적화합니다. 매 산정마다 모델을 `max_epochs` 에폭만큼
학습시킨 후, 그 검증 오차를 계산하여 반환합니다.

```{.python .input  n=5}
%%tab pytorch
def hpo_objective_softmax_classification(config, max_epochs=8):
    learning_rate = config["learning_rate"]
    trainer = d2l.HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=16)
    model = d2l.SoftmaxRegression(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    return d2l.numpy(trainer.validation_error())
```

### 구성 공간
:label:`sec_intro_config_spaces`

목적 함수 $f(\mathbf{x})$와 더불어, 최적화의 대상이 되는 가능한 집합
$\mathbf{x} \in \mathcal{X}$도 정의해야 하며, 이를 *구성 공간(configuration
space)* 혹은 *탐색 공간(search space)*이라고 합니다. 로지스틱 회귀
예제에서는 다음을 사용합니다.

```{.python .input  n=6}
config_space = {"learning_rate": stats.loguniform(1e-4, 1)}
```

여기서는 SciPy의 `loguniform` 객체를 사용했는데, 이는 로그 공간에서
-4와 -1 사이의 균등 분포를 나타냅니다. 이 객체를 사용해 이 분포에서
랜덤 변수를 표집할 수 있습니다.

각 하이퍼파라미터는 `learning_rate`의 `float`과 같은 데이터 타입을
가지며, 닫힌 유계 범위(즉, 하한과 상한)도 가집니다. 보통은 표집을
위해 각 하이퍼파라미터에 사전 분포(예: 균등 분포 또는 로그 균등
분포)를 부여합니다. `learning_rate`와 같은 일부 양의 파라미터는
최적값이 수 자리수 차이가 날 수 있기 때문에 로그 스케일로 표현하는
것이 가장 좋고, 모멘텀과 같은 다른 파라미터들은 선형 스케일이
적합합니다.

아래에는 다층 퍼셉트론의 전형적인 하이퍼파라미터들로 구성된 간단한
구성 공간 예시를 그 타입 및 표준 범위와 함께 보여드립니다.

: 다층 퍼셉트론의 구성 공간 예시
:label:`tab_example_configspace`

| 이름                 | 타입         | 하이퍼파라미터 범위              | 로그 스케일 |
| :----:              | :----:      |:------------------------------:|:---------:|
| 학습률                | float       |      $[10^{-6},10^{-1}]$       |    예      |
| 배치 크기              | integer     |           $[8,256]$            |    예      |
| 모멘텀                | float       |           $[0,0.99]$           |    아니오   |
| 활성화 함수             | categorical | $\{\textrm{tanh}, \textrm{relu}\}$ |     -     |
| 유닛 수                | integer     |          $[32, 1024]$          |    예      |
| 층 수                 | integer     |            $[1, 6]$            |    아니오   |



일반적으로 구성 공간 $\mathcal{X}$의 구조는 복잡할 수 있으며,
$\mathbb{R}^d$와는 상당히 다를 수 있습니다. 실제로는 일부
하이퍼파라미터가 다른 하이퍼파라미터의 값에 의존할 수도 있습니다.
예를 들어, 다층 퍼셉트론의 층 수와 각 층의 유닛 수를 조정하려 한다고
가정해 보겠습니다. $l\textrm{-번째}$ 층의 유닛 수는 신경망의 층이
적어도 $l+1$개 있을 때만 의미가 있습니다. 이러한 고급 HPO 문제는
이 장의 범위를 벗어납니다. 관심 있는 독자들께는
:cite:`hutter-lion11a,jenatton-icml17a,baptista-icml18a`를 참고하시기를
권합니다.

구성 공간은 하이퍼파라미터 최적화에 있어 중요한 역할을 합니다.
어떤 알고리즘도 구성 공간에 포함되지 않은 것을 찾아낼 수는 없기
때문입니다. 반면에, 범위가 너무 크면 좋은 성능의 구성을 찾기 위한
계산 예산이 실행 불가능한 수준이 될 수 있습니다.

## 랜덤 탐색
:label:`sec_rs`

*랜덤 탐색(random search)*은 저희가 살펴볼 첫 번째 하이퍼파라미터
최적화 알고리즘입니다. 랜덤 탐색의 핵심 아이디어는 미리 정해진
예산(예: 최대 반복 횟수)이 소진될 때까지 구성 공간에서 독립적으로
표집하고, 관측된 것 중 가장 좋은 구성을 반환하는 것입니다. 모든
산정은 독립적으로 병렬 실행이 가능하지만(:numref:`sec_rs_async` 참고),
여기서는 단순화를 위해 순차 루프를 사용합니다.

```{.python .input  n=7}
errors, values = [], []
num_iterations = 5

for i in range(num_iterations):
    learning_rate = config_space["learning_rate"].rvs()
    print(f"Trial {i}: learning_rate = {learning_rate}")
    y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
    print(f"    validation_error = {y}")
    values.append(learning_rate)
    errors.append(y)
```

최적의 학습률은 단순히 검증 오차가 가장 낮은 학습률입니다.

```{.python .input  n=7}
best_idx = np.argmin(errors)
print(f"optimal learning rate = {values[best_idx]}")
```

단순성과 일반성 덕분에, 랜덤 탐색은 가장 자주 사용되는 HPO 알고리즘
중 하나입니다. 정교한 구현이 필요하지 않으며, 각 하이퍼파라미터에
대해 어떤 확률 분포만 정의할 수 있다면 어떤 구성 공간에도 적용할
수 있습니다.

안타깝게도 랜덤 탐색에도 몇 가지 단점이 있습니다. 첫째, 지금까지
수집한 이전 관측에 기반하여 표집 분포를 적응시키지 않습니다. 따라서
성능이 나쁜 구성을 표집할 가능성이 성능이 좋은 구성을 표집할
가능성과 같습니다. 둘째, 일부는 초기 성능이 나쁘고 이전에 본
구성들을 능가할 가능성이 적더라도, 모든 구성에 동일한 양의 자원이
소비됩니다.

다음 절들에서는, 탐색을 안내하는 모델을 사용하여 랜덤 탐색의 단점을
극복하는, 표본 효율이 더 높은 하이퍼파라미터 최적화 알고리즘들을
살펴봅니다. 또한 성능이 나쁜 구성의 산정 과정을 자동으로 중단하여
최적화 과정을 가속화하는 알고리즘들도 살펴봅니다.

## 요약

이 절에서는 하이퍼파라미터 최적화(HPO)를 소개하고, 구성 공간과 목적
함수를 정의함으로써 어떻게 이를 전역 최적화 문제로 정식화할 수 있는지
보였습니다. 또한 첫 번째 HPO 알고리즘인 랜덤 탐색을 구현하여, 간단한
소프트맥스 분류 문제에 적용했습니다.

랜덤 탐색은 매우 단순하지만, 단순히 고정된 하이퍼파라미터 집합을
산정하는 그리드 탐색에 대한 더 나은 대안입니다. 랜덤 탐색은 차원의
저주 :cite:`bellman-science66`를 어느 정도 완화하며, 기준이 주로
하이퍼파라미터의 작은 부분집합에 강하게 의존하는 경우 그리드 탐색보다
훨씬 효율적일 수 있습니다.

## 연습 문제

1. 이 장에서는 분리된 학습 셋으로 학습한 후 모델의 검증 오차를 최적화합니다. 단순화를 위해, 저희 코드는 `Trainer.val_dataloader`를 사용하며, 이는 `FashionMNIST.val` 주변의 로더에 매핑됩니다.
    1. (코드를 살펴보면서) 이것이 의미하는 바, 즉 저희가 학습에는 원래의 FashionMNIST 학습 셋(예제 60000개)을, 검증에는 원래의 *테스트 셋*(예제 10000개)을 사용한다는 점을 스스로 확인해 보세요.
    2. 왜 이러한 관행이 문제가 될 수 있을까요? 힌트: 특히 *모델 선택*에 관한 부분을 중심으로 :numref:`sec_generalization_basics`를 다시 읽어 보세요.
    3. 대신 어떻게 해야 했을까요?
2. 위에서 경사 하강법에 의한 하이퍼파라미터 최적화는 매우 수행하기 어렵다고 말씀드렸습니다. FashionMNIST 데이터셋(:numref:`sec_mlp-implementation`)에서 배치 크기 256으로 2층 퍼셉트론을 학습하는 것과 같은 작은 문제를 고려해 보겠습니다. 1 에폭 학습 후 검증 지표를 최소화하기 위해 SGD의 학습률을 조정하고자 합니다.
    1. 이 목적으로 검증 *오차*를 사용할 수 없는 이유는 무엇일까요? 검증 셋에서 어떤 지표를 사용하시겠습니까?
    2. 1 에폭 학습 후 검증 지표의 계산 그래프를 (대략) 스케치해 보세요. 초기 가중치와 (학습률과 같은) 하이퍼파라미터가 이 그래프의 입력 노드라고 가정해도 됩니다. 힌트: :numref:`sec_backprop`에서 계산 그래프에 대한 내용을 다시 읽어 보세요.
    3. 이 그래프에서 순방향 패스 동안 저장해야 하는 부동소수점 값의 수를 대략적으로 추정해 보세요. 힌트: FashionMNIST에는 60000개의 케이스가 있습니다. 필요한 메모리가 각 층 이후의 활성값에 의해 지배된다고 가정하고, 층 너비는 :numref:`sec_mlp-implementation`에서 찾아 보세요.
    5. 필요한 계산량과 저장 공간의 규모를 떠나, 그래디언트 기반 하이퍼파라미터 최적화는 어떤 다른 문제를 겪게 될까요? 힌트: :numref:`sec_numerical_stability`에서 그래디언트 소실 및 폭발에 관한 내용을 다시 읽어 보세요.
    6. *심화*: 그래디언트 기반 HPO에 대한 우아한(그러나 여전히 다소 비실용적인) 접근법은 :cite:`maclaurin-icml15`를 읽어 보세요.
3. 그리드 탐색은 또 다른 HPO 베이스라인으로, 각 하이퍼파라미터에 대해 등간격의 그리드를 정의한 후, 구성을 제안하기 위해 (조합적) 데카르트 곱을 반복합니다.
    1. 위에서 기준이 주로 하이퍼파라미터의 작은 부분집합에 강하게 의존하는 경우, 상당한 수의 하이퍼파라미터에 대한 HPO에서는 랜덤 탐색이 그리드 탐색보다 훨씬 효율적일 수 있다고 말씀드렸습니다. 왜 그럴까요? 힌트: :cite:`bergstra2011algorithms`를 읽어 보세요.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12090)
:end_tab:
