```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# 다중 충실도 하이퍼파라미터 최적화
:label:`sec_mf_hpo`

신경망 학습은 중간 크기의 데이터셋에서도 비용이 클 수 있습니다.
구성 공간(:numref:`sec_intro_config_spaces`)에 따라, 하이퍼파라미터
최적화는 좋은 성능의 하이퍼파라미터 구성을 찾기 위해 수십에서 수백
번의 함수 산정이 필요합니다. :numref:`sec_rs_async`에서 보았듯이,
병렬 자원을 활용하여 HPO의 전체 벽시계 시간을 크게 단축할 수 있지만,
이것이 필요한 총 계산량을 줄여 주지는 않습니다.

이 절에서는 하이퍼파라미터 구성의 산정을 어떻게 가속할 수 있는지
보여드립니다. 랜덤 탐색과 같은 방법들은 각 하이퍼파라미터 산정에
동일한 양의 자원(예: 에폭 수, 학습 데이터 점의 수)을 할당합니다.
:numref:`img_samples_lc`는 서로 다른 하이퍼파라미터 구성으로 학습된
신경망 집합의 학습 곡선을 보여줍니다. 몇 에폭이 지나면, 저희는 이미
시각적으로 좋은 성능과 차선의 구성을 구분할 수 있습니다. 그러나 학습
곡선은 노이즈가 있으므로, 가장 성능이 좋은 구성을 식별하기 위해서는
여전히 100 에폭 전부가 필요할 수도 있습니다.

![무작위 하이퍼파라미터 구성의 학습 곡선](../img/samples_lc.svg)
:label:`img_samples_lc`

다중 충실도 하이퍼파라미터 최적화는 유망한 구성에 더 많은 자원을
할당하고, 성능이 나쁜 구성의 산정은 일찍 중단합니다. 동일한 총
자원량으로 더 많은 수의 구성을 시도할 수 있기 때문에, 이것은
최적화 과정을 가속화합니다.

더 형식적으로는, :numref:`sec_definition_hpo`의 정의를 확장하여,
목적 함수 $f(\mathbf{x}, r)$이 추가 입력
$r \in [r_{\mathrm{min}}, r_{max}]$를 받도록 합니다. 이 입력은 구성
$\mathbf{x}$의 산정에 저희가 기꺼이 쓸 자원의 양을 지정합니다.
오차 $f(\mathbf{x}, r)$은 $r$에 따라 감소하고, 계산 비용
$c(\mathbf{x}, r)$은 증가한다고 가정합니다. 일반적으로 $r$은 신경망을
학습시키는 에폭 수를 나타내지만, 학습 부분집합의 크기나 교차 검증
폴드의 수가 될 수도 있습니다.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
from scipy import stats
from collections import defaultdict
d2l.set_figsize()
```

## 연속 하빙(Successive Halving)
:label:`sec_mf_hpo_sh`

랜덤 탐색을 다중 충실도 설정에 적응시키는 가장 간단한 방법 중 하나는
*연속 하빙(successive halving)* :cite:`jamieson-aistats16,karnin-icml13`입니다.
기본 아이디어는, 예를 들어 구성 공간에서 무작위로 표집된 $N$개의 구성으로
시작하여, 각각을 $r_{\mathrm{min}}$ 에폭만큼만 학습시키는 것입니다. 그
다음 성능이 가장 나쁜 트라이얼들의 일부를 폐기하고, 남은 것들을 더
오래 학습시킵니다. 이 과정을 반복하면, 더 적은 수의 트라이얼이 더
오래 실행되고, 결국 적어도 하나의 트라이얼은 $r_{max}$ 에폭에 도달합니다.

더 형식적으로는, 최소 예산 $r_{\mathrm{min}}$ (예: 1 에폭), 최대 예산
$r_{max}$ (예: 이전 예제의 `max_epochs`), 그리고 하빙 상수
$\eta\in\{2, 3, \dots\}$를 고려해 보겠습니다. 단순화를 위해
$K \in \mathbb{I}$로 $r_{max} = r_{\mathrm{min}} \eta^K$라고 가정합니다.
그러면 초기 구성의 수는 $N = \eta^K$입니다. 룽(rung) 집합을
$\mathcal{R} = \{ r_{\mathrm{min}}, r_{\mathrm{min}}\eta, r_{\mathrm{min}}\eta^2, \dots, r_{max} \}$로
정의합시다.

연속 하빙의 한 라운드는 다음과 같이 진행됩니다. 먼저 $N$개의
트라이얼을 첫 번째 룽 $r_{\mathrm{min}}$까지 실행합니다. 검증 오차를
정렬하여, 상위 $1 / \eta$ 분율($\eta^{K-1}$개의 구성에 해당)을 남기고
나머지는 모두 폐기합니다. 살아남은 트라이얼들은 다음 룽
($r_{\mathrm{min}}\eta$ 에폭)까지 학습되며, 이 과정이 반복됩니다.
각 룽마다 $1 / \eta$ 분율의 트라이얼이 살아남고, 그것들의 학습은
$\eta$배 더 큰 예산으로 계속됩니다. 이러한 특정 $N$의 선택으로,
오직 하나의 트라이얼만이 전체 예산 $r_{max}$까지 학습됩니다. 이러한
연속 하빙 라운드가 끝나면, 새로운 초기 구성 집합으로 다음 라운드를
시작하고, 총 예산이 소진될 때까지 반복합니다.

![무작위 하이퍼파라미터 구성의 학습 곡선.](../img/sh.svg)

연속 하빙을 구현하기 위해 :numref:`sec_api_hpo`의 `HPOScheduler` 기본
클래스를 서브클래싱하여, 일반적인 `HPOSearcher` 객체가 구성을 표집할
수 있도록 합니다(아래 예제에서는 `RandomSearcher`가 됩니다).
추가적으로, 사용자는 입력으로 최소 자원 $r_{\mathrm{min}}$, 최대 자원
$r_{max}$, 그리고 $\eta$를 전달해야 합니다. 스케줄러 내부에서, 저희는
현재 룽 $r_i$에서 아직 산정되어야 할 구성들의 큐를 유지합니다. 다음
룽으로 점프할 때마다 큐를 업데이트합니다.

```{.python .input  n=2}
class SuccessiveHalvingScheduler(d2l.HPOScheduler):  #@save
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.save_hyperparameters()
        # Compute K, which is later used to determine the number of configurations
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # Define the rungs
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            # The final rung should be r_max
            self.rung_levels.append(r_max)
            self.K += 1
        # Bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        # Our processing queue
        self.queue = []
```

처음에 저희의 큐는 비어 있고, $n = \textrm{prefact} \cdot \eta^{K}$개의
구성으로 큐를 채우며, 이들은 먼저 가장 작은 룽 $r_{\mathrm{min}}$에서
산정됩니다. 여기서 $\textrm{prefact}$는 저희 코드를 다른 맥락에서
재사용할 수 있게 해 줍니다. 이 절의 목적상, $\textrm{prefact} = 1$로
고정합니다. 자원이 사용 가능해질 때마다, 그리고 `HPOTuner` 객체가
`suggest` 함수를 호출할 때마다, 큐에서 원소를 반환합니다. 연속 하빙의
한 라운드가 끝나면(즉, 가장 높은 자원 수준 $r_{max}$에서 살아남은
모든 구성을 산정했고 큐가 비어 있다는 의미), 무작위로 표집된 새로운
구성 집합으로 전체 과정을 다시 시작합니다.

```{.python .input  n=12}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def suggest(self):
    if len(self.queue) == 0:
        # Start a new round of successive halving
        # Number of configurations for the first rung:
        n0 = int(self.prefact * self.eta ** self.K)
        for _ in range(n0):
            config = self.searcher.sample_configuration()
            config["max_epochs"] = self.r_min  # Set r = r_min
            self.queue.append(config)
    # Return an element from the queue
    return self.queue.pop()
```

새로운 데이터 점을 수집하면, 먼저 서처 모듈을 업데이트합니다. 그 후
현재 룽의 모든 데이터 점을 이미 수집했는지 확인합니다. 그렇다면, 모든
구성을 정렬하고 상위 $\frac{1}{\eta}$ 구성을 큐에 넣습니다.

```{.python .input  n=4}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def update(self, config: dict, error: float, info=None):
    ri = int(config["max_epochs"])  # Rung r_i
    # Update our searcher, e.g if we use Bayesian optimization later
    self.searcher.update(config, error, additional_info=info)
    self.all_observed_error_at_rungs[ri].append((config, error))
    if ri < self.r_max:
        # Bookkeeping
        self.observed_error_at_rungs[ri].append((config, error))
        # Determine how many configurations should be evaluated on this rung
        ki = self.K - self.rung_levels.index(ri)
        ni = int(self.prefact * self.eta ** ki)
        # If we observed all configuration on this rung r_i, we estimate the
        # top 1 / eta configuration, add them to queue and promote them for
        # the next rung r_{i+1}
        if len(self.observed_error_at_rungs[ri]) >= ni:
            kiplus1 = ki - 1
            niplus1 = int(self.prefact * self.eta ** kiplus1)
            best_performing_configurations = self.get_top_n_configurations(
                rung_level=ri, n=niplus1
            )
            riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
            # Queue may not be empty: insert new entries at the beginning
            self.queue = [
                dict(config, max_epochs=riplus1)
                for config in best_performing_configurations
            ] + self.queue
            self.observed_error_at_rungs[ri] = []  # Reset
```

구성은 현재 룽에서 관측된 성능에 기반해 정렬됩니다.

```{.python .input  n=4}
%%tab pytorch

@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def get_top_n_configurations(self, rung_level, n):
    rung = self.observed_error_at_rungs[rung_level]
    if not rung:
        return []
    sorted_rung = sorted(rung, key=lambda x: x[1])
    return [x[0] for x in sorted_rung[:n]]
```

저희의 신경망 예제에서 연속 하빙이 어떻게 동작하는지 살펴보겠습니다.
$r_{\mathrm{min}} = 2$, $\eta = 2$, $r_{max} = 10$을 사용하므로, 룽
레벨은 $2, 4, 8, 10$이 됩니다.

```{.python .input  n=5}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2
num_gpus=1

config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

스케줄러를 새로운 `SuccessiveHalvingScheduler`로 교체하기만 하면
됩니다.

```{.python .input  n=14}
searcher = d2l.RandomSearcher(config_space, initial_config=initial_config)
scheduler = SuccessiveHalvingScheduler(
    searcher=searcher,
    eta=eta,
    r_min=min_number_of_epochs,
    r_max=max_number_of_epochs,
)
tuner = d2l.HPOTuner(
    scheduler=scheduler,
    objective=d2l.hpo_objective_lenet,
)
tuner.run(number_of_trials=30)
```

저희가 산정한 모든 구성의 학습 곡선을 시각화할 수 있습니다. 대부분의
구성은 일찍 중단되고, 더 성능이 좋은 구성만이 $r_{max}$까지 살아남습니다.
이를 모든 구성에 $r_{max}$를 할당하는 일반 랜덤 탐색과 비교해 보세요.

```{.python .input  n=19}
for rung_index, rung in scheduler.all_observed_error_at_rungs.items():
    errors = [xi[1] for xi in rung]
    d2l.plt.scatter([rung_index] * len(errors), errors)
d2l.plt.xlim(min_number_of_epochs - 0.5, max_number_of_epochs + 0.5)
d2l.plt.xticks(
    np.arange(min_number_of_epochs, max_number_of_epochs + 1),
    np.arange(min_number_of_epochs, max_number_of_epochs + 1)
)
d2l.plt.ylabel("validation error")
d2l.plt.xlabel("epochs")
```

마지막으로, 저희의 `SuccessiveHalvingScheduler` 구현에서 약간의 복잡한
부분에 주목해 주세요. 어떤 작업자가 작업을 실행할 여유가 있고, 현재
룽이 거의 완전히 채워졌을 때 `suggest`가 호출되지만, 또 다른 작업자가
여전히 산정 중이라고 해 보겠습니다. 이 작업자의 지표 값이 없기 때문에,
다음 룽을 열기 위한 상위 $1 / \eta$ 분율을 결정할 수 없습니다.
한편으로는, 여유 작업자에게 작업을 할당하여 유휴 상태가 되지 않게
하고 싶습니다. 저희의 해결책은 새로운 연속 하빙 라운드를 시작하고,
그 작업자를 거기서 첫 번째 트라이얼에 할당하는 것입니다. 그러나
`update`에서 룽이 완료되면, 새로운 구성을 큐의 시작 부분에 삽입하여
다음 라운드의 구성보다 우선순위를 가지도록 합니다.

## 요약

이 절에서는 다중 충실도 하이퍼파라미터 최적화의 개념을 소개했습니다.
여기서는 전체 에폭 수에 대한 검증 오차의 대용물로서, 특정 학습 에폭
수 이후의 검증 오차와 같이 목적 함수에 대한 산정 비용이 저렴한
근사에 접근할 수 있다고 가정합니다. 다중 충실도 하이퍼파라미터
최적화는 단순히 벽시계 시간을 줄이는 것이 아니라, HPO의 전체 계산량을
줄일 수 있게 해 줍니다.

저희는 간단하지만 효율적인 다중 충실도 HPO 알고리즘인 연속 하빙을
구현하고 산정했습니다.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12094)
:end_tab:
