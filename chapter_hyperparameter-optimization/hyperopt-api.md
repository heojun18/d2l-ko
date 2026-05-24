```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# 하이퍼파라미터 최적화 API
:label:`sec_api_hpo`

방법론에 대해 본격적으로 다루기 전에, 먼저 다양한 HPO 알고리즘을
효율적으로 구현할 수 있게 해 주는 기본적인 코드 구조를 논의합니다.
일반적으로, 여기서 다루는 모든 HPO 알고리즘은 두 가지 의사결정
기본 동작인 *탐색(searching)*과 *스케줄링(scheduling)*을 구현해야
합니다. 첫째, 새로운 하이퍼파라미터 구성을 표집해야 하는데, 이는
보통 구성 공간에 대한 일종의 탐색을 수반합니다. 둘째, 각 구성에
대해 HPO 알고리즘은 그 산정을 스케줄링하고 얼마나 많은 자원을
할당할지 결정해야 합니다. 어떤 구성의 산정을 시작하고 나면, 이를
*트라이얼(trial)*이라고 부르겠습니다. 저희는 이 결정들을 두 개의
클래스인 `HPOSearcher`와 `HPOScheduler`에 매핑합니다. 그 위에,
최적화 과정을 실행하는 `HPOTuner` 클래스도 제공합니다.

이러한 스케줄러와 서처(searcher) 개념은 Syne Tune
:cite:`salinas-automl22`, Ray Tune :cite:`liaw-arxiv18`, Optuna
:cite:`akiba-sigkdd19`와 같은 널리 사용되는 HPO 라이브러리에서도
구현되어 있습니다.

```{.python .input  n=2}
%%tab pytorch
import time
from d2l import torch as d2l
from scipy import stats
```

## 서처(Searcher)

아래에서는 `sample_configuration` 함수를 통해 새로운 후보 구성을
제공하는, 서처의 기본 클래스를 정의합니다. 이 함수를 구현하는 간단한
방법은 :numref:`sec_what_is_hpo`의 랜덤 탐색에서 했던 것처럼, 구성을
균등하게 무작위로 표집하는 것입니다. 베이지안 최적화와 같은 더
정교한 알고리즘들은 이전 트라이얼들의 성능에 기반하여 이러한 결정을
내립니다. 그 결과, 이러한 알고리즘들은 시간이 지남에 따라 더 유망한
후보들을 표집할 수 있게 됩니다. 저희는 이전 트라이얼들의 이력을
업데이트하기 위해 `update` 함수를 추가하는데, 이 이력은 이후 표집
분포를 개선하는 데 활용될 수 있습니다.

```{.python .input  n=3}
%%tab pytorch
class HPOSearcher(d2l.HyperParameters):  #@save
    def sample_configuration() -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        pass
```

다음 코드는 이 API에서 이전 절의 랜덤 탐색 옵티마이저를 구현하는
방법을 보여줍니다. 약간의 확장으로, 사용자가 `initial_config`를 통해
가장 먼저 산정할 구성을 지정할 수 있게 하고, 이후의 구성들은 무작위로
추출됩니다.

```{.python .input  n=4}
%%tab pytorch
class RandomSearcher(HPOSearcher):  #@save
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result
```

## 스케줄러(Scheduler)

새로운 트라이얼을 위한 구성을 표집하는 것 외에도, 트라이얼을 언제,
얼마나 오랫동안 실행할지 결정해야 합니다. 실제로, 이러한 모든 결정은
`HPOScheduler`에 의해 이루어지며, 새로운 구성의 선택은 `HPOSearcher`에
위임합니다. `suggest` 메서드는 학습을 위한 어떤 자원이 사용 가능해질
때마다 호출됩니다. 서처의 `sample_configuration`을 호출하는 것 외에도,
`max_epochs`와 같은 파라미터(즉, 모델을 얼마나 오래 학습시킬지)도
결정할 수 있습니다. `update` 메서드는 트라이얼이 새로운 관측을 반환할
때마다 호출됩니다.

```{.python .input  n=5}
%%tab pytorch
class HPOScheduler(d2l.HyperParameters):  #@save
    def suggest(self) -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
```

랜덤 탐색뿐만 아니라 다른 HPO 알고리즘들을 구현하기 위해서도, 새로운
자원이 사용 가능해질 때마다 새로운 구성을 스케줄링하는 기본 스케줄러만
있으면 됩니다.

```{.python .input  n=6}
%%tab pytorch
class BasicScheduler(HPOScheduler):  #@save
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)
```

## 튜너(Tuner)

마지막으로, 스케줄러/서처를 실행하고 결과에 대한 약간의 기록 관리를
하는 구성요소가 필요합니다. 다음 코드는 한 학습 작업을 다음 작업에
이어 순차적으로 산정하는 HPO 트라이얼의 순차 실행을 구현하며, 기본
예제 역할을 합니다. 이후 더 확장 가능한 분산 HPO 사례에서는
*Syne Tune*을 사용하겠습니다.

```{.python .input  n=7}
%%tab pytorch
class HPOTuner(d2l.HyperParameters):  #@save
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        # Bookeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")
            error = self.objective(**config)
            error = float(d2l.numpy(error.cpu()))
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")
```

## HPO 알고리즘의 성능 기록 관리

어떤 HPO 알고리즘을 사용하든, 저희가 가장 관심 있는 것은 가장 성능이
좋은 구성(이를 *인컴번트(incumbent)*라고 함)과 주어진 벽시계 시간
이후의 그 검증 오차입니다. 이 때문에 저희는 반복마다의 `runtime`을
추적하며, 여기에는 한 번의 산정 실행에 걸린 시간(`objective` 호출)과
의사결정에 걸린 시간(`scheduler.suggest` 호출)이 모두 포함됩니다.
이후에는 `cumulative_runtime`을 `incumbent_trajectory`에 대해 그려,
`scheduler`(및 `searcher`)로 정의된 HPO 알고리즘의 *언제든지 성능
(any-time performance)*을 시각화하겠습니다. 이를 통해 옵티마이저가
찾은 구성이 얼마나 잘 동작하는지뿐만 아니라, 옵티마이저가 얼마나
빨리 그것을 찾을 수 있는지도 정량화할 수 있습니다.

```{.python .input  n=8}
%%tab pytorch
@d2l.add_to_class(HPOTuner)  #@save
def bookkeeping(self, config: dict, error: float, runtime: float):
    self.records.append({"config": config, "error": error, "runtime": runtime})
    # Check if the last hyperparameter configuration performs better 
    # than the incumbent
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
    # Add current best observed performance to the optimization trajectory
    self.incumbent_trajectory.append(self.incumbent_error)
    # Update runtime
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

## 예제: 합성곱 신경망의 하이퍼파라미터 최적화

이제 저희의 새로운 랜덤 탐색 구현을 사용하여, :numref:`sec_lenet`의
`LeNet` 합성곱 신경망의 *배치 크기*와 *학습률*을 최적화해 보겠습니다.
검증 오차로 다시 한 번 사용될 목적 함수를 정의하는 것부터 시작합니다.

```{.python .input  n=9}
%%tab pytorch
def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):  #@save
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error
```

구성 공간도 정의해야 합니다. 또한, 가장 먼저 산정될 구성은
:numref:`sec_lenet`에서 사용된 기본 설정입니다.

```{.python .input  n=10}
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

이제 랜덤 탐색을 시작할 수 있습니다.

```{.python .input}
searcher = RandomSearcher(config_space, initial_config=initial_config)
scheduler = BasicScheduler(searcher=searcher)
tuner = HPOTuner(scheduler=scheduler, objective=hpo_objective_lenet)
tuner.run(number_of_trials=5)
```

아래에서는 랜덤 탐색의 언제든지 성능(any-time performance)을 얻기
위해 인컴번트의 최적화 궤적을 그려 봅니다.

```{.python .input  n=11}
board = d2l.ProgressBoard(xlabel="time", ylabel="error")
for time_stamp, error in zip(
    tuner.cumulative_runtime, tuner.incumbent_trajectory
):
    board.draw(time_stamp, error, "random search", every_n=1)
```

## HPO 알고리즘 비교

학습 알고리즘이나 모델 아키텍처를 비교할 때와 마찬가지로, 서로 다른
HPO 알고리즘을 가장 잘 비교하는 방법을 이해하는 것이 중요합니다. 매
HPO 실행은 두 가지 주요 무작위성의 원천에 의존합니다. 무작위 가중치
초기화나 미니배치 순서와 같은 학습 과정의 무작위 효과와, 랜덤 탐색의
무작위 표집과 같은 HPO 알고리즘 자체의 내재적 무작위성입니다. 따라서
서로 다른 알고리즘을 비교할 때는, 각 실험을 여러 번 실행하고, 난수
생성기의 서로 다른 시드에 기반한 알고리즘의 여러 반복 모집단에 대해
평균이나 중앙값과 같은 통계를 보고하는 것이 매우 중요합니다.

이를 설명하기 위해, 피드포워드 신경망의 하이퍼파라미터 조정에 대해
랜덤 탐색(:numref:`sec_rs` 참고)과 베이지안 최적화
:cite:`snoek-nips12`를 비교해 보겠습니다. 각 알고리즘은 서로 다른
난수 시드로 $50$번 산정되었습니다. 실선은 이 $50$회 반복에 걸친
인컴번트의 평균 성능을 나타내고, 점선은 표준 편차를 나타냅니다. 약
1000초까지는 랜덤 탐색과 베이지안 최적화가 대체로 비슷하게 동작하는
것을 볼 수 있지만, 베이지안 최적화는 과거 관측을 활용하여 더 나은
구성을 식별할 수 있으므로, 그 이후 빠르게 랜덤 탐색을 능가합니다.


![두 알고리즘 A와 B를 비교하기 위한 언제든지 성능 플롯 예시.](../img/example_anytime_performance.svg)
:label:`example_anytime_performance`

## 요약

이 절에서는 이 장에서 살펴볼 다양한 HPO 알고리즘을 구현하기 위한
간단하면서도 유연한 인터페이스를 제시했습니다. 비슷한 인터페이스를
널리 사용되는 오픈소스 HPO 프레임워크에서도 찾을 수 있습니다. 또한
HPO 알고리즘을 어떻게 비교할 수 있는지, 그리고 주의해야 할 잠재적인
함정들도 살펴보았습니다.

## 연습 문제

1. 이 연습 문제의 목표는 약간 더 도전적인 HPO 문제를 위한 목적 함수를 구현하고, 더 현실적인 실험을 실행하는 것입니다. :numref:`sec_dropout`에서 구현된 2개의 은닉층 MLP `DropoutMLP`를 사용하겠습니다.
    1. 모델의 모든 하이퍼파라미터와 `batch_size`에 의존해야 하는 목적 함수를 코드로 작성하세요. `max_epochs=50`을 사용합니다. 여기서는 GPU가 도움이 되지 않으므로 `num_gpus=0`으로 합니다. 힌트: `hpo_objective_lenet`을 수정해 보세요.
    2. `num_hiddens_1`, `num_hiddens_2`는 $[8, 1024]$ 범위의 정수, 드롭아웃 값은 $[0, 0.95]$ 범위, `batch_size`는 $[16, 384]$ 범위인 합리적인 탐색 공간을 선택하세요. `scipy.stats`의 합리적인 분포를 사용하여 `config_space`에 대한 코드를 제공하세요.
    3. 이 예제에서 `number_of_trials=20`으로 랜덤 탐색을 실행하고 결과를 그려 보세요. :numref:`sec_dropout`의 기본 구성 `initial_config = {'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`을 먼저 산정해야 합니다.
2. 이 연습 문제에서는 과거 데이터에 기반하여 결정을 내리는 새로운 서처(`HPOSearcher`의 서브클래스)를 구현해 봅니다. 이 서처는 `probab_local`, `num_init_random` 파라미터에 의존합니다. 그 `sample_configuration` 메서드는 다음과 같이 동작합니다. 처음 `num_init_random`번 호출에서는, `RandomSearcher.sample_configuration`과 동일하게 수행합니다. 그렇지 않으면, 확률 `1 - probab_local`로 `RandomSearcher.sample_configuration`과 동일하게 수행합니다. 그 외의 경우, 지금까지 가장 작은 검증 오차를 달성한 구성을 골라, 그 하이퍼파라미터 중 하나를 무작위로 선택하고, 그 값을 `RandomSearcher.sample_configuration`에서처럼 무작위로 표집하되, 다른 모든 값은 그대로 둡니다. 이 하나의 하이퍼파라미터를 제외하고는 지금까지의 최적 구성과 동일한, 이 구성을 반환하세요.
    1. 이 새로운 `LocalSearcher`를 코드로 작성하세요. 힌트: 여러분의 서처는 생성 시 `config_space`를 인자로 필요로 합니다. `RandomSearcher` 타입의 멤버를 자유롭게 사용해도 됩니다. `update` 메서드도 구현해야 합니다.
    2. 이전 연습 문제의 실험을 다시 실행하되, `RandomSearcher` 대신 여러분의 새로운 서처를 사용해 보세요. `probab_local`, `num_init_random`의 서로 다른 값으로 실험해 보세요. 다만, 서로 다른 HPO 방법 간의 적절한 비교를 위해서는 실험을 여러 번 반복하고, 이상적으로는 여러 벤치마크 태스크를 고려해야 한다는 점에 유의하세요.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12092)
:end_tab:
