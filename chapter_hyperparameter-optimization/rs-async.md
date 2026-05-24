```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# 비동기 랜덤 탐색
:label:`sec_rs_async`

이전 :numref:`sec_api_hpo`에서 보았듯이, 하이퍼파라미터 구성의 산정
비용이 크기 때문에 랜덤 탐색이 좋은 하이퍼파라미터 구성을 반환할
때까지 몇 시간 혹은 며칠을 기다려야 할 수도 있습니다. 실제로는 같은
머신의 여러 GPU나, GPU 하나씩을 가진 여러 머신과 같이, 자원 풀에
접근할 수 있는 경우가 많습니다. 그렇다면 이런 질문이 떠오릅니다.
*어떻게 하면 랜덤 탐색을 효율적으로 분산시킬 수 있을까요?*

일반적으로, 저희는 동기식과 비동기식 병렬 하이퍼파라미터 최적화를
구분합니다(:numref:`distributed_scheduling` 참고). 동기식 설정에서는,
다음 배치를 시작하기 전에 동시에 실행 중인 모든 트라이얼이 끝나기를
기다립니다. 심층 신경망의 필터 수나 층 수와 같은 하이퍼파라미터를
포함하는 구성 공간을 생각해 보세요. 더 많은 층이나 필터를 포함하는
하이퍼파라미터 구성은 자연스럽게 완료에 더 많은 시간이 걸리고, 같은
배치의 다른 모든 트라이얼은 최적화 과정을 계속하기 전에 동기화
지점(:numref:`distributed_scheduling`의 회색 영역)에서 기다려야 합니다.

비동기식 설정에서는 자원이 사용 가능해지는 즉시 새로운 트라이얼을
스케줄링합니다. 어떠한 동기화 오버헤드도 피할 수 있기 때문에, 이는
저희의 자원을 최적으로 활용합니다. 랜덤 탐색의 경우, 각각의 새로운
하이퍼파라미터 구성은 다른 모든 구성과 독립적으로 선택되며, 특히 이전
산정의 관측을 활용하지 않습니다. 즉, 랜덤 탐색은 비동기적으로 사소하게
병렬화할 수 있습니다. 이는 이전 관측에 기반하여 결정을 내리는 더
정교한 방법들(:numref:`sec_sh_async` 참고)에서는 간단하지 않습니다.
순차적인 설정보다 더 많은 자원이 필요하지만, 비동기 랜덤 탐색은 $K$개의
트라이얼을 병렬로 실행할 수 있다면 특정 성능에 도달하는 시간이 $K$배
빨라진다는 선형적 가속을 보입니다.


![하이퍼파라미터 최적화 과정을 동기식 또는 비동기식으로 분산하기. 순차 설정과 비교하면, 총 계산량은 일정하게 유지하면서 전체 벽시계 시간을 줄일 수 있습니다. 동기식 스케줄링은 낙오자(straggler)가 있는 경우 작업자(worker)가 유휴 상태가 되는 결과를 낳을 수 있습니다.](../img/distributed_scheduling.svg)
:label:`distributed_scheduling`

이 노트북에서는 같은 머신에서 여러 파이썬 프로세스로 트라이얼이 실행되는
비동기 랜덤 탐색을 살펴봅니다. 분산 작업 스케줄링과 실행을 처음부터
구현하기는 어렵습니다. 비동기 HPO를 위한 간단한 인터페이스를 제공하는
*Syne Tune* :cite:`salinas-automl22`을 사용하겠습니다. Syne Tune은
서로 다른 실행 백엔드와 함께 실행되도록 설계되어 있으며, 분산 HPO에
대해 더 알고 싶은 독자들께서는 그 간단한 API들을 살펴보시기를 권합니다.

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## 목적 함수

먼저, 이제 `report` 콜백을 통해 성능을 Syne Tune에 다시 반환하도록
새로운 목적 함수를 정의해야 합니다.

```{.python .input  n=34}
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l    
    from syne_tune import Reporter

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter() 
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Initialize the state of Trainer
            trainer.fit(model=model, data=data) 
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

Syne Tune의 `PythonBackend`는 함수 정의 내부에서 의존성을 임포트해야
한다는 점에 유의하세요.

## 비동기 스케줄러

먼저, 트라이얼을 동시에 산정하는 작업자(worker)의 수를 정의합니다.
또한 총 벽시계 시간의 상한을 정의함으로써 랜덤 탐색을 얼마나 오래
실행할지 지정해야 합니다.

```{.python .input  n=37}
n_workers = 2  # Needs to be <= the number of available GPUs

max_wallclock_time = 12 * 60  # 12 minutes
```

다음으로, 어떤 지표를 최적화할지, 그리고 그 지표를 최소화할지 최대화할지
명시합니다. 구체적으로, `metric`은 `report` 콜백에 전달되는 인자
이름과 일치해야 합니다.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

이전 예제의 구성 공간을 사용합니다. Syne Tune에서는 이 딕셔너리를
사용해 학습 스크립트에 상수 속성을 전달할 수도 있습니다. 저희는
`max_epochs`를 전달하기 위해 이 기능을 활용합니다. 더불어, 가장 먼저
산정될 구성을 `initial_config`에 지정합니다.

```{.python .input  n=39}
config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": 10,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

다음으로, 작업 실행을 위한 백엔드를 지정해야 합니다. 여기서는 병렬
작업이 서브프로세스로 실행되는 로컬 머신 상에서의 분산만 고려합니다.
다만, 대규모 HPO의 경우, 각 트라이얼이 전체 인스턴스를 사용하는
클러스터나 클라우드 환경에서도 실행할 수 있습니다.

```{.python .input  n=40}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)
```

이제 비동기 랜덤 탐색을 위한 스케줄러를 만들 수 있는데, 그 동작은
:numref:`sec_api_hpo`의 `BasicScheduler`와 유사합니다.

```{.python .input  n=41}
scheduler = RandomSearch(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
)
```

Syne Tune은 또한 주 실험 루프와 기록 관리가 중앙집중화되고, 스케줄러와
백엔드 간의 상호작용이 중재되는 `Tuner` 기능을 제공합니다.

```{.python .input  n=42}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler, 
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
```

이제 분산 HPO 실험을 실행해 보겠습니다. 저희의 정지 기준에 따라, 이
실험은 약 12분 동안 실행됩니다.

```{.python .input  n=43}
tuner.run()
```

산정된 모든 하이퍼파라미터 구성의 로그는 추가 분석을 위해 저장됩니다.
튜닝 작업 중 언제든지, 지금까지 얻은 결과를 손쉽게 가져와 인컴번트
궤적을 그릴 수 있습니다.

```{.python .input  n=46}
d2l.set_figsize()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## 비동기 최적화 과정의 시각화

아래에서는 비동기 최적화 과정 동안 모든 트라이얼의 학습 곡선(플롯의
각 색상은 하나의 트라이얼을 나타냄)이 어떻게 진화하는지 시각화합니다.
어느 시점에서든, 저희가 가진 작업자 수만큼 많은 트라이얼이 동시에
실행 중입니다. 어떤 트라이얼이 끝나면, 다른 트라이얼이 끝나기를
기다리지 않고 즉시 다음 트라이얼을 시작합니다. 비동기 스케줄링을
통해 작업자의 유휴 시간이 최소화됩니다.

```{.python .input  n=45}
d2l.set_figsize([6, 2.5])
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
    
d2l.plt.xlabel("wall-clock time")
d2l.plt.ylabel("objective function")
```

## 요약

병렬 자원에 걸쳐 트라이얼을 분산시킴으로써 랜덤 탐색의 대기 시간을
상당히 줄일 수 있습니다. 일반적으로, 저희는 동기식 스케줄링과
비동기식 스케줄링을 구분합니다. 동기식 스케줄링이란, 이전 배치가
끝난 후 하이퍼파라미터 구성의 새로운 배치를 표집하는 것을 의미합니다.
낙오자(다른 트라이얼들보다 완료에 시간이 더 오래 걸리는 트라이얼)가
있다면, 저희의 작업자들은 동기화 지점에서 기다려야 합니다. 비동기식
스케줄링은 자원이 사용 가능해지는 즉시 새로운 하이퍼파라미터 구성을
산정하므로, 모든 작업자가 어느 시점에서든 바쁘게 동작하도록 보장합니다.
랜덤 탐색은 비동기로 분산하기 쉽고 실제 알고리즘의 어떤 변경도 필요하지
않은 반면, 다른 방법들은 약간의 추가적인 수정이 필요합니다.

## 연습 문제

1. :numref:`sec_dropout`에서 구현되고 :numref:`sec_api_hpo`의 연습 문제 1에서 사용된 `DropoutMLP` 모델을 고려해 보세요.
    1. Syne Tune과 함께 사용할 목적 함수 `hpo_objective_dropoutmlp_synetune`을 구현하세요. 여러분의 함수가 매 에폭 후의 검증 오차를 보고하도록 하세요.
    2. :numref:`sec_api_hpo`의 연습 문제 1 설정을 사용하여, 랜덤 탐색과 베이지안 최적화를 비교하세요. SageMaker를 사용한다면, 실험을 병렬로 실행하기 위해 Syne Tune의 벤치마킹 기능을 자유롭게 사용해 보세요. 힌트: 베이지안 최적화는 `syne_tune.optimizer.baselines.BayesianOptimization`으로 제공됩니다.
    3. 이 연습 문제는 적어도 4개의 CPU 코어가 있는 인스턴스에서 실행해야 합니다. 위에서 사용된 방법들(랜덤 탐색, 베이지안 최적화) 중 하나에 대해, `n_workers=1`, `n_workers=2`, `n_workers=4`로 실험을 실행하고 결과(인컴번트 궤적)를 비교하세요. 적어도 랜덤 탐색에 대해서는, 작업자 수에 대한 선형적 스케일링을 관찰할 수 있어야 합니다. 힌트: 견고한 결과를 위해, 각각 여러 반복에 대해 평균을 내야 할 수도 있습니다.
2. *심화*. 이 연습 문제의 목표는 Syne Tune에서 새로운 스케줄러를 구현하는 것입니다.
    1. [d2lbook](https://github.com/d2l-ai/d2l-en/blob/master/INFO.md#installation-for-developers)과 [syne-tune](https://syne-tune.readthedocs.io/en/latest/getting_started.html) 소스를 모두 포함하는 가상 환경을 만드세요.
    2. :numref:`sec_api_hpo`의 연습 문제 2의 `LocalSearcher`를 Syne Tune의 새로운 서처로 구현하세요. 힌트: [이 튜토리얼](https://syne-tune.readthedocs.io/en/latest/tutorials/developer/README.html)을 읽어 보세요. 또는, 이 [예제](https://syne-tune.readthedocs.io/en/latest/examples.html#launch-hpo-experiment-with-home-made-scheduler)를 따라가도 됩니다.
    3. `DropoutMLP` 벤치마크에서 여러분의 새로운 `LocalSearcher`를 `RandomSearch`와 비교하세요.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12093)
:end_tab:
