```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# 비동기 연속 하빙

:label:`sec_sh_async`

:numref:`sec_rs_async`에서 보았듯이, 여러 인스턴스나 단일 인스턴스의
여러 CPU/GPU에 걸쳐 하이퍼파라미터 구성의 산정을 분산시킴으로써 HPO를
가속할 수 있습니다. 그러나 랜덤 탐색과 비교했을 때, 분산 설정에서
연속 하빙(SH)을 비동기적으로 실행하는 것은 간단하지 않습니다. 다음에
어떤 구성을 실행할지 결정하려면, 먼저 현재 룽 레벨의 모든 관측값을
수집해야 합니다. 이는 각 룽 레벨에서 작업자들을 동기화해야 함을
의미합니다. 예를 들어, 가장 낮은 룽 레벨 $r_{\mathrm{min}}$의 경우,
그 중 $\frac{1}{\eta}$를 다음 룽 레벨로 승격시키기 전에 먼저
모든 $N = \eta^K$개의 구성을 산정해야 합니다.

어떤 분산 시스템에서든, 동기화는 일반적으로 작업자의 유휴 시간을
의미합니다. 첫째, 저희는 하이퍼파라미터 구성에 걸쳐 학습 시간에 큰
변동성을 자주 관찰합니다. 예를 들어, 층당 필터 수가 하이퍼파라미터라고
가정하면, 더 적은 필터를 가진 신경망은 더 많은 필터를 가진 신경망보다
학습이 빨리 끝나므로, 낙오자로 인한 작업자의 유휴 시간이 발생합니다.
더욱이, 룽 레벨의 슬롯 수가 항상 작업자 수의 배수인 것은 아니어서,
어떤 작업자들은 한 배치 동안 통째로 유휴 상태로 있을 수도 있습니다.

:numref:`synchronous_sh` 그림은 두 작업자로 4개의 서로 다른 트라이얼에
대해 $\eta=2$인 동기식 SH의 스케줄링을 보여줍니다. Trial-0와 Trial-1을
1 에폭 동안 산정하는 것으로 시작하여, 끝나자마자 즉시 다음 두
트라이얼을 진행합니다. 그 후 다음 룽 레벨로 최고의 두 트라이얼,
즉 Trial-0와 Trial-3을 승격하기 전에, 다른 트라이얼들보다 상당히 더
많은 시간이 걸리는 Trial-2가 끝나기를 먼저 기다려야 합니다. 이로 인해
Worker-1에 유휴 시간이 발생합니다. 그 다음 Rung 1을 진행합니다. 여기서도
Trial-3이 Trial-0보다 오래 걸려, Worker-0에도 추가적인 유휴 시간이
발생합니다. Rung-2에 도달하면, 최고의 트라이얼인 Trial-0만 남아
한 작업자만 사용하게 됩니다. 그 시간 동안 Worker-1이 유휴 상태가
되는 것을 피하기 위해, 대부분의 SH 구현은 이미 다음 라운드로 진행하여
첫 번째 룽에서 새로운 트라이얼(예: Trial-4)을 산정하기 시작합니다.

![두 작업자를 사용한 동기식 연속 하빙.](../img/sync_sh.svg)
:label:`synchronous_sh`

비동기 연속 하빙(asynchronous successive halving, ASHA) :cite:`li-arxiv18`은
SH를 비동기 병렬 시나리오에 적응시킵니다. ASHA의 핵심 아이디어는
현재 룽 레벨에서 적어도 $\eta$개의 관측을 수집하자마자 구성들을 다음
룽 레벨로 승격하는 것입니다. 이 결정 규칙은 차선의 승격을 초래할 수
있습니다. 즉, 구성이 다음 룽 레벨로 승격될 수 있지만, 돌이켜보면 같은
룽 레벨의 대부분의 다른 구성들과 비교했을 때 우호적으로 비교되지
않을 수도 있습니다. 한편으로는, 이 방식으로 모든 동기화 지점을
없앨 수 있습니다. 실제로는, 그러한 차선의 초기 승격이 성능에 미치는
영향은 적은데, 이는 하이퍼파라미터 구성의 순위가 룽 레벨에 걸쳐
상당히 일관적인 경우가 많기도 하지만, 룽이 시간이 지나면서 커지면서
그 레벨의 지표 값 분포를 점점 더 잘 반영하기 때문이기도 합니다.
작업자가 여유가 있지만 어떤 구성도 승격될 수 없다면, $r = r_{\mathrm{min}}$,
즉 첫 번째 룽 레벨에서 새로운 구성을 시작합니다.

:numref:`asha`는 ASHA에 대해 동일한 구성들의 스케줄링을 보여줍니다.
Trial-1이 끝나면, 두 트라이얼(즉, Trial-0과 Trial-1)의 결과를 수집하고
즉시 그 중 더 나은 쪽(Trial-0)을 다음 룽 레벨로 승격합니다. Trial-0이
룽 1에서 끝난 후에는, 추가 승격을 지원하기에는 그곳에 트라이얼이
너무 적습니다. 따라서 룽 0을 계속하여 Trial-3을 산정합니다. Trial-3이
끝나면, Trial-2는 여전히 진행 중입니다. 이 시점에서 룽 0에서 산정된
3개의 트라이얼과 룽 1에서 이미 산정된 한 개의 트라이얼이 있습니다.
Trial-3이 룽 0에서 Trial-0보다 성능이 나쁘고 $\eta=2$이기 때문에,
아직 새로운 트라이얼을 승격시킬 수 없으며, 대신 Worker-1이 처음부터
Trial-4를 시작합니다. 그러나 Trial-2가 끝나고 Trial-3보다 점수가 나쁘면,
후자가 룽 1로 승격됩니다. 그 후, 룽 1에서 2개의 산정값을 수집하게
되며, 이는 이제 Trial-0을 룽 2로 승격할 수 있다는 뜻입니다. 동시에,
Worker-1은 룽 0에서 새로운 트라이얼(즉, Trial-5)을 산정하는 것을
계속합니다.


![두 작업자를 사용한 비동기 연속 하빙(ASHA).](../img/asha.svg)
:label:`asha`

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## 목적 함수

:numref:`sec_rs_async`와 동일한 목적 함수와 함께 *Syne Tune*을
사용하겠습니다.

```{.python .input  n=54}
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

이전과 동일한 구성 공간도 사용하겠습니다.

```{.python .input  n=55}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2

config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": max_number_of_epochs,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## 비동기 스케줄러

먼저, 트라이얼을 동시에 산정하는 작업자의 수를 정의합니다. 또한
총 벽시계 시간의 상한을 정의함으로써 랜덤 탐색을 얼마나 오래 실행할지
지정해야 합니다.

```{.python .input  n=56}
n_workers = 2  # Needs to be <= the number of available GPUs
max_wallclock_time = 12 * 60  # 12 minutes
```

ASHA를 실행하기 위한 코드는 비동기 랜덤 탐색에서 했던 것의 간단한
변형입니다.

```{.python .input  n=56}
mode = "min"
metric = "validation_error"
resource_attr = "epoch"

scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
    max_resource_attr="max_epochs",
    resource_attr=resource_attr,
    grace_period=min_number_of_epochs,
    reduction_factor=eta,
)
```

여기서 `metric`과 `resource_attr`은 `report` 콜백과 함께 사용되는
키 이름을 지정하고, `max_resource_attr`은 목적 함수의 어떤 입력이
$r_{\mathrm{max}}$에 해당하는지를 나타냅니다. 또한 `grace_period`는
$r_{\mathrm{min}}$을 제공하고, `reduction_factor`는 $\eta$입니다.
이전처럼 Syne Tune을 실행할 수 있습니다(약 12분이 걸립니다).

```{.python .input  n=57}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)

stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
tuner.run()
```

저희가 실행하는 것은 성능이 낮은 트라이얼이 조기에 중단되는 ASHA의
변형이라는 점에 유의하세요. 이는 각 학습 작업이 고정된 `max_epochs`로
시작되는 :numref:`sec_mf_hpo_sh`의 구현과 다릅니다. 후자의 경우,
전체 10 에폭에 도달하는 좋은 성능의 트라이얼은 먼저 1, 그 다음 2,
그 다음 4, 그 다음 8 에폭을 학습해야 하며, 매번 처음부터 시작해야
합니다. 이러한 일시 중지 및 재개(pause-and-resume) 스케줄링은 각
에폭 이후 학습 상태를 체크포인팅함으로써 효율적으로 구현될 수 있지만,
여기서는 이 추가적인 복잡성을 피합니다. 실험이 끝난 후, 결과를 가져와
시각화할 수 있습니다.

```{.python .input  n=59}
d2l.set_figsize()
e = load_experiment(tuner.name)
e.plot()
```

## 최적화 과정의 시각화

다시 한 번, 모든 트라이얼의 학습 곡선(플롯의 각 색상은 하나의 트라이얼을
나타냄)을 시각화합니다. 이를 :numref:`sec_rs_async`의 비동기 랜덤
탐색과 비교해 보세요. :numref:`sec_mf_hpo`의 연속 하빙에서 보았듯이,
대부분의 트라이얼은 1 또는 2 에폭($r_{\mathrm{min}}$ 또는
$\eta * r_{\mathrm{min}}$)에서 중단됩니다. 그러나 트라이얼들은 에폭당
필요한 시간이 다르기 때문에 같은 지점에서 멈추지 않습니다. ASHA 대신
표준 연속 하빙을 실행했다면, 구성을 다음 룽 레벨로 승격시키기 전에
저희 작업자들을 동기화해야 했을 것입니다.

```{.python .input  n=60}
d2l.set_figsize([6, 2.5])
results = e.results
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

랜덤 탐색과 비교했을 때, 연속 하빙은 비동기 분산 설정에서 실행하기가
그렇게 사소하지 않습니다. 동기화 지점을 피하기 위해, 잘못된 구성을
승격하는 것을 의미할 수도 있지만, 가능한 한 빨리 구성을 다음 룽
레벨로 승격합니다. 실제로 이것은 보통 큰 문제가 되지 않으며, 비동기
스케줄링 대 동기식 스케줄링의 이득이 차선의 의사결정으로 인한 손실보다
보통 훨씬 큽니다.


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12101)
:end_tab:
