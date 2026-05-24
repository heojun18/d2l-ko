# 계산 성능
:label:`chap_performance`

딥러닝에서는,
데이터셋과 모델이 보통 크기 때문에,
무거운 계산을 수반합니다.
따라서 계산 성능은 아주 중요합니다.
이 챕터는 계산 성능에 영향을 미치는 주요 요인들에 초점을 맞춥니다.
명령형 프로그래밍, 심볼릭 프로그래밍, 비동기 계산, 자동 병렬화, 다중 GPU 계산이 그것입니다.
이 챕터를 학습함으로써, 이전 챕터들에서 구현된 모델들의 계산 성능을 더 향상시킬 수 있습니다.
예를 들어, 정확도에 영향을 주지 않으면서 학습 시간을 줄임으로써 그렇게 할 수 있습니다.

```toc
:maxdepth: 2

hybridize
async-computation
auto-parallelism
hardware
multiple-gpus
multiple-gpus-concise
parameterserver
```

