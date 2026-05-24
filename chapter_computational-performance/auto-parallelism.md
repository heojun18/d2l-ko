# 자동 병렬화
:label:`sec_auto_para`


딥러닝 프레임워크(예: MXNet과 PyTorch)는 백엔드에서 자동으로 계산 그래프를 구성합니다.
계산 그래프를 사용하면 시스템이 모든 의존성을 인지할 수 있고,
서로 의존하지 않는 여러 작업을 선택적으로 병렬로 실행하여
속도를 향상시킬 수 있습니다. 예를 들어 :numref:`sec_async`의 :numref:`fig_asyncgraph`는 두 변수를 독립적으로 초기화합니다. 따라서 시스템은 이를 병렬로 실행하도록 선택할 수 있습니다.


일반적으로 단일 연산자는 모든 CPU나 단일 GPU의 모든 계산 리소스를 사용합니다. 예를 들어, `dot` 연산자는 단일 머신에 여러 개의 CPU 프로세서가 있더라도 모든 CPU의 모든 코어(및 스레드)를 사용합니다. 같은 원리가 단일 GPU에도 적용됩니다. 따라서 병렬화는 단일 디바이스 컴퓨터에서는 그다지 유용하지 않습니다. 여러 디바이스가 있을 때 더 중요해집니다. 병렬화는 일반적으로 여러 GPU 사이에서 가장 의미가 있지만, 로컬 CPU를 추가하면 성능이 약간 향상됩니다. 예를 들어 GPU와 CPU를 결합하여 컴퓨터 비전 모델을 학습시키는 데 초점을 맞춘 :citet:`Hadjis.Zhang.Mitliagkas.ea.2016`을 참조하십시오. 자동으로 병렬화하는 프레임워크의 편리함 덕분에 저희는 몇 줄의 Python 코드로 같은 목적을 달성할 수 있습니다. 더 넓게 보면, 자동 병렬 계산에 대한 저희의 논의는 CPU와 GPU를 모두 사용하는 병렬 계산뿐 아니라 계산과 통신의 병렬화에 초점을 맞춥니다.

이 절의 실험을 실행하려면 최소 두 개의 GPU가 필요하다는 점에 유의하십시오.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU에서의 병렬 계산

먼저 테스트를 위한 참조 워크로드를 정의해 보겠습니다. 아래의 `run` 함수는 `x_gpu1`과 `x_gpu2`라는 두 변수에 할당된 데이터를 사용하여 선택한 디바이스에서 10번의 행렬-행렬 곱셈을 수행합니다.

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
이제 함수를 데이터에 적용해 보겠습니다. 캐싱이 결과에 영향을 미치지 않도록 측정 전에 두 디바이스 중 하나에서 한 번 패스를 수행하여 디바이스를 워밍업합니다.
:end_tab:

:begin_tab:`pytorch`
이제 함수를 데이터에 적용해 보겠습니다. 캐싱이 결과에 영향을 미치지 않도록 측정 전에 두 디바이스 중 하나에서 한 번 패스를 수행하여 디바이스를 워밍업합니다. `torch.cuda.synchronize()`는 CUDA 디바이스의 모든 스트림에 있는 모든 커널이 완료될 때까지 기다립니다. 동기화해야 할 디바이스에 대한 `device` 인자를 받습니다. `device` 인자가 `None`(기본값)이면, `current_device()`로 주어진 현재 디바이스를 사용합니다.
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
두 작업 사이의 `waitall` 문을 제거하면 시스템은 두 디바이스에서의 계산을 자동으로 자유롭게 병렬화할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
두 작업 사이의 `synchronize` 문을 제거하면 시스템은 두 디바이스에서의 계산을 자동으로 자유롭게 병렬화할 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

위 경우 딥러닝 프레임워크가 사용자가 정교한 코드를 작성할 필요 없이 자동으로 두 GPU 디바이스에서 계산을 스케줄링하기 때문에 총 실행 시간은 각 부분의 합보다 짧습니다.



## 병렬 계산과 통신

많은 경우 저희는 서로 다른 디바이스 사이, 예를 들어 CPU와 GPU 사이, 혹은 서로 다른 GPU 사이에서 데이터를 이동시켜야 합니다.
예를 들어,
여러 가속기 카드에 걸쳐 그레이디언트를 집계해야 하는 분산 최적화를 수행하고자 할 때 이런 일이 발생합니다. GPU에서 계산한 다음 결과를 CPU로 다시 복사하는 것으로 이를 시뮬레이션해 보겠습니다.

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
이는 다소 비효율적입니다. 리스트의 나머지가 아직 계산되고 있는 동안 `y`의 일부를 이미 CPU로 복사하기 시작할 수 있다는 점에 유의하십시오. 이런 상황은 예를 들어 미니배치에 대한 그레이디언트를 계산할 때 발생합니다. 일부 파라미터의 그레이디언트는 다른 것보다 더 일찍 사용 가능해질 것입니다. 따라서 GPU가 여전히 실행되는 동안 PCI-Express 버스 대역폭을 사용하기 시작하는 것이 저희에게 이득이 됩니다. 두 부분 사이의 `waitall`을 제거하면 이 시나리오를 시뮬레이션할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
이는 다소 비효율적입니다. 리스트의 나머지가 아직 계산되고 있는 동안 `y`의 일부를 이미 CPU로 복사하기 시작할 수 있다는 점에 유의하십시오. 이런 상황은 예를 들어 미니배치에 대한 (역전파) 그레이디언트를 계산할 때 발생합니다. 일부 파라미터의 그레이디언트는 다른 것보다 더 일찍 사용 가능해질 것입니다. 따라서 GPU가 여전히 실행되는 동안 PCI-Express 버스 대역폭을 사용하기 시작하는 것이 저희에게 이득이 됩니다. PyTorch에서는 `to()`와 `copy_()` 같은 여러 함수가 명시적인 `non_blocking` 인자를 받아들이는데, 이를 통해 호출자가 동기화가 필요 없을 때 이를 우회할 수 있습니다. `non_blocking=True`로 설정하면 이 시나리오를 시뮬레이션할 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

두 연산에 필요한 총 시간은 (예상대로) 각 부분의 합보다 짧습니다.
이 작업은 다른 리소스인 CPU와 GPU 사이의 버스를 사용하기 때문에 병렬 계산과 다르다는 점에 유의하십시오. 실제로 저희는 두 디바이스에서 계산하고 동시에 통신할 수 있습니다. 위에서 언급했듯, 계산과 통신 사이에는 의존성이 있습니다. `y[i]`는 CPU로 복사되기 전에 계산되어야 합니다. 다행히도, 시스템은 `y[i]`를 계산하는 동안 `y[i-1]`을 복사하여 총 실행 시간을 줄일 수 있습니다.

저희는 :numref:`fig_twogpu`에 묘사된 것처럼, CPU와 두 개의 GPU에서 학습할 때 간단한 2층 MLP에 대한 계산 그래프와 그 의존성을 보여주는 것으로 마치겠습니다. 이로부터 발생하는 병렬 프로그램을 수동으로 스케줄링하는 것은 상당히 고통스러울 것입니다. 이것이 최적화를 위해 그래프 기반의 컴퓨팅 백엔드를 갖는 것이 이점이 되는 지점입니다.

![CPU와 두 개의 GPU에서 2층 MLP의 계산 그래프와 그 의존성.](../img/twogpu.svg)
:label:`fig_twogpu`


## 요약

* 현대 시스템은 여러 GPU와 CPU 같은 다양한 디바이스를 가지고 있습니다. 이들은 비동기적으로 병렬로 사용될 수 있습니다.
* 현대 시스템은 또한 PCI Express, 저장소(보통 솔리드 스테이트 드라이브 혹은 네트워크를 통한), 네트워크 대역폭 같은 통신을 위한 다양한 리소스를 가지고 있습니다. 이들은 최고 효율을 위해 병렬로 사용될 수 있습니다.
* 백엔드는 자동 병렬 계산과 통신을 통해 성능을 향상시킬 수 있습니다.

## 연습문제

1. 이 절에서 정의한 `run` 함수에서 8개의 연산이 수행되었습니다. 이들 사이에는 의존성이 없습니다. 딥러닝 프레임워크가 이들을 자동으로 병렬로 실행하는지 확인하는 실험을 설계해 보십시오.
1. 개별 연산자의 워크로드가 충분히 작을 때는 단일 CPU나 GPU에서도 병렬화가 도움이 될 수 있습니다. 이를 검증하는 실험을 설계해 보십시오.
1. CPU, GPU, 그리고 두 디바이스 간의 통신에서 병렬 계산을 사용하는 실험을 설계해 보십시오.
1. NVIDIA의 [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) 같은 디버거를 사용하여 코드가 효율적인지 확인해 보십시오.
1. 더 복잡한 데이터 의존성을 포함하는 계산 작업을 설계하고, 성능을 향상시키면서도 올바른 결과를 얻을 수 있는지 실험을 실행해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1681)
:end_tab:
