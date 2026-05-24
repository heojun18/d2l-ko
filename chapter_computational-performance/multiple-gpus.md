# 다중 GPU에서의 학습
:label:`sec_multi_gpu`

지금까지 저희는 CPU와 GPU에서 모델을 효율적으로 학습하는 방법을 논의했습니다. 심지어 :numref:`sec_auto_para`에서 딥러닝 프레임워크가 어떻게 그것들 사이에 계산과 통신을 자동으로 병렬화할 수 있게 해주는지도 보여드렸습니다. 또한 :numref:`sec_use_gpu`에서 `nvidia-smi` 명령을 사용해 컴퓨터에서 사용 가능한 모든 GPU를 나열하는 방법도 보여드렸습니다.
저희가 논의하지 *않은* 것은 실제로 딥러닝 학습을 어떻게 병렬화하는가입니다.
대신, 저희는 데이터를 어떤 식으로든 여러 디바이스에 걸쳐 나누어 동작하게 만들면 된다고 지나가며 암시했습니다. 본 절에서는 세부 사항을 채워넣고 처음부터 시작할 때 네트워크를 어떻게 병렬로 학습하는지 보여드립니다. 고수준 API의 기능을 활용하는 방법에 대한 세부 사항은 :numref:`sec_multi_gpu_concise`로 미룹니다.
저희는 여러분이 :numref:`sec_minibatch_sgd`에 설명된 것과 같은 미니배치 확률적 경사 하강법 알고리즘에 친숙하다고 가정합니다.


## 문제 분할하기

간단한 컴퓨터 비전 문제와 약간 구식인 네트워크부터 시작해 봅시다. 예를 들어, 컨볼루션과 풀링의 여러 층, 그리고 끝에 아마도 몇 개의 완전 연결 층이 있는 것 같은 네트워크입니다.
즉, LeNet :cite:`LeCun.Bottou.Bengio.ea.1998`이나 AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`과 상당히 비슷해 보이는 네트워크부터 시작해 봅시다.
여러 GPU(데스크탑 서버라면 2개, AWS g4dn.12xlarge 인스턴스에는 4개, p3.16xlarge에는 8개, 또는 p2.16xlarge에는 16개)가 주어지면, 저희는 좋은 속도 향상을 달성하면서도 단순하고 재현 가능한 설계 선택의 이점을 동시에 누리는 방식으로 학습을 분할하고자 합니다. 결국 여러 GPU는 *메모리*와 *계산* 능력을 모두 증가시킵니다. 요약하면, 분류하고자 하는 학습 데이터의 미니배치가 주어졌을 때 저희는 다음의 선택지를 가집니다.

첫째, 저희는 여러 GPU에 걸쳐 네트워크를 분할할 수 있습니다. 즉, 각 GPU는 특정 층으로 흐르는 데이터를 입력으로 받아 여러 후속 층에 걸쳐 데이터를 처리한 다음 데이터를 다음 GPU로 보냅니다.
이는 저희가 단일 GPU가 처리할 수 있는 것에 비해 더 큰 네트워크로 데이터를 처리할 수 있게 해줍니다.
또한,
GPU당 메모리 풋프린트를 잘 통제할 수 있습니다(이는 전체 네트워크 풋프린트의 일부입니다).

그러나, 층 간(따라서 GPU 간)의 인터페이스는 긴밀한 동기화를 요구합니다. 특히 층 간의 계산 워크로드가 적절히 일치하지 않으면 이는 까다로울 수 있습니다. 많은 수의 GPU에 대해 문제는 악화됩니다.
층 간의 인터페이스는 또한
활성값과 그레이디언트 같은
대량의 데이터 전송을 요구합니다.
이는 GPU 버스의 대역폭을 압도할 수 있습니다.
게다가, 계산 집약적이면서 순차적인 연산은 분할하기 사소하지 않습니다. 이와 관련한 최선의 노력에 대해서는 예를 들어 :citet:`Mirhoseini.Pham.Le.ea.2017`을 참조하십시오. 이는 여전히 어려운 문제이며, 사소하지 않은 문제에 대해 좋은(선형) 스케일링을 달성하는 것이 가능한지는 불분명합니다. 여러 GPU를 연결하기 위한 뛰어난 프레임워크나 운영체제 지원이 있지 않는 한 권장하지 않습니다.


둘째, 저희는 작업을 층별로 나눌 수 있습니다. 예를 들어, 단일 GPU에서 64개 채널을 계산하는 대신 저희는 문제를 4개의 GPU에 걸쳐 나눠 각각이 16개 채널에 대한 데이터를 생성하도록 할 수 있습니다.
마찬가지로, 완전 연결 층에 대해 저희는 출력 유닛의 수를 나눌 수 있습니다. (:citet:`Krizhevsky.Sutskever.Hinton.2012`에서 가져온) :numref:`fig_alexnet_original`
은 이 설계를 보여주는데, 여기서 이 전략은 매우 작은 메모리 풋프린트(당시 2 GB)를 가진 GPU를 다루기 위해 사용되었습니다.
이는 채널(또는 유닛)의 수가 너무 작지 않다면 계산 측면에서 좋은 스케일링을 가능하게 합니다.
또한,
사용 가능한 메모리가 선형으로 스케일되므로 여러 GPU는 점점 더 큰 네트워크를 처리할 수 있습니다.

![제한된 GPU 메모리 때문에 원래의 AlexNet 설계에서의 모델 병렬화.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

그러나,
각 층이 다른 모든 층의 결과에 의존하기 때문에 저희는 *매우 많은* 수의 동기화 또는 배리어 연산이 필요합니다.
게다가, 전송되어야 하는 데이터의 양은 잠재적으로 GPU에 걸쳐 층을 분산시킬 때보다 훨씬 더 많습니다. 따라서 대역폭 비용과 복잡성 때문에 저희는 이 접근법을 권장하지 않습니다.

마지막으로, 저희는 여러 GPU에 걸쳐 데이터를 분할할 수 있습니다. 이 방식으로 모든 GPU가 다른 관측치에 대해서이긴 하지만 같은 종류의 작업을 수행합니다. 그레이디언트는 학습 데이터의 각 미니배치 후에 GPU에 걸쳐 집계됩니다.
이것은 가장 간단한 접근법이며 어떤 상황에서도 적용될 수 있습니다.
저희는 각 미니배치 후에만 동기화하면 됩니다. 그렇긴 해도, 다른 것들이 여전히 계산되고 있는 동안 그레이디언트 파라미터를 이미 교환하기 시작하는 것이 매우 바람직합니다.
게다가, 더 많은 수의 GPU는 더 큰 미니배치 크기로 이어져 학습 효율성을 증가시킵니다.
그러나, 더 많은 GPU를 추가한다고 해서 더 큰 모델을 학습할 수 있게 되는 것은 아닙니다.


![다중 GPU에서의 병렬화. 왼쪽에서 오른쪽으로: 원래 문제, 네트워크 분할, 층별 분할, 데이터 병렬화.](../img/splitting.svg)
:label:`fig_splitting`


다중 GPU에서의 서로 다른 병렬화 방식의 비교가 :numref:`fig_splitting`에 묘사되어 있습니다.
대체로, 충분히 큰 메모리를 가진 GPU에 접근할 수 있다면 데이터 병렬화가 진행하기 가장 편리한 방법입니다. 분산 학습을 위한 분할의 자세한 설명은 :cite:`Li.Andersen.Park.ea.2014`도 참조하십시오. GPU 메모리는 딥러닝 초기에는 문제였습니다. 지금쯤이면 이 이슈는 가장 특이한 경우를 제외하고는 모두 해결되었습니다. 다음에서 저희는 데이터 병렬화에 초점을 맞춥니다.

## 데이터 병렬화

머신에 $k$개의 GPU가 있다고 가정합시다. 학습할 모델이 주어지면, 각 GPU는 모델 파라미터의 완전한 집합을 독립적으로 유지하지만 GPU에 걸친 파라미터 값들은 동일하고 동기화됩니다.
예로,
:numref:`fig_data_parallel`은
$k=2$일 때
데이터 병렬화로 학습하는 것을 보여줍니다.


![두 GPU에서 데이터 병렬화를 사용한 미니배치 확률적 경사 하강법의 계산.](../img/data-parallel.svg)
:label:`fig_data_parallel`

일반적으로, 학습은 다음과 같이 진행됩니다.

* 학습의 어떤 반복에서든, 랜덤 미니배치가 주어지면 저희는 배치의 예제를 $k$개의 부분으로 나누고 GPU에 걸쳐 균등하게 분산시킵니다.
* 각 GPU는 자신에게 할당된 미니배치 부분 집합에 기반하여 모델 파라미터의 손실과 그레이디언트를 계산합니다.
* $k$개의 GPU 각각의 지역 그레이디언트가 집계되어 현재의 미니배치 확률적 그레이디언트를 얻습니다.
* 집계된 그레이디언트는 각 GPU로 재분배됩니다.
* 각 GPU는 이 미니배치 확률적 그레이디언트를 사용해 자신이 유지하는 모델 파라미터의 완전한 집합을 업데이트합니다.




실제로는 저희가 $k$개의 GPU에서 학습할 때 미니배치 크기를 $k$배 *증가시켜* 각 GPU가 단지 단일 GPU에서 학습하는 것과 같은 양의 작업을 하도록 한다는 점에 유의하십시오. 16-GPU 서버에서 이는 미니배치 크기를 상당히 증가시킬 수 있고 저희는 그에 따라 학습률을 증가시켜야 할 수도 있습니다.
또한 :numref:`sec_batch_norm`의 배치 정규화는 예를 들어 GPU별로 별도의 배치 정규화 계수를 유지함으로써 조정되어야 한다는 점에 유의하십시오.
다음에서 저희는 다중 GPU 학습을 보여주기 위해 간단한 네트워크를 사용할 것입니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**간단한 네트워크**]

저희는 :numref:`sec_lenet`에서 소개된 LeNet을 (약간의 수정과 함께) 사용합니다. 파라미터 교환과 동기화를 자세히 보여주기 위해 처음부터 정의합니다.

```{.python .input}
#@tab mxnet
# Initialize model parameters
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## 데이터 동기화

효율적인 다중 GPU 학습을 위해 저희는 두 가지 기본 연산이 필요합니다.
먼저 [**파라미터 리스트를 여러 디바이스에 분산**]시키고 그레이디언트를 부착(`get_params`)할 수 있는 능력이 필요합니다. 파라미터 없이는 GPU에서 네트워크를 평가할 수 없습니다.
둘째, 저희는 여러 디바이스에 걸쳐 파라미터를 합할 수 있는 능력이 필요합니다, 즉 `allreduce` 함수가 필요합니다.

```{.python .input}
#@tab mxnet
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

모델 파라미터를 하나의 GPU로 복사하여 시도해 봅시다.

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

저희가 아직 어떤 계산도 수행하지 않았기 때문에, 편향 파라미터에 대한 그레이디언트는 여전히 0입니다.
이제 저희가 여러 GPU에 걸쳐 분산된 벡터를 가지고 있다고 가정합시다. 다음의 [**`allreduce` 함수는 모든 벡터를 합하고 결과를 모든 GPU로 다시 브로드캐스트합니다**]. 이것이 동작하려면 저희는 결과를 누적하는 디바이스로 데이터를 복사해야 한다는 점에 유의하십시오.

```{.python .input}
#@tab mxnet
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

서로 다른 디바이스에서 서로 다른 값을 가진 벡터를 생성하고 집계하여 이를 테스트해 봅시다.

```{.python .input}
#@tab mxnet
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## 데이터 분산

저희는 [**미니배치를 여러 GPU에 걸쳐 균등하게 분산**]시키는 간단한 유틸리티 함수가 필요합니다. 예를 들어, 두 GPU에서 저희는 데이터의 절반을 각 GPU로 복사하고자 합니다.
더 편리하고 더 간결하기 때문에, 저희는 $4 \times 5$ 행렬에서 그것을 시도하기 위해 딥러닝 프레임워크의 내장 함수를 사용합니다.

```{.python .input}
#@tab mxnet
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

이후의 재사용을 위해 저희는 데이터와 레이블을 모두 분할하는 `split_batch` 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 학습

이제 저희는 [**단일 미니배치에서의 다중 GPU 학습**]을 구현할 수 있습니다. 그 구현은 주로 이 절에서 설명된 데이터 병렬화 접근법에 기반합니다. 저희는 방금 논의한 보조 함수 `allreduce`와 `split_and_load`를 사용해 여러 GPU 사이에서 데이터를 동기화할 것입니다. 병렬화를 달성하기 위해 어떤 특정 코드도 작성할 필요가 없다는 점에 유의하십시오. 계산 그래프가 미니배치 내에서 디바이스에 걸친 어떤 의존성도 가지지 않기 때문에, *자동으로* 병렬로 실행됩니다.

```{.python .input}
#@tab mxnet
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

이제 [**학습 함수**]를 정의할 수 있습니다. 이는 이전 챕터에서 사용된 것과 약간 다릅니다. 저희는 GPU를 할당하고 모든 모델 파라미터를 모든 디바이스로 복사해야 합니다.
분명히 각 배치는 여러 GPU를 다루기 위해 `train_batch` 함수를 사용하여 처리됩니다. 편의(및 코드의 간결함)를 위해 저희는 단일 GPU에서 정확도를 계산하지만, 다른 GPU들이 유휴 상태이기 때문에 이는 *비효율적*입니다.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

[**단일 GPU에서**] 이것이 얼마나 잘 동작하는지 봅시다.
먼저 배치 크기 256과 학습률 0.2를 사용합니다.

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

배치 크기와 학습률을 변경하지 않고 [**GPU의 수를 2로 증가시키면**], 이전 실험과 비교하여 테스트 정확도가 대략 같게 유지됨을
볼 수 있습니다.
최적화 알고리즘 측면에서, 그것들은 동일합니다. 불행히도 여기서 얻을 수 있는 의미 있는 속도 향상은 없습니다. 모델이 단순히 너무 작습니다. 게다가 저희는 작은 데이터셋만을 가지고 있는데, 여기서 다중 GPU 학습을 구현하는 다소 정교하지 않은 저희의 접근법은 상당한 Python 오버헤드를 겪었습니다. 저희는 앞으로 더 복잡한 모델과 더 정교한 병렬화 방식을 만날 것입니다.
그럼에도 불구하고 Fashion-MNIST에 대해 무슨 일이 일어나는지 봅시다.

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## 요약

* 여러 GPU에 걸쳐 심층 네트워크 학습을 나누는 여러 방법이 있습니다. 저희는 그것들을 층 사이에서, 층에 걸쳐, 또는 데이터에 걸쳐 나눌 수 있습니다. 전자의 두 가지는 긴밀하게 조율된 데이터 전송을 요구합니다. 데이터 병렬화가 가장 단순한 전략입니다.
* 데이터 병렬 학습은 직관적입니다. 그러나 효율적이기 위해 유효 미니배치 크기를 증가시킵니다.
* 데이터 병렬화에서 데이터는 여러 GPU에 걸쳐 나뉘며, 각 GPU는 자체 순방향 및 역방향 연산을 실행하고 이후 그레이디언트가 집계되며 결과가 GPU로 다시 브로드캐스트됩니다.
* 더 큰 미니배치에 대해 약간 증가된 학습률을 사용할 수 있습니다.

## 연습문제

1. $k$개의 GPU에서 학습할 때, 미니배치 크기를 $b$에서 $k \cdot b$로 변경하십시오, 즉 GPU의 수만큼 스케일업하십시오.
1. 서로 다른 학습률에 대한 정확도를 비교하십시오. GPU의 수에 따라 어떻게 스케일됩니까?
1. 서로 다른 GPU에서 서로 다른 파라미터를 집계하는 더 효율적인 `allreduce` 함수를 구현해 보십시오. 왜 더 효율적입니까?
1. 다중 GPU 테스트 정확도 계산을 구현해 보십시오.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1669)
:end_tab:
