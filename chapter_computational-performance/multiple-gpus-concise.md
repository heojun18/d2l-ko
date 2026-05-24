# 다중 GPU의 간결한 구현
:label:`sec_multi_gpu_concise`

새로운 모델마다 처음부터 병렬화를 구현하는 것은 재미있는 일이 아닙니다. 게다가 고성능을 위해 동기화 도구를 최적화하는 것에는 상당한 이점이 있습니다. 다음에서 저희는 딥러닝 프레임워크의 고수준 API를 사용해 이를 어떻게 수행하는지 보여드리겠습니다.
수학과 알고리즘은 :numref:`sec_multi_gpu`와 동일합니다.
별로 놀랍지 않게도 이 절의 코드를 실행하려면 최소 두 개의 GPU가 필요합니다.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**간단한 네트워크**]

여전히 학습하기 충분히 쉽고 빠른, :numref:`sec_multi_gpu`의 LeNet보다는 약간 더 의미 있는 네트워크를 사용해 보겠습니다.
ResNet-18의 변형 :cite:`He.Zhang.Ren.ea.2016`을 선택합니다. 입력 이미지가 작으므로 약간 수정합니다. 특히, :numref:`sec_resnet`과의 차이는 시작 부분에서 더 작은 컨볼루션 커널, 스트라이드, 패딩을 사용한다는 점입니다.
또한, 최대 풀링 층을 제거합니다.

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## 네트워크 초기화

:begin_tab:`mxnet`
`initialize` 함수는 저희가 선택한 디바이스에서 파라미터를 초기화할 수 있게 해줍니다.
초기화 방법에 대한 복습은 :numref:`sec_numerical_stability`를 참조하십시오. 특히 편리한 점은 *여러* 디바이스에서 동시에 네트워크를 초기화할 수도 있다는 것입니다. 실제로 이것이 어떻게 동작하는지 시도해 봅시다.
:end_tab:

:begin_tab:`pytorch`
저희는 학습 루프 내부에서 네트워크를 초기화할 것입니다.
초기화 방법에 대한 복습은 :numref:`sec_numerical_stability`를 참조하십시오.
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
:numref:`sec_multi_gpu`에서 소개된 `split_and_load` 함수를 사용하여 데이터의 미니배치를 나누고 일부를 `devices` 변수에서 제공된 디바이스 리스트로 복사할 수 있습니다. 네트워크 인스턴스는 *자동으로* 순전파의 값을 계산하기 위해 적절한 GPU를 사용합니다. 여기서 저희는 4개의 관측치를 생성하고 GPU에 걸쳐 나눕니다.
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
일단 데이터가 네트워크를 통과하면, 대응하는 파라미터는 *데이터가 통과한 디바이스에서* 초기화됩니다.
이는 초기화가 디바이스 단위로 일어남을 의미합니다. 저희가 초기화에 GPU 0과 GPU 1을 선택했으므로, 네트워크는 거기에서만 초기화되며 CPU에서는 초기화되지 않습니다. 사실, 파라미터는 CPU에 존재하지조차 않습니다. 저희는 파라미터를 출력하고 발생할 수 있는 오류를 관찰함으로써 이를 검증할 수 있습니다.
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
다음으로, [**정확도를 평가**]하는 코드를 (**여러 디바이스에 걸쳐 병렬로**) 동작하는 것으로 대체해 보겠습니다. 이는 :numref:`sec_lenet`의 `evaluate_accuracy_gpu` 함수를 대체하는 역할을 합니다. 주요 차이는 네트워크를 호출하기 전에 미니배치를 나눈다는 것입니다. 다른 모든 것은 본질적으로 동일합니다.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**학습**]

이전처럼, 효율적인 병렬화를 위해 학습 코드는 몇 가지 기본 기능을 수행해야 합니다.

* 네트워크 파라미터는 모든 디바이스에 걸쳐 초기화되어야 합니다.
* 데이터셋을 순회하는 동안 미니배치는 모든 디바이스에 걸쳐 나뉘어야 합니다.
* 저희는 디바이스에 걸쳐 병렬로 손실과 그 그레이디언트를 계산합니다.
* 그레이디언트가 집계되고 파라미터가 그에 따라 업데이트됩니다.

마지막에 저희는 네트워크의 최종 성능을 보고하기 위해 (다시 병렬로) 정확도를 계산합니다. 학습 루틴은 데이터를 나누고 집계해야 한다는 점을 제외하면 이전 챕터의 구현과 상당히 유사합니다.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

실제로 이것이 어떻게 동작하는지 봅시다. 워밍업으로 [**단일 GPU에서 네트워크를 학습시킵니다.**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

다음으로 [**학습에 2개의 GPU를 사용**]합니다. :numref:`sec_multi_gpu`에서 평가된 LeNet과 비교하면,
ResNet-18 모델은 상당히 더 복잡합니다. 이것이 병렬화가 그 이점을 보여주는 지점입니다. 계산 시간이 파라미터 동기화 시간보다 의미 있게 큽니다. 이는 병렬화의 오버헤드가 덜 관련 있게 되기 때문에 확장성을 향상시킵니다.

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## 요약

:begin_tab:`mxnet`
* Gluon은 컨텍스트 리스트를 제공함으로써 여러 디바이스에 걸친 모델 초기화를 위한 프리미티브를 제공합니다.
:end_tab:
* 데이터는 데이터가 발견된 디바이스에서 자동으로 평가됩니다.
* 디바이스의 파라미터에 접근하기 전에 각 디바이스에서 네트워크를 초기화하는 데 주의하십시오. 그렇지 않으면 오류를 만나게 될 것입니다.
* 최적화 알고리즘은 자동으로 여러 GPU에 걸쳐 집계합니다.



## 연습문제

:begin_tab:`mxnet`
1. 이 절에서는 ResNet-18을 사용합니다. 서로 다른 에포크, 배치 크기, 학습률을 시도해 보십시오. 계산에 더 많은 GPU를 사용하십시오. 16개의 GPU(예: AWS p2.16xlarge 인스턴스)로 이를 시도하면 어떻게 됩니까?
1. 때때로 서로 다른 디바이스는 서로 다른 컴퓨팅 파워를 제공합니다. 저희는 GPU와 CPU를 동시에 사용할 수 있습니다. 작업을 어떻게 나눠야 합니까? 그만한 노력의 가치가 있습니까? 왜입니까? 왜 아닙니까?
1. `npx.waitall()`을 제거하면 어떻게 됩니까? 병렬화를 위해 최대 두 단계의 오버랩을 갖도록 학습을 어떻게 수정하시겠습니까?
:end_tab:

:begin_tab:`pytorch`
1. 이 절에서는 ResNet-18을 사용합니다. 서로 다른 에포크, 배치 크기, 학습률을 시도해 보십시오. 계산에 더 많은 GPU를 사용하십시오. 16개의 GPU(예: AWS p2.16xlarge 인스턴스)로 이를 시도하면 어떻게 됩니까?
1. 때때로 서로 다른 디바이스는 서로 다른 컴퓨팅 파워를 제공합니다. 저희는 GPU와 CPU를 동시에 사용할 수 있습니다. 작업을 어떻게 나눠야 합니까? 그만한 노력의 가치가 있습니까? 왜입니까? 왜 아닙니까?
:end_tab:



:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1403)
:end_tab:
