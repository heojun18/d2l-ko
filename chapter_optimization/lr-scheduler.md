# 학습률 스케줄링(Learning Rate Scheduling)
:label:`sec_scheduler`

지금까지 저희는 주로 가중치 벡터를 어떻게 업데이트할지에 대한 최적화 *알고리즘*에 초점을 맞췄지, 그것들이 업데이트되는 *속도*에 초점을 맞추지는 않았습니다. 그럼에도 불구하고, 학습률을 조정하는 것은 종종 실제 알고리즘만큼이나 중요합니다. 고려해야 할 여러 측면이 있습니다.

* 가장 명백하게는 학습률의 *크기*가 중요합니다. 너무 크면 최적화가 발산하고, 너무 작으면 학습에 너무 오래 걸리거나 차선의 결과로 끝나게 됩니다. 저희는 이전에 문제의 조건수가 중요함을 보았습니다(자세한 내용은 예를 들어 :numref:`sec_momentum` 참조). 직관적으로 이는 가장 둔감한 방향에서의 변화량과 가장 민감한 방향에서의 변화량의 비율입니다.
* 둘째로, 감쇠 속도도 똑같이 중요합니다. 학습률이 크게 유지되면 저희는 단순히 최솟값 주변에서 튀어다닐 수 있고 따라서 최적성에 도달하지 못합니다. :numref:`sec_minibatch_sgd`는 이를 다소 자세히 논의했고 저희는 :numref:`sec_sgd`에서 성능 보장을 분석했습니다. 요컨대, 저희는 속도가 감쇠하되, 볼록 문제에 좋은 선택인 $\mathcal{O}(t^{-\frac{1}{2}})$보다는 아마도 더 천천히 감쇠하기를 원합니다.
* 똑같이 중요한 또 다른 측면은 *초기화*입니다. 이는 파라미터가 초기에 어떻게 설정되는지(자세한 내용은 :numref:`sec_numerical_stability` 검토)와 또한 그것들이 초기에 어떻게 진화하는지에 관련됩니다. 이는 *워밍업*이라는 별명으로 통하는데, 즉 저희가 초기에 얼마나 빠르게 해를 향해 움직이기 시작하는지를 말합니다. 특히 초기 파라미터 집합이 무작위이므로 초반의 큰 스텝은 유익하지 않을 수 있습니다. 초기 업데이트 방향도 꽤 무의미할 수 있습니다.
* 마지막으로, 주기적 학습률 조정을 수행하는 여러 최적화 변종이 있습니다. 이는 현재 장의 범위를 벗어납니다. 저희는 독자가 :citet:`Izmailov.Podoprikhin.Garipov.ea.2018`에서 자세한 내용을 검토할 것을 권장합니다. 예를 들어, 파라미터의 전체 *경로*에 걸쳐 평균화하여 더 나은 해를 얻는 방법입니다.

학습률을 관리하는 데 많은 세부사항이 필요하다는 사실을 고려할 때, 대부분의 딥러닝 프레임워크에는 이를 자동으로 처리하기 위한 도구가 있습니다. 현재 장에서 저희는 다양한 스케줄이 정확도에 미치는 영향을 검토하고 *학습률 스케줄러*를 통해 이를 어떻게 효율적으로 관리할 수 있는지도 보일 것입니다.

## 장난감 문제

저희는 쉽게 계산할 수 있을 만큼 저렴하지만 핵심 측면 일부를 설명하기에 충분히 비자명한 장난감 문제로 시작합니다. 이를 위해 Fashion-MNIST에 적용된 LeNet의 약간 현대화된 버전(`sigmoid` 활성화 대신 `relu`, AveragePooling 대신 MaxPooling)을 선택합니다. 더욱이, 성능을 위해 네트워크를 하이브리드화합니다. 대부분의 코드가 표준이므로 저희는 더 자세한 논의 없이 기본만 소개합니다. 필요에 따라 :numref:`chap_cnn`을 복습용으로 참고하세요.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

학습률 $0.3$과 같은 기본 설정으로 이 알고리즘을 호출하고 $30$번의 반복 동안 학습하면 어떤 일이 일어나는지 살펴봅시다. 학습 정확도가 어떻게 계속 증가하는 동안 테스트 정확도 측면에서의 진전은 어느 시점을 넘어서면 멈추는지 주목하세요. 두 곡선 사이의 간격은 과적합을 나타냅니다.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## 스케줄러

학습률을 조정하는 한 가지 방법은 각 단계에서 그것을 명시적으로 설정하는 것입니다. 이는 `set_learning_rate` 메서드에 의해 편리하게 달성됩니다. 저희는 매 에포크 후(또는 심지어 매 미니배치 후)에 그것을 아래로 조정할 수 있습니다. 예를 들어, 최적화가 어떻게 진행되는지에 반응하여 동적인 방식으로 말이죠.

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

더 일반적으로 저희는 스케줄러를 정의하고 싶습니다. 업데이트 수와 함께 호출되면 학습률의 적절한 값을 반환합니다. 학습률을 $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$로 설정하는 간단한 것을 정의해 봅시다.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

다양한 값의 범위에 걸쳐 그 동작을 그려 봅시다.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

이제 Fashion-MNIST에서의 학습에 대해 이것이 어떻게 진행되는지 봅시다. 저희는 단지 학습 알고리즘에 추가 인자로 스케줄러를 제공합니다.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

이는 이전보다 꽤 잘 작동했습니다. 두 가지가 두드러집니다. 곡선이 이전보다 다소 더 평활했습니다. 둘째, 과적합이 덜 했습니다. 안타깝게도 어떤 전략이 *이론적으로* 왜 더 적은 과적합으로 이어지는지에 대해서는 잘 해결된 질문이 아닙니다. 더 작은 스텝 크기는 0에 더 가깝고 따라서 더 단순한 파라미터로 이어질 것이라는 주장이 있습니다. 그러나 저희가 사실 일찍 중단하지 않고 단순히 학습률을 부드럽게 줄이기만 하므로 이는 현상을 완전히 설명하지 못합니다.

## 정책

학습률 스케줄러의 전체 다양성을 다 다룰 수는 없지만, 저희는 아래에서 인기 있는 정책의 간단한 개요를 제공하려고 시도합니다. 일반적인 선택은 다항 감쇠와 조각별 상수 스케줄입니다. 그 외에도, 코사인 학습률 스케줄이 일부 문제에서 경험적으로 잘 작동하는 것으로 밝혀졌습니다. 마지막으로, 일부 문제에서는 큰 학습률을 사용하기 전에 옵티마이저를 워밍업하는 것이 유익합니다.

### 인수 스케줄러(Factor Scheduler)

다항 감쇠에 대한 한 가지 대안은 곱셈적인 것일 수 있습니다. 즉, $\alpha \in (0, 1)$에 대해 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$입니다. 학습률이 합리적인 하한 아래로 감쇠하는 것을 막기 위해 업데이트 방정식은 종종 $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$로 수정됩니다.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

이는 또한 `lr_scheduler.FactorScheduler` 객체를 통해 MXNet의 내장 스케줄러에 의해 달성될 수 있습니다. 이는 워밍업 기간, 워밍업 모드(선형 또는 상수), 원하는 최대 업데이트 수 등과 같은 몇 가지 더 많은 파라미터를 받습니다. 앞으로 저희는 적절한 경우 내장 스케줄러를 사용하고 여기서는 그 기능만 설명합니다. 보여진 바와 같이, 필요할 경우 자신만의 스케줄러를 구축하는 것은 꽤 직관적입니다.

### 다중 인수 스케줄러(Multi Factor Scheduler)

딥 네트워크를 학습시키기 위한 일반적인 전략은 학습률을 조각별 상수로 유지하고 가끔씩 주어진 양만큼 감소시키는 것입니다. 즉, $s = \{5, 10, 20\}$와 같이 속도를 감소시킬 시점들의 집합이 주어졌을 때, $t \in s$일 때마다 $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$를 감소시킵니다. 각 단계에서 값이 반감된다고 가정하면 다음과 같이 구현할 수 있습니다.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

이 조각별 상수 학습률 스케줄 뒤의 직관은 가중치 벡터의 분포 측면에서 정상점에 도달할 때까지 최적화가 진행되도록 한다는 것입니다. 그런 다음에야(그리고 그 후에만) 좋은 지역 최솟값에 대한 더 높은 품질의 대리를 얻기 위해 속도를 감소시킵니다. 아래 예는 이것이 어떻게 점점 조금씩 더 나은 해를 만들 수 있는지 보여줍니다.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 코사인 스케줄러

:citet:`Loshchilov.Hutter.2016`에 의해 다소 당혹스러운 휴리스틱이 제안되었습니다. 이는 처음에는 학습률을 너무 급격하게 줄이고 싶지 않을 수도 있고, 더욱이 마지막에는 매우 작은 학습률을 사용해 해를 "정제"하고 싶을 수도 있다는 관찰에 의존합니다. 이는 범위 $t \in [0, T]$에서 학습률에 대해 다음 함수 형태를 가진 코사인 형 스케줄을 결과로 가져옵니다.

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


여기서 $\eta_0$는 초기 학습률, $\eta_T$는 시간 $T$에서의 목표 속도입니다. 더욱이, $t > T$의 경우 저희는 단순히 값을 다시 증가시키지 않고 $\eta_T$로 고정합니다. 다음 예에서 저희는 최대 업데이트 단계 $T = 20$을 설정합니다.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

컴퓨터 비전의 맥락에서 이 스케줄은 개선된 결과로 이어질 *수* 있습니다. 그러나 그러한 개선이 보장되지 않는다는 점에 유의하세요(아래에서 볼 수 있듯이).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 워밍업(Warmup)

어떤 경우에는 파라미터를 초기화하는 것만으로 좋은 해를 보장하기에 충분하지 않습니다. 이는 특히 불안정한 최적화 문제로 이어질 수 있는 일부 고급 네트워크 설계에 문제가 됩니다. 저희는 처음에 발산을 막기 위해 충분히 작은 학습률을 선택함으로써 이를 해결할 수 있습니다. 안타깝게도 이는 진전이 느리다는 것을 의미합니다. 반대로, 초기에 큰 학습률은 발산으로 이어집니다.

이 딜레마에 대한 다소 간단한 해결책은 학습률이 초기 최대치로 *증가*하는 워밍업 기간을 사용하고 최적화 과정 끝까지 속도를 식히는 것입니다. 단순성을 위해 일반적으로 이 목적을 위해 선형 증가를 사용합니다. 이는 아래에 표시된 형태의 스케줄로 이어집니다.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

네트워크가 초기에 더 잘 수렴함을 주목하세요(특히 첫 5 에포크 동안의 성능을 관찰하세요).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

워밍업은 코사인뿐만 아니라 임의의 스케줄러에 적용될 수 있습니다. 학습률 스케줄과 훨씬 많은 실험에 대한 더 자세한 논의는 :cite:`Gotmare.Keskar.Xiong.ea.2018`도 참조하세요. 특히 그들은 워밍업 단계가 매우 깊은 네트워크에서 파라미터의 발산 양을 제한한다는 것을 발견했습니다. 이는 직관적으로 말이 되는데, 초기에 진전을 이루는 데 가장 오래 걸리는 네트워크 부분에서 무작위 초기화로 인한 상당한 발산을 예상할 수 있기 때문입니다.

## 요약

* 학습 중에 학습률을 감소시키는 것은 개선된 정확도와 (가장 당혹스럽게도) 모델의 감소된 과적합으로 이어질 수 있습니다.
* 진전이 정체될 때마다 학습률의 조각별 감소는 실제로 효과적입니다. 본질적으로 이는 저희가 적절한 해로 효율적으로 수렴하도록 보장한 다음에만 학습률을 줄임으로써 파라미터의 본질적인 분산을 줄이도록 보장합니다.
* 코사인 스케줄러는 일부 컴퓨터 비전 문제에 인기가 있습니다. 그러한 스케줄러의 세부 사항은 예를 들어 [GluonCV](http://gluon-cv.mxnet.io)를 참조하세요.
* 최적화 전 워밍업 기간은 발산을 막을 수 있습니다.
* 최적화는 딥러닝에서 여러 목적을 수행합니다. 학습 목적을 최소화하는 것 외에도, 최적화 알고리즘과 학습률 스케줄링의 다양한 선택은 테스트 세트에서 (동일한 양의 학습 오차에 대해) 상당히 다른 양의 일반화와 과적합으로 이어질 수 있습니다.

## 연습문제

1. 주어진 고정 학습률에 대해 최적화 동작을 실험해 보세요. 이 방식으로 얻을 수 있는 최상의 모델은 무엇입니까?
1. 학습률의 감소의 지수를 변경하면 수렴이 어떻게 변하나요? 실험에서 편의를 위해 `PolyScheduler`를 사용하세요.
1. 코사인 스케줄러를 ImageNet 학습과 같은 대규모 컴퓨터 비전 문제에 적용해 보세요. 다른 스케줄러에 비해 성능에 어떻게 영향을 미치나요?
1. 워밍업은 얼마나 오래 지속되어야 하나요?
1. 최적화와 샘플링을 연결할 수 있나요? :citet:`Welling.Teh.2011`의 확률적 경사 랑주뱅 동역학에 대한 결과부터 사용해 시작하세요.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:
