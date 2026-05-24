# 생성적 적대 신경망
:label:`sec_basic_gan`

이 책의 대부분에서 저희는 예측을 어떻게 만드는지에 대해 이야기해 왔습니다. 어떤 형태로든, 저희는 데이터 예제에서 레이블로의 매핑을 학습하기 위해 심층 신경망을 사용했습니다. 이러한 종류의 학습은 판별 학습(discriminative learning)이라고 불립니다. 예를 들어 저희는 고양이 사진과 개 사진을 구분할 수 있기를 원합니다. 분류기와 회귀기 모두 판별 학습의 예입니다. 그리고 역전파로 훈련된 신경망은 대규모의 복잡한 데이터셋에서 판별 학습에 대해 저희가 알고 있다고 생각했던 모든 것을 뒤집어 놓았습니다. 고해상도 이미지에 대한 분류 정확도는 단 5~6년 만에 쓸모없는 수준에서 (몇 가지 단서를 붙이면) 인간 수준에 이르게 되었습니다. 심층 신경망이 놀라울 정도로 잘 수행하는 다른 모든 판별 작업에 대한 또 다른 장황한 설명은 생략하겠습니다.

하지만 머신러닝에는 판별 작업을 푸는 것 이상의 것이 있습니다. 예를 들어, 레이블이 없는 큰 데이터셋이 주어졌을 때, 저희는 이 데이터의 특성을 간결하게 포착하는 모델을 학습하고 싶을 수 있습니다. 이러한 모델이 주어지면, 저희는 훈련 데이터의 분포와 닮은 합성 데이터 예제를 샘플링할 수 있습니다. 예를 들어, 얼굴 사진의 대규모 코퍼스가 주어졌을 때, 저희는 마치 같은 데이터셋에서 나온 것처럼 보이는 새로운 사실적인 이미지를 생성할 수 있기를 원할 수 있습니다. 이러한 종류의 학습을 생성 모델링(generative modeling)이라고 합니다.

최근까지 저희에게는 새로운 사실적인 이미지를 합성할 수 있는 방법이 없었습니다. 하지만 판별 학습을 위한 심층 신경망의 성공은 새로운 가능성을 열었습니다. 지난 3년간 큰 추세 중 하나는 일반적으로 지도 학습 문제로 생각하지 않는 문제에서의 도전 과제를 극복하기 위해 판별 심층망을 적용하는 것이었습니다. 순환 신경망 언어 모델은 한 번 훈련되면 생성 모델로 작동할 수 있는 판별 신경망(다음 문자를 예측하도록 훈련된)을 사용하는 한 가지 예입니다.

2014년, 한 획기적인 논문이 좋은 생성 모델을 얻기 위해 판별 모델의 힘을 활용하는 영리하고 새로운 방법인 생성적 적대 신경망(Generative adversarial networks, GANs) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`을 소개했습니다. 본질적으로 GAN은 데이터 생성기가 가짜 데이터와 진짜 데이터를 구분할 수 없다면 그 생성기가 좋은 것이라는 아이디어에 의존합니다. 통계학에서 이것은 이표본 검정(two-sample test)이라고 불립니다. 데이터셋 $X=\{x_1,\ldots, x_n\}$과 $X'=\{x'_1,\ldots, x'_n\}$이 같은 분포에서 추출되었는지 묻는 검정입니다. 대부분의 통계 논문과 GAN의 주요 차이점은 후자가 이 아이디어를 구성적 방식으로 사용한다는 것입니다. 다시 말해, 단지 "이봐, 이 두 데이터셋은 같은 분포에서 나온 것처럼 보이지 않아"라고 말하는 모델을 훈련시키는 것이 아니라, [이표본 검정](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing)을 사용하여 생성 모델에 훈련 신호를 제공합니다. 이를 통해 저희는 데이터 생성기가 실제 데이터와 닮은 것을 생성할 때까지 개선할 수 있습니다. 적어도 이는 저희의 분류기가 최첨단 심층 신경망이라 하더라도 그 분류기를 속여야 합니다.

![생성적 적대 신경망](../img/gan.svg)
:label:`fig_gan`


GAN 아키텍처는 :numref:`fig_gan`에 설명되어 있습니다.
보시다시피, GAN 아키텍처에는 두 가지 부분이 있습니다. 첫째, 저희는 실제와 똑같이 보이는 데이터를 잠재적으로 생성할 수 있는 장치(예를 들면 심층망일 수도 있지만, 게임 렌더링 엔진과 같이 사실상 무엇이든 될 수 있습니다)가 필요합니다. 이미지를 다루고 있다면, 이는 이미지를 생성해야 합니다. 음성을 다루고 있다면, 오디오 시퀀스를 생성해야 하는 식입니다. 저희는 이것을 생성기(generator) 신경망이라고 부릅니다. 두 번째 구성 요소는 판별기(discriminator) 신경망입니다. 이는 가짜와 진짜 데이터를 서로 구별하려고 시도합니다. 두 신경망은 서로 경쟁합니다. 생성기 신경망은 판별기 신경망을 속이려고 시도합니다. 그 시점에서 판별기 신경망은 새로운 가짜 데이터에 적응합니다. 이 정보는 다시 생성기 신경망을 개선하는 데 사용되는 식입니다.


판별기는 입력 $x$가 실제 데이터에서 온 진짜인지 또는 생성기에서 온 가짜인지를 구분하는 이진 분류기입니다. 일반적으로 판별기는 입력 $\mathbf x$에 대해 스칼라 예측 $o\in\mathbb R$을 출력합니다. 예를 들면, 은닉 크기가 1인 완전 연결층을 사용한 다음 시그모이드 함수를 적용하여 예측 확률 $D(\mathbf x) = 1/(1+e^{-o})$를 얻는 식입니다. 진짜 데이터의 레이블 $y$는 $1$이고 가짜 데이터의 레이블은 $0$이라고 가정합니다. 저희는 교차 엔트로피 손실을 최소화하도록 판별기를 훈련합니다. *즉*,

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

생성기의 경우, 먼저 무작위성의 원천(*예를 들면* 정규분포 $\mathbf z \sim \mathcal{N} (0, 1)$)에서 일부 매개변수 $\mathbf z\in\mathbb R^d$를 추출합니다. 저희는 종종 $\mathbf z$를 잠재 변수(latent variable)라고 부릅니다.
그런 다음 함수를 적용하여 $\mathbf x'=G(\mathbf z)$를 생성합니다. 생성기의 목표는 판별기를 속여서 $\mathbf x'=G(\mathbf z)$를 진짜 데이터로 분류하도록 하는 것입니다. *즉*, 저희는 $D( G(\mathbf z)) \approx 1$이 되기를 원합니다.
다시 말해, 주어진 판별기 $D$에 대해, 저희는 $y=0$일 때 교차 엔트로피 손실을 최대화하도록 생성기 $G$의 매개변수를 업데이트합니다. *즉*,

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

생성기가 완벽한 일을 한다면, $D(\mathbf x')\approx 1$이므로 위의 손실은 0에 가까워지고, 이는 판별기에 대해 좋은 진전을 이루기에 너무 작은 기울기를 초래합니다. 그래서 일반적으로 저희는 다음 손실을 최소화합니다.

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

이는 단지 $\mathbf x'=G(\mathbf z)$를 판별기에 입력하지만 레이블 $y=1$을 주는 것입니다.


요약하면, $D$와 $G$는 다음의 포괄적인 목적 함수를 가진 "최소최대(minimax)" 게임을 하고 있습니다.

$$\min_D \max_G \{ -E_{x \sim \textrm{Data}} \log D(\mathbf x) - E_{z \sim \textrm{Noise}} \log(1 - D(G(\mathbf z))) \}.$$



GAN 응용의 상당수는 이미지의 맥락에 있습니다. 시연 목적으로, 저희는 먼저 훨씬 간단한 분포를 적합시키는 것으로 만족하겠습니다. 저희는 GAN을 사용하여 세계에서 가장 비효율적인 가우시안 매개변수 추정기를 만들면 어떻게 되는지 설명하겠습니다. 시작해 봅시다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 일부 "진짜" 데이터 생성하기

이것은 세계에서 가장 시시한 예제가 될 것이기 때문에, 저희는 단순히 가우시안에서 추출된 데이터를 생성합니다.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((1000, 2), 0.0, 1)
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2], tf.float32)
data = d2l.matmul(X, A) + b
```

저희가 얻은 것을 봅시다. 이는 평균 $b$와 공분산 행렬 $A^TA$를 가진, 다소 임의적인 방식으로 이동된 가우시안이어야 합니다.

```{.python .input}
#@tab mxnet, pytorch
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{tf.matmul(A, A, transpose_a=True)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## 생성기

저희의 생성기 신경망은 가능한 한 가장 단순한 신경망(단일층 선형 모델)이 될 것입니다. 이는 저희가 그 선형 신경망을 가우시안 데이터 생성기로 구동할 것이기 때문입니다. 따라서, 그것은 말 그대로 완벽하게 위조하기 위한 매개변수만 학습하면 됩니다.

```{.python .input}
#@tab mxnet
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```

```{.python .input}
#@tab tensorflow
net_G = tf.keras.layers.Dense(2)
```

## 판별기

판별기에 대해서는 좀 더 까다롭게(discriminating) 할 것입니다. 좀 더 흥미롭게 만들기 위해 3개 층의 MLP를 사용하겠습니다.

```{.python .input}
#@tab mxnet
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

```{.python .input}
#@tab tensorflow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## 훈련

먼저 판별기를 업데이트하는 함수를 정의합니다.

```{.python .input}
#@tab mxnet
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for `net_G`, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input}
#@tab tensorflow
#@save
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,)) # Labels corresponding to real data
    zeros = tf.zeros((batch_size,)) # Labels corresponding to fake data
    # Do not need to compute gradient for `net_G`, so it is outside GradientTape
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogitsLoss
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

생성기도 비슷하게 업데이트됩니다. 여기서 저희는 교차 엔트로피 손실을 재사용하지만 가짜 데이터의 레이블을 $0$에서 $1$로 변경합니다.

```{.python .input}
#@tab mxnet
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input}
#@tab tensorflow
#@save
def update_G(Z, net_D, net_G, loss, optimizer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogits loss
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

판별기와 생성기 모두 교차 엔트로피 손실을 사용하여 이진 로지스틱 회귀를 수행합니다. 저희는 훈련 과정을 부드럽게 하기 위해 Adam을 사용합니다. 각 반복에서, 저희는 먼저 판별기를 업데이트한 다음 생성기를 업데이트합니다. 저희는 두 손실과 생성된 예제를 모두 시각화합니다.

```{.python .input}
#@tab mxnet
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], nrows=2,
        figsize=(5, 5), legend=["discriminator", "generator"])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(
                mean=0, stddev=1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
        # Visualize generated examples
        Z = tf.random.normal(mean=0, stddev=1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(["real", "generated"])

        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))

    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

이제 저희는 가우시안 분포를 적합시키기 위한 하이퍼파라미터를 지정합니다.

```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## 요약

* 생성적 적대 신경망(GAN)은 생성기와 판별기라는 두 개의 심층망으로 구성됩니다.
* 생성기는 교차 엔트로피 손실을 최대화함으로써 판별기를 속이기 위해 진짜 이미지에 가능한 한 가까운 이미지를 생성합니다. *즉*, $\max \log(D(\mathbf{x'}))$.
* 판별기는 교차 엔트로피 손실을 최소화함으로써 생성된 이미지를 진짜 이미지와 구별하려고 시도합니다. *즉*, $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$.

## 연습문제

* 생성기가 이기는 평형이 존재할까요? *즉*, 판별기가 결국 유한한 샘플에서 두 분포를 구별할 수 없게 되는 경우 말입니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/408)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1082)
:end_tab:
