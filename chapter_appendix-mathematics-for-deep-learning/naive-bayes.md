# 나이브 베이즈
:label:`sec_naive_bayes`

이전 절들 전반에 걸쳐, 저희는 확률과 확률 변수의 이론에 대해 배웠습니다. 이 이론을 작동시키기 위해, *나이브 베이즈* 분류기를 소개합시다. 이는 숫자 분류를 수행할 수 있도록 확률적 기초 외에는 아무것도 사용하지 않습니다.

학습은 모두 가정에 관한 것입니다. 만약 저희가 본 적이 없는 새로운 데이터 예제를 분류하고 싶다면, 어떤 데이터 예제가 서로 유사한지에 대해 몇 가지 가정을 해야 합니다. 인기 있고 놀라울 정도로 명확한 알고리즘인 나이브 베이즈 분류기는 계산을 단순화하기 위해 모든 특징이 서로 독립이라고 가정합니다. 이 절에서, 저희는 이 모델을 이미지의 문자를 인식하는 데 적용할 것입니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## 광학 문자 인식

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`는 널리 사용되는 데이터셋 중 하나입니다. 이는 훈련을 위한 60,000개의 이미지와 검증을 위한 10,000개의 이미지를 포함합니다. 각 이미지는 0에서 9까지의 손글씨 숫자를 포함합니다. 작업은 각 이미지를 해당 숫자로 분류하는 것입니다.

Gluon은 인터넷에서 데이터셋을 자동으로 가져오기 위해 `data.vision` 모듈에 `MNIST` 클래스를 제공합니다. 그 후, Gluon은 이미 다운로드된 로컬 복사본을 사용할 것입니다. 저희는 매개변수 `train`의 값을 각각 `True` 또는 `False`로 설정함으로써 훈련 세트 또는 테스트 세트를 요청하는지 지정합니다. 각 이미지는 너비와 높이가 모두 $28$인 모양 ($28$,$28$,$1$)의 흑백 이미지입니다. 저희는 마지막 채널 차원을 제거하기 위해 사용자 정의 변환을 사용합니다. 추가로, 데이터셋은 각 픽셀을 부호 없는 $8$비트 정수로 나타냅니다. 저희는 문제를 단순화하기 위해 그것들을 이진 특징으로 양자화합니다.

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

저희는 이미지와 해당 레이블을 포함하는 특정 예제에 접근할 수 있습니다.

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

여기 변수 `image`에 저장된 저희의 예제는 높이와 너비가 $28$ 픽셀인 이미지에 해당합니다.

```{.python .input}
#@tab all
image.shape, image.dtype
```

저희의 코드는 각 이미지의 레이블을 스칼라로 저장합니다. 그 유형은 $32$비트 정수입니다.

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

저희는 또한 동시에 여러 예제에 접근할 수 있습니다.

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

이 예제들을 시각화해 봅시다.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## 분류를 위한 확률적 모델

분류 작업에서, 저희는 예제를 범주로 매핑합니다. 여기서 예제는 흑백 $28\times 28$ 이미지이고, 범주는 숫자입니다. (더 자세한 설명은 :numref:`sec_softmax`를 참조하십시오.)
분류 작업을 표현하는 한 가지 자연스러운 방법은 확률적 질문을 통해서입니다. 특징(즉, 이미지 픽셀)이 주어졌을 때 가장 가능성 있는 레이블은 무엇입니까? 예제의 특징을 $\mathbf x\in\mathbb R^d$로, 레이블을 $y\in\mathbb R$로 표기합시다. 여기서 특징은 이미지 픽셀이며, 저희는 $2$차원 이미지를 벡터로 다시 모양 변경할 수 있어 $d=28^2=784$이고, 레이블은 숫자입니다.
특징이 주어졌을 때의 레이블의 확률은 $p(y  \mid  \mathbf{x})$입니다. 만약 저희의 예제에서 $y=0, \ldots,9$에 대한 $p(y  \mid  \mathbf{x})$인 이러한 확률을 계산할 수 있다면, 분류기는 다음 식에 의해 주어진 예측 $\hat{y}$를 출력할 것입니다.

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

불행히도, 이는 저희가 $\mathbf{x} = x_1, ..., x_d$의 모든 값에 대해 $p(y  \mid  \mathbf{x})$를 추정해야 합니다. 각 특징이 $2$개의 값 중 하나를 취할 수 있다고 상상해 보십시오. 예를 들어, 특징 $x_1 = 1$은 주어진 문서에 단어 apple이 나타난다는 것을 나타낼 수 있고 $x_1 = 0$은 그렇지 않다는 것을 나타낼 수 있습니다. 만약 저희가 $30$개의 그러한 이진 특징을 가지고 있다면, 그것은 저희가 입력 벡터 $\mathbf{x}$의 $2^{30}$(10억 이상!) 가지 가능한 값 중 어느 것이든 분류할 준비가 되어 있어야 함을 의미합니다.

게다가, 학습은 어디에 있습니까? 만약 해당 레이블을 예측하기 위해 모든 단일 가능한 예제를 봐야 한다면, 저희는 실제로 패턴을 학습하는 것이 아니라 단지 데이터셋을 암기하는 것입니다.

## 나이브 베이즈 분류기

다행히도, 조건부 독립에 대해 몇 가지 가정을 함으로써, 저희는 약간의 귀납적 편향을 도입하고 비교적 적은 훈련 예제 선택에서 일반화할 수 있는 모델을 구축할 수 있습니다. 시작하기 위해, 베이즈 정리를 사용하여 분류기를 다음과 같이 표현합시다.

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

분모는 레이블 $y$의 값에 의존하지 않는 정규화 항 $p(\mathbf{x})$임에 유의하십시오. 결과적으로, 저희는 $y$의 다른 값에 걸쳐 분자를 비교하는 것에 대해서만 걱정하면 됩니다. 분모를 계산하는 것이 다루기 어렵다고 판명되더라도, 분자를 평가할 수 있는 한 그것을 무시할 수 있습니다. 다행히도, 정규화 상수를 복구하고 싶더라도, 저희는 할 수 있습니다. $\sum_y p(y  \mid  \mathbf{x}) = 1$이므로 저희는 항상 정규화 항을 복구할 수 있습니다.

이제, $p( \mathbf{x}  \mid  y)$에 초점을 맞춥시다. 확률의 연쇄 규칙을 사용하여, 저희는 항 $p( \mathbf{x}  \mid  y)$를 다음과 같이 표현할 수 있습니다.

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

그 자체로는, 이 식이 저희를 더 멀리 데려가지 않습니다. 저희는 여전히 대략 $2^d$ 매개변수를 추정해야 합니다. 그러나, 만약 *레이블이 주어졌을 때 특징이 서로 조건부 독립이라고* 가정한다면, 갑자기 저희는 훨씬 더 나은 모양에 있습니다. 이 항이 $\prod_i p(x_i  \mid  y)$로 단순화되어, 저희에게 예측자를 줍니다.

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

만약 모든 $i$와 $y$에 대해 $p(x_i=1  \mid  y)$를 추정하고 그 값을 $P_{xy}[i, y]$에 저장할 수 있다면, 여기서 $P_{xy}$는 $n$이 클래스의 수이고 $y\in\{1, \ldots, n\}$인 $d\times n$ 행렬, 저희는 또한 이를 $p(x_i = 0 \mid y)$를 추정하는 데 사용할 수 있습니다. 즉,

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

추가로, 저희는 모든 $y$에 대해 $p(y)$를 추정하고 그것을 $P_y[y]$에 저장합니다. $P_y$는 $n$ 길이 벡터입니다. 그러면, 어떤 새로운 예제 $\mathbf t = (t_1, t_2, \ldots, t_d)$에 대해, 저희는 다음을 계산할 수 있습니다.

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

어떤 $y$에 대해서도 말입니다. 따라서 조건부 독립이라는 저희의 가정은 저희 모델의 복잡도를 특징의 수에 대한 지수적 의존성 $\mathcal{O}(2^dn)$에서 선형 의존성, 즉 $\mathcal{O}(dn)$으로 가져갔습니다.


## 훈련

이제 문제는 저희가 $P_{xy}$와 $P_y$를 모른다는 것입니다. 그래서 저희는 먼저 어떤 훈련 데이터가 주어졌을 때 그들의 값을 추정해야 합니다. 이것이 모델을 *훈련*하는 것입니다. $P_y$를 추정하는 것은 너무 어렵지 않습니다. 저희가 $10$개의 클래스만 다루고 있으므로, 각 숫자에 대한 발생 수 $n_y$를 세고 총 데이터 양 $n$으로 나눌 수 있습니다. 예를 들어, 만약 숫자 8이 $n_8 = 5,800$번 발생하고 총 $n = 60,000$개의 이미지가 있다면, 확률 추정은 $p(y=8) = 0.0967$입니다.

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

이제 약간 더 어려운 것들 $P_{xy}$로 갑시다. 저희가 흑백 이미지를 선택했으므로, $p(x_i  \mid  y)$는 클래스 $y$에 대해 픽셀 $i$가 켜질 확률을 나타냅니다. 이전과 마찬가지로 저희는 사건이 발생하는 횟수 $n_{iy}$를 세고 $y$의 총 발생 횟수, 즉 $n_y$로 나눌 수 있습니다. 그러나 약간 골치 아픈 것이 있습니다. 특정 픽셀은 결코 검은색이 아닐 수 있습니다(예: 잘 잘린 이미지의 경우 모서리 픽셀은 항상 흰색일 수 있습니다). 통계학자들이 이 문제를 다루는 편리한 방법은 모든 발생에 의사 카운트를 추가하는 것입니다. 따라서, $n_{iy}$ 대신 저희는 $n_{iy}+1$을 사용하고 $n_y$ 대신 $n_{y}+2$를 사용합니다(픽셀 $i$가 취할 수 있는 두 가지 가능한 값 - 검은색 또는 흰색이 있으므로). 이는 또한 *라플라스 스무딩*이라고도 합니다. 이는 임시처럼 보일 수 있지만, 베타-이항 모델에 의해 베이지안 관점에서 동기 부여될 수 있습니다.

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

이러한 $10\times 28\times 28$ 확률(각 클래스에 대한 각 픽셀)을 시각화함으로써 저희는 평균처럼 보이는 숫자를 얻을 수 있습니다.

이제 저희는 새 이미지를 예측하기 위해 :eqref:`eq_naive_bayes_estimation`을 사용할 수 있습니다. $\mathbf x$가 주어지면, 다음 함수는 모든 $y$에 대해 $p(\mathbf x \mid y)p(y)$를 계산합니다.

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

이것이 끔찍하게 잘못되었습니다! 이유를 알아내기 위해, 픽셀당 확률을 봅시다. 그들은 일반적으로 $0.001$과 $1$ 사이의 숫자입니다. 저희는 그 중 $784$개를 곱하고 있습니다. 이 시점에서 저희가 컴퓨터에서, 따라서 지수에 대한 고정 범위로 이 숫자를 계산하고 있다는 것을 언급할 가치가 있습니다. 일어나는 일은 저희가 *수치적 언더플로*를 경험한다는 것입니다. 즉, 모든 작은 숫자를 곱하면 0으로 반올림될 때까지 훨씬 더 작은 것으로 이어집니다. 저희는 :numref:`sec_maximum_likelihood`에서 이를 이론적 문제로 논의했지만, 여기서 실제로 현상을 분명히 봅니다.

그 절에서 논의된 것처럼, 저희는 $\log a b = \log a + \log b$라는 사실을 사용함으로써 이를 수정합니다. 즉, 로그의 합산으로 전환합니다.
$a$와 $b$ 둘 다 작은 숫자라 하더라도, 로그 값은 적절한 범위에 있어야 합니다.

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

로그는 증가 함수이므로, 저희는 :eqref:`eq_naive_bayes_estimation`을 다음과 같이 다시 쓸 수 있습니다.

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

저희는 다음과 같은 안정적인 버전을 구현할 수 있습니다.

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

이제 예측이 올바른지 확인할 수 있습니다.

```{.python .input}
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

만약 이제 몇 개의 검증 예제를 예측한다면, 베이즈 분류기가 꽤 잘 작동한다는 것을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

마지막으로, 분류기의 전체 정확도를 계산해 봅시다.

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

현대의 심층 신경망은 $0.01$ 미만의 오류율을 달성합니다. 상대적으로 낮은 성능은 저희가 모델에서 만든 잘못된 통계적 가정 때문입니다. 저희는 각각의 모든 픽셀이 레이블에만 의존하여 *독립적으로* 생성된다고 가정했습니다. 이는 분명히 인간이 숫자를 쓰는 방식이 아니며, 이 잘못된 가정이 저희의 지나치게 나이브한 (베이즈) 분류기의 몰락으로 이어졌습니다.

## 요약
* 베이즈 규칙을 사용하여, 모든 관찰된 특징이 독립이라고 가정함으로써 분류기를 만들 수 있습니다.
* 이 분류기는 레이블과 픽셀 값의 조합의 발생 수를 셈으로써 데이터셋에서 훈련될 수 있습니다.
* 이 분류기는 스팸 감지와 같은 작업에서 수십 년 동안 황금 표준이었습니다.

## 연습문제
1. 두 원소의 XOR로 주어진 레이블 $[0,1,1,0]$을 가진 데이터셋 $[[0,0], [0,1], [1,0], [1,1]]$을 고려해 보십시오. 이 데이터셋에 구축된 나이브 베이즈 분류기에 대한 확률은 무엇입니까? 저희의 점들을 성공적으로 분류합니까? 그렇지 않다면, 어떤 가정이 위반됩니까?
1. 만약 확률을 추정할 때 라플라스 스무딩을 사용하지 않았고, 훈련에서 결코 관찰되지 않은 값을 포함하는 데이터 예제가 테스트 시간에 도착했다고 가정해 보십시오. 모델은 무엇을 출력할까요?
1. 나이브 베이즈 분류기는 확률 변수의 의존성이 그래프 구조로 인코딩되는 베이지안 네트워크의 특정 예입니다. 전체 이론은 이 절의 범위를 벗어나지만(전체 세부사항은 :citet:`Koller.Friedman.2009`를 참조), XOR 모델에서 두 입력 변수 사이의 명시적 의존성을 허용하는 것이 왜 성공적인 분류기의 생성을 허용하는지 설명하십시오.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:
