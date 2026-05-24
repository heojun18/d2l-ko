# 기하학과 선형대수 연산
:label:`sec_geometry-linear-algebraic-ops`

:numref:`sec_linear-algebra`에서 저희는 선형대수의 기초를 만났고, 데이터를 변환하기 위한 일반적인 연산을 표현하는 데 어떻게 사용될 수 있는지 보았습니다. 선형대수는 딥러닝과 더 넓게는 머신러닝에서 저희가 하는 많은 작업의 기저에 있는 핵심 수학적 기둥 중 하나입니다. :numref:`sec_linear-algebra`에는 현대 딥러닝 모델의 메커니즘을 전달하기에 충분한 도구가 포함되어 있었지만, 이 주제에는 훨씬 더 많은 것이 있습니다. 이 절에서는 저희가 더 깊이 들어가, 선형대수 연산의 일부 기하학적 해석을 강조하고, 고윳값과 고유벡터를 포함한 몇 가지 기본 개념을 소개할 것입니다.

## 벡터의 기하학
먼저 저희는 벡터에 대한 두 가지 일반적인 기하학적 해석, 즉 공간의 점 또는 방향으로서의 해석에 대해 논의할 필요가 있습니다. 근본적으로, 벡터는 아래의 Python 리스트와 같이 숫자의 리스트입니다.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

수학자들은 이를 대부분 *열* 또는 *행* 벡터로 작성합니다. 즉, 다음과 같이

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

또는

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

이들은 종종 다른 해석을 갖는데, 데이터 예제는 열 벡터이고 가중합을 형성하는 데 사용되는 가중치는 행 벡터입니다. 그러나 유연하게 다루는 것이 유익할 수 있습니다. :numref:`sec_linear-algebra`에서 설명했듯이, 단일 벡터의 기본 방향은 열 벡터이지만, 테이블 형태의 데이터셋을 나타내는 어떤 행렬의 경우, 각 데이터 예제를 행렬의 행 벡터로 다루는 것이 더 일반적입니다.

벡터가 주어지면, 저희가 부여해야 할 첫 번째 해석은 공간의 점으로서의 해석입니다. 2차원 또는 3차원에서, 저희는 벡터의 성분을 사용하여 *원점*이라고 부르는 고정된 기준에 비교한 공간상 점의 위치를 정의함으로써 이 점들을 시각화할 수 있습니다. 이는 :numref:`fig_grid`에서 볼 수 있습니다.

![벡터를 평면상의 점으로 시각화하는 것에 대한 그림. 벡터의 첫 번째 성분은 $\mathit{x}$ 좌표를 제공하고, 두 번째 성분은 $\mathit{y}$ 좌표를 제공합니다. 더 높은 차원도 유사하지만, 시각화하기는 훨씬 더 어렵습니다.](../img/grid-points.svg)
:label:`fig_grid`

이 기하학적 관점은 저희가 더 추상적인 수준에서 문제를 고려할 수 있게 해줍니다. 사진을 고양이 또는 개로 분류하는 것과 같은 극복할 수 없어 보이는 문제에 직면하는 대신, 작업을 공간의 점들의 모음으로 추상적으로 고려하기 시작할 수 있고, 작업을 두 개의 서로 다른 점들의 군집을 분리하는 방법을 발견하는 것으로 그려볼 수 있습니다.

이와 병행하여, 사람들이 종종 벡터에 대해 갖는 두 번째 관점이 있습니다. 즉, 공간상의 방향으로서의 관점입니다. 벡터 $\mathbf{v} = [3,2]^\top$를 원점에서 오른쪽으로 $3$ 단위, 위로 $2$ 단위에 있는 위치로 생각할 수 있을 뿐만 아니라, 오른쪽으로 $3$ 걸음, 위로 $2$ 걸음 이동하는 방향 그 자체로도 생각할 수 있습니다. 이런 방식으로, 저희는 :numref:`fig_arrow`의 모든 벡터를 같은 것으로 간주합니다.

![어떤 벡터든 평면상의 화살표로 시각화할 수 있습니다. 이 경우, 그려진 모든 벡터는 벡터 $(3,2)^\top$의 표현입니다.](../img/par-vec.svg)
:label:`fig_arrow`

이러한 전환의 이점 중 하나는 벡터 덧셈 행위를 시각적으로 이해할 수 있게 된다는 것입니다. 특히, 한 벡터가 제시한 방향을 따라가고, 그런 다음 다른 벡터가 제시한 방향을 따라갑니다. :numref:`fig_add-vec`에서 볼 수 있는 것처럼 말입니다.

![먼저 한 벡터를 따라가고, 그런 다음 다른 벡터를 따라감으로써 벡터 덧셈을 시각화할 수 있습니다.](../img/vec-add.svg)
:label:`fig_add-vec`

벡터 뺄셈도 유사한 해석을 가집니다. 항등식 $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$를 고려하면, 벡터 $\mathbf{u}-\mathbf{v}$가 점 $\mathbf{v}$에서 점 $\mathbf{u}$로 이동하는 방향임을 알 수 있습니다.


## 내적과 각도
:numref:`sec_linear-algebra`에서 보았듯이, 두 열 벡터 $\mathbf{u}$와 $\mathbf{v}$를 취하면, 다음을 계산하여 그들의 내적을 형성할 수 있습니다.

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

:eqref:`eq_dot_def`는 대칭이기 때문에, 저희는 고전적인 곱셈의 표기법을 따라 다음과 같이 쓸 것입니다.

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

벡터의 순서를 바꿔도 같은 답이 나온다는 사실을 강조하기 위해서 입니다.

내적 :eqref:`eq_dot_def`는 또한 기하학적 해석을 허용합니다. 이는 두 벡터 사이의 각도와 밀접하게 관련되어 있습니다. :numref:`fig_angle`에 표시된 각도를 고려해 보십시오.

![평면상의 두 벡터 사이에는 잘 정의된 각도 $\theta$가 있습니다. 이 각도가 내적과 밀접하게 관련되어 있음을 보게 될 것입니다.](../img/vec-angle.svg)
:label:`fig_angle`

시작하기 위해, 두 특정 벡터를 고려해 보겠습니다.

$$
\mathbf{v} = (r,0) \; \textrm{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

벡터 $\mathbf{v}$는 길이 $r$이고 $x$축에 평행하며, 벡터 $\mathbf{w}$는 길이 $s$이고 $x$축과 각도 $\theta$를 이룹니다. 이 두 벡터의 내적을 계산하면 다음을 볼 수 있습니다.

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

간단한 대수적 조작으로, 항을 재배열하여 다음을 얻을 수 있습니다.

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

요약하면, 이 두 특정 벡터의 경우, 내적과 노름을 결합하면 두 벡터 사이의 각도를 알 수 있습니다. 이와 같은 사실은 일반적으로 참입니다. 여기서 식을 유도하지는 않겠지만, $\|\mathbf{v} - \mathbf{w}\|^2$를 두 가지 방법으로 작성하는 것을 고려해 보십시오. 하나는 내적을 사용하는 것이고, 다른 하나는 코사인 법칙을 사용하여 기하학적으로 작성하는 것입니다. 그러면 완전한 관계를 얻을 수 있습니다. 사실, 두 벡터 $\mathbf{v}$와 $\mathbf{w}$에 대해, 두 벡터 사이의 각도는

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

이는 계산에서 어떤 것도 2차원을 참조하지 않기 때문에 좋은 결과입니다. 사실, 저희는 이를 3차원 또는 3백만 차원에서도 문제없이 사용할 수 있습니다.

간단한 예로, 한 쌍의 벡터 사이의 각도를 계산하는 방법을 살펴보겠습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

지금 당장 사용하지는 않겠지만, 각도가 $\pi/2$(또는 동등하게 $90^{\circ}$)인 벡터를 *직교*한다고 부른다는 것을 알아두는 것이 유용합니다. 위의 방정식을 살펴보면, 이는 $\theta = \pi/2$일 때 발생함을 알 수 있으며, 이는 $\cos(\theta) = 0$과 같습니다. 이것이 발생할 수 있는 유일한 방법은 내적 자체가 0인 경우이며, 두 벡터는 $\mathbf{v}\cdot\mathbf{w} = 0$일 때 그리고 오직 그때만 직교합니다. 이는 객체를 기하학적으로 이해할 때 유용한 공식이 될 것입니다.

다음과 같은 질문을 하는 것이 합리적입니다. 왜 각도를 계산하는 것이 유용할까요? 답은 저희가 데이터가 가지기를 기대하는 종류의 불변성에서 옵니다. 어떤 이미지와, 모든 픽셀 값이 같지만 밝기가 $10\%$인 복제 이미지를 고려해 보십시오. 개별 픽셀의 값은 일반적으로 원래 값과는 거리가 멉니다. 따라서 원본 이미지와 더 어두운 이미지 사이의 거리를 계산하면, 그 거리는 클 수 있습니다. 그러나 대부분의 ML 응용에서, *내용*은 동일합니다(고양이/개 분류기에 관한 한 여전히 고양이의 이미지입니다). 그러나 각도를 고려하면, 어떤 벡터 $\mathbf{v}$에 대해서도 $\mathbf{v}$와 $0.1\cdot\mathbf{v}$ 사이의 각도가 0임을 보기는 어렵지 않습니다. 이는 벡터를 스케일링해도 같은 방향이 유지되고 길이만 변한다는 사실에 해당합니다. 각도는 더 어두운 이미지를 동일한 것으로 간주합니다.

이와 같은 예는 어디에나 있습니다. 텍스트에서, 같은 것을 말하는 두 배 더 긴 문서를 작성하더라도 논의되는 주제가 변하지 않기를 원할 수 있습니다. 일부 인코딩(예: 어떤 어휘에서 단어의 출현 횟수를 세는 것)의 경우, 이는 문서를 인코딩하는 벡터의 두 배에 해당하므로, 다시 각도를 사용할 수 있습니다.

### 코사인 유사도
ML 맥락에서 두 벡터의 근접성을 측정하기 위해 각도가 사용되는 경우, 실무자들은 다음 부분을 가리키기 위해 *코사인 유사도*라는 용어를 채택합니다.
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

코사인은 두 벡터가 같은 방향을 가리킬 때 최댓값 $1$, 반대 방향을 가리킬 때 최솟값 $-1$, 그리고 두 벡터가 직교할 때 값 $0$을 가집니다. 만약 고차원 벡터의 성분이 평균 $0$으로 무작위로 샘플링된다면, 그들의 코사인은 거의 항상 $0$에 가까울 것이라는 점에 유의하십시오.


## 초평면

벡터를 다루는 것 외에도, 선형대수에서 멀리 나아가기 위해 이해해야 할 또 다른 핵심 객체는 *초평면*입니다. 이는 직선(2차원) 또는 평면(3차원)을 더 높은 차원으로 일반화한 것입니다. $d$차원 벡터 공간에서, 초평면은 $d-1$ 차원을 가지며 공간을 두 개의 반공간으로 나눕니다.

예제로 시작해 보겠습니다. 열 벡터 $\mathbf{w}=[2,1]^\top$가 있다고 가정해 보십시오. 저희는 "$\mathbf{w}\cdot\mathbf{v} = 1$인 점 $\mathbf{v}$는 무엇인가?"를 알고 싶습니다. 위의 내적과 각도 사이의 관계 :eqref:`eq_angle_forumla`를 떠올리면, 이것이 다음과 동치임을 알 수 있습니다.
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![삼각법을 떠올려보면, 공식 $\|\mathbf{v}\|\cos(\theta)$가 벡터 $\mathbf{v}$를 $\mathbf{w}$의 방향으로 사영한 길이임을 알 수 있습니다.](../img/proj-vec.svg)
:label:`fig_vector-project`

이 식의 기하학적 의미를 고려해보면, 이는 $\mathbf{v}$를 $\mathbf{w}$의 방향으로 사영한 길이가 정확히 $1/\|\mathbf{w}\|$라는 것과 동치임을 알 수 있습니다. 이는 :numref:`fig_vector-project`에 표시되어 있습니다. 이것이 참인 모든 점들의 집합은 벡터 $\mathbf{w}$에 직각인 직선입니다. 원한다면, 이 직선에 대한 방정식을 찾을 수 있고, 이것이 $2x + y = 1$ 또는 동등하게 $y = 1 - 2x$임을 알 수 있습니다.

이제 $\mathbf{w}\cdot\mathbf{v} > 1$ 또는 $\mathbf{w}\cdot\mathbf{v} < 1$인 점들의 집합에 대해 물을 때 어떤 일이 일어나는지 보면, 이는 각각 사영이 $1/\|\mathbf{w}\|$보다 길거나 짧은 경우임을 알 수 있습니다. 따라서 이 두 부등식은 직선의 양쪽 면을 정의합니다. 이런 식으로, 저희는 공간을 두 반쪽으로 자르는 방법을 찾았는데, 한쪽의 모든 점은 내적이 임계값 아래이고, 다른 쪽은 :numref:`fig_space-division`에서 볼 수 있듯이 임계값 위입니다.

![이제 식의 부등식 버전을 고려하면, 저희의 초평면(이 경우: 단지 직선)이 공간을 두 반쪽으로 분리한다는 것을 알 수 있습니다.](../img/space-division.svg)
:label:`fig_space-division`

더 높은 차원에서의 이야기도 거의 같습니다. 이제 $\mathbf{w} = [1,2,3]^\top$를 취하고 $\mathbf{w}\cdot\mathbf{v} = 1$인 3차원의 점들에 대해 물으면, 주어진 벡터 $\mathbf{w}$에 직각인 평면을 얻습니다. 두 부등식은 다시 :numref:`fig_higher-division`에서 볼 수 있듯이 평면의 양쪽을 정의합니다.

![어떤 차원의 초평면이든 공간을 두 반쪽으로 분리합니다.](../img/space-division-3d.svg)
:label:`fig_higher-division`

저희의 시각화 능력은 이 시점에서 한계에 도달하지만, 수십, 수백, 또는 수십억 차원에서 이를 하는 것을 막을 수 있는 것은 없습니다. 이는 머신러닝 모델에 대해 생각할 때 자주 발생합니다. 예를 들어, :numref:`sec_softmax`의 선형 분류 모델과 같은 것들을 다른 대상 클래스를 분리하는 초평면을 찾는 방법으로 이해할 수 있습니다. 이 맥락에서, 이러한 초평면은 종종 *결정 평면*이라고 합니다. 대부분의 딥러닝 분류 모델은 소프트맥스로 공급되는 선형 계층으로 끝나므로, 심층 신경망의 역할을 대상 클래스가 초평면으로 깔끔하게 분리될 수 있도록 비선형 임베딩을 찾는 것으로 해석할 수 있습니다.

수작업으로 만든 예를 제시하기 위해, Fashion-MNIST 데이터셋(:numref:`sec_fashion_mnist`에서 본)에서 티셔츠와 바지의 작은 이미지를 분류하기 위한 합리적인 모델을 그들 평균 사이의 벡터를 취하여 결정 평면을 정의하고 대략적인 임계값을 어림짐작함으로써 만들 수 있다는 점에 유의하십시오. 먼저 데이터를 로드하고 평균을 계산합니다.

```{.python .input}
#@tab mxnet
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

이러한 평균을 자세히 살펴보는 것이 유익할 수 있으므로, 어떻게 보이는지 그려보겠습니다. 이 경우, 평균이 실제로 흐릿한 티셔츠 이미지와 닮았다는 것을 볼 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

두 번째 경우에도, 평균이 흐릿한 바지 이미지와 닮았다는 것을 다시 볼 수 있습니다.

```{.python .input}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

완전한 머신러닝 솔루션에서는, 데이터셋으로부터 임계값을 학습할 것입니다. 이 경우, 저는 단순히 훈련 데이터에서 좋아 보이는 임계값을 손으로 어림짐작했습니다.

```{.python .input}
#@tab mxnet
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean((predictions.type(y_test.dtype) == y_test).float(), dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Print test set accuracy with eyeballed threshold
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## 선형 변환의 기하학

:numref:`sec_linear-algebra`와 위의 논의를 통해, 저희는 벡터, 길이, 그리고 각도의 기하학에 대한 견고한 이해를 갖게 되었습니다. 그러나 저희가 논의를 빠뜨린 중요한 객체가 하나 있는데, 그것은 행렬로 표현되는 선형 변환에 대한 기하학적 이해입니다. 행렬이 데이터를 잠재적으로 다른 두 개의 고차원 공간 사이에서 변환하는 데 무엇을 할 수 있는지 완전히 내재화하는 것은 상당한 연습이 필요하며, 이 부록의 범위를 벗어납니다. 그러나 저희는 2차원에서 직관을 쌓기 시작할 수 있습니다.

어떤 행렬이 있다고 가정해 보십시오.

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

이를 임의의 벡터 $\mathbf{v} = [x, y]^\top$에 적용하고 싶다면, 곱셈을 수행하여 다음을 봅니다.

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

이는 명확했던 것이 다소 이해하기 어려워진 이상한 계산처럼 보일 수 있습니다. 그러나 이는 행렬이 *어떤* 벡터를 변환하는 방식을 *두 개의 특정 벡터*, 즉 $[1,0]^\top$과 $[0,1]^\top$를 어떻게 변환하는지의 관점에서 쓸 수 있다는 것을 알려줍니다. 이는 잠시 고려해볼 가치가 있습니다. 저희는 본질적으로 무한한 문제(어떤 한 쌍의 실수에 무슨 일이 일어나는가)를 유한한 문제(이 특정 벡터에 무슨 일이 일어나는가)로 축소한 것입니다. 이러한 벡터는 *기저*의 예이며, 저희는 공간의 어떤 벡터든 이러한 *기저 벡터*의 가중합으로 쓸 수 있습니다.

특정 행렬을 사용할 때 어떤 일이 일어나는지 그려보겠습니다.

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

특정 벡터 $\mathbf{v} = [2, -1]^\top$를 보면, 이것이 $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$임을 알 수 있고, 따라서 행렬 $A$가 이것을 $2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$로 보낸다는 것을 알 수 있습니다. 이 논리를 모든 정수 쌍 점의 격자를 고려하여 신중하게 따라가면, 행렬 곱셈이 격자를 기울이고, 회전시키고, 스케일링할 수 있지만, :numref:`fig_grid-transform`에서 볼 수 있듯이 격자 구조는 유지되어야 한다는 것을 알 수 있습니다.

![주어진 기저 벡터에 작용하는 행렬 $\mathbf{A}$. 전체 격자가 그것과 함께 어떻게 운반되는지 주목하십시오.](../img/grid-transform.svg)
:label:`fig_grid-transform`

이것이 행렬로 표현되는 선형 변환에 대해 내재화해야 할 가장 중요한 직관적인 점입니다. 행렬은 공간의 일부 부분을 다른 부분과 다르게 왜곡할 수 없습니다. 행렬이 할 수 있는 것은 저희의 공간에서 원래의 좌표를 취하여 기울이고, 회전시키고, 스케일링하는 것뿐입니다.

일부 왜곡은 심할 수 있습니다. 예를 들어, 행렬

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

은 전체 2차원 평면을 단일 직선으로 압축합니다. 이러한 변환을 식별하고 다루는 것은 후반 절의 주제이지만, 기하학적으로는 이것이 위에서 본 변환 유형과 근본적으로 다르다는 것을 알 수 있습니다. 예를 들어, 행렬 $\mathbf{A}$의 결과는 원래의 격자로 "되돌릴" 수 있습니다. 행렬 $\mathbf{B}$의 결과는 그럴 수 없는데, 왜냐하면 벡터 $[1,2]^\top$가 어디에서 왔는지 결코 알 수 없기 때문입니다(그것이 $[1,1]^\top$였는지 $[0, -1]^\top$였는지?).

이 그림은 $2\times2$ 행렬에 대한 것이었지만, 배운 교훈을 더 높은 차원으로 가져가는 것을 막을 수 있는 것은 없습니다. $[1,0, \ldots,0]$과 같은 유사한 기저 벡터를 취하고 저희의 행렬이 그것들을 어디로 보내는지 보면, 저희가 다루고 있는 어떤 차원 공간에서든 행렬 곱셈이 전체 공간을 어떻게 왜곡하는지에 대한 감을 잡기 시작할 수 있습니다.

## 선형 종속

다시 행렬을 고려해 보십시오.

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

이는 전체 평면을 단일 직선 $y = 2x$에 살도록 압축합니다. 이제 질문이 생깁니다. 행렬 자체만 보고도 이것을 감지할 수 있는 방법이 있을까요? 답은 우리가 실제로 할 수 있다는 것입니다. $\mathbf{b}_1 = [2,4]^\top$와 $\mathbf{b}_2 = [-1, -2]^\top$를 $\mathbf{B}$의 두 열이라고 합시다. 행렬 $\mathbf{B}$로 변환된 모든 것을 행렬의 열의 가중합, 즉 $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$와 같이 쓸 수 있다는 것을 기억하십시오. 저희는 이를 *선형 결합*이라고 부릅니다. $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$라는 사실은, 다음과 같이 그 두 열의 어떤 선형 결합도 $\mathbf{b}_2$로만 완전히 쓸 수 있다는 것을 의미합니다.

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

이는 어떤 의미에서 열 중 하나가 중복된다는 것을 의미하는데, 왜냐하면 그것이 공간에서 고유한 방향을 정의하지 않기 때문입니다. 이 행렬이 전체 평면을 단일 직선으로 붕괴시킨다는 것을 이미 보았기 때문에, 이는 저희를 너무 놀라게 해서는 안 됩니다. 게다가, 선형 종속 $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$가 이를 포착한다는 것을 알 수 있습니다. 두 벡터 사이를 더 대칭적으로 만들기 위해, 이를 다음과 같이 쓰겠습니다.

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

일반적으로, 벡터의 모음 $\mathbf{v}_1, \ldots, \mathbf{v}_k$가 *선형 종속*이라고 말하는 경우는 다음과 같이 *모두가 0과 같지는 않은* 계수 $a_1, \ldots, a_k$가 존재할 때입니다.

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

이 경우, 다른 것들의 어떤 조합의 관점에서 벡터 중 하나를 풀 수 있고, 효과적으로 그것을 중복되게 만들 수 있습니다. 따라서 행렬의 열에 있는 선형 종속은 저희의 행렬이 공간을 더 낮은 차원으로 압축하고 있다는 사실에 대한 증인입니다. 선형 종속이 없는 경우, 벡터가 *선형 독립*이라고 말합니다. 행렬의 열이 선형 독립이라면, 압축이 발생하지 않고 연산을 되돌릴 수 있습니다.

## 랭크

일반적인 $n\times m$ 행렬이 있는 경우, 그 행렬이 어떤 차원 공간으로 매핑하는지 묻는 것은 합리적입니다. *랭크*로 알려진 개념이 저희의 답이 될 것입니다. 이전 절에서, 선형 종속이 공간을 더 낮은 차원으로 압축하는 것에 대한 증인임을 언급했고, 따라서 이를 사용하여 랭크의 개념을 정의할 수 있을 것입니다. 특히, 행렬 $\mathbf{A}$의 랭크는 열의 모든 부분집합 중에서 선형 독립인 열의 최대 개수입니다. 예를 들어, 행렬

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

는 두 열이 선형 종속이지만, 어느 한 열 자체는 선형 종속이 아니므로 $\textrm{rank}(B)=1$입니다. 더 도전적인 예로, 다음을 고려할 수 있습니다.

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

그리고 $\mathbf{C}$가 랭크 2를 가짐을 보여줍니다. 왜냐하면, 예를 들어, 처음 두 열은 선형 독립이지만, 세 열의 네 가지 모음 중 어느 것이든 종속이기 때문입니다.

이 절차는 설명된 바와 같이 매우 비효율적입니다. 주어진 행렬의 모든 열의 부분집합을 살펴봐야 하므로, 잠재적으로 열의 수에 지수적입니다. 나중에 행렬의 랭크를 계산하는 더 계산적으로 효율적인 방법을 보겠지만, 지금은 이 개념이 잘 정의되어 있고 의미를 이해하기에 충분합니다.

## 가역성

선형 종속인 열을 가진 행렬에 의한 곱셈은 되돌릴 수 없음을 위에서 보았습니다. 즉, 항상 입력을 복구할 수 있는 역 연산이 없습니다. 그러나 풀랭크 행렬(즉, 랭크 $n$을 가진 $n \times n$ 행렬인 어떤 $\mathbf{A}$)에 의한 곱셈은 항상 되돌릴 수 있어야 합니다. 행렬을 고려해 보십시오.

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

이는 대각선을 따라 1이 있고 다른 곳은 0인 행렬입니다. 저희는 이를 *항등* 행렬이라고 부릅니다. 이는 적용했을 때 저희의 데이터를 변경하지 않는 행렬입니다. 저희의 행렬 $\mathbf{A}$가 한 일을 되돌리는 행렬을 찾기 위해서는, 다음과 같은 행렬 $\mathbf{A}^{-1}$를 찾아야 합니다.

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

이를 시스템으로 본다면, $n \times n$개의 미지수($\mathbf{A}^{-1}$의 항목)와 $n \times n$개의 방정식(곱 $\mathbf{A}^{-1}\mathbf{A}$의 모든 항목과 $\mathbf{I}$의 모든 항목 사이에 성립해야 하는 등식)이 있으므로, 일반적으로 해가 존재할 것으로 예상해야 합니다. 사실, 다음 절에서 *행렬식*이라 불리는 양을 볼 것인데, 이는 행렬식이 0이 아닌 한, 해를 찾을 수 있는 속성을 가지고 있습니다. 저희는 이러한 행렬 $\mathbf{A}^{-1}$를 *역* 행렬이라고 부릅니다. 예를 들어, $\mathbf{A}$가 일반적인 $2 \times 2$ 행렬이라면

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

역행렬은 다음과 같음을 볼 수 있습니다.

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

위의 공식에 의해 주어진 역행렬로 곱하는 것이 실제로 작동하는지 봄으로써 이를 테스트할 수 있습니다.

```{.python .input}
#@tab mxnet
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### 수치적 문제
행렬의 역행렬은 이론상으로는 유용하지만, 실제로는 대부분의 시간 동안 문제를 풀기 위해 행렬 역행렬을 *사용하기*를 원하지 않는다고 말해야 합니다. 일반적으로, 다음과 같은 선형 방정식을 푸는

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

수치적으로 훨씬 더 안정적인 알고리즘이 있습니다. 이는 역행렬을 계산하고 곱하여 다음을 얻는 것보다 더 안정적입니다.

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

작은 수로 나누는 것이 수치적 불안정성으로 이어질 수 있는 것처럼, 낮은 랭크에 가까운 행렬의 역행렬도 그럴 수 있습니다.

게다가, 행렬 $\mathbf{A}$가 *희소*인 경우가 일반적입니다. 즉, 0이 아닌 값을 적은 수만 포함합니다. 예제를 살펴보면, 이것이 역행렬이 희소함을 의미하지는 않는다는 것을 알 수 있습니다. 만약 $\mathbf{A}$가 $5$백만 개의 0이 아닌 항목만 있는 $1$백만 x $1$백만 행렬이었다면(따라서 그 $5$백만 개만 저장하면 되었다면), 역행렬은 일반적으로 거의 모든 항목이 음수가 아니어서, 저희가 $1\textrm{M}^2$ 항목 모두를 저장해야 할 것입니다(즉, $1$조 항목입니다!).

선형대수를 다룰 때 자주 마주치는 까다로운 수치적 문제에 대해 모든 것을 다이빙할 시간은 없지만, 언제 주의해서 진행해야 하는지에 대한 약간의 직관을 제공하고 싶고, 일반적으로 실제로 역행렬을 피하는 것이 좋은 경험 법칙입니다.

## 행렬식
선형대수의 기하학적 관점은 *행렬식*으로 알려진 기본 양을 해석하는 직관적인 방법을 제공합니다. 이전의 격자 이미지를 고려하되, 이제는 강조된 영역이 있습니다(:numref:`fig_grid-filled`).

![다시 격자를 왜곡하는 행렬 $\mathbf{A}$. 이번에는, 강조된 정사각형에 무슨 일이 일어나는지에 특별히 주의를 끌고 싶습니다.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

강조된 정사각형을 보십시오. 이는 $(0, 1)$과 $(1, 0)$으로 주어진 모서리를 가진 정사각형이며, 따라서 면적이 1입니다. $\mathbf{A}$가 이 정사각형을 변환한 후, 평행사변형이 되는 것을 볼 수 있습니다. 이 평행사변형이 시작했던 것과 같은 면적을 가질 이유는 없으며, 실제로 여기 표시된 다음의 특정 경우에서

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

이 평행사변형의 면적을 계산하고 면적이 $5$임을 얻는 것은 좌표 기하학의 연습입니다.

일반적으로, 행렬이 있다면

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

약간의 계산으로 결과 평행사변형의 면적이 $ad-bc$임을 알 수 있습니다. 이 면적은 *행렬식*이라고 합니다.

예제 코드로 이를 빠르게 확인해 보겠습니다.

```{.python .input}
#@tab mxnet
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

저희 중 매서운 눈을 가진 이들은 이 식이 0이거나 심지어 음수일 수 있음을 알아챌 것입니다. 음수 항의 경우, 이는 일반적으로 수학에서 취해지는 관례의 문제입니다. 행렬이 도형을 뒤집으면, 저희는 면적이 부정된다고 말합니다. 이제 행렬식이 0일 때 더 많이 배우게 됩니다.

다음을 고려해 보겠습니다.

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

이 행렬의 행렬식을 계산하면, $2\cdot(-2 ) - 4\cdot(-1) = 0$을 얻습니다. 위의 이해를 고려할 때, 이는 말이 됩니다. $\mathbf{B}$는 원본 이미지의 정사각형을 면적이 0인 선분으로 압축합니다. 그리고 실제로, 변환 후에 면적이 0이 되는 유일한 방법은 더 낮은 차원의 공간으로 압축되는 것입니다. 따라서 다음 결과가 참임을 알 수 있습니다. 행렬 $A$는 행렬식이 0이 아닌 경우에 그리고 오직 그 경우에만 가역적입니다.

마지막 코멘트로, 평면에 그려진 어떤 도형이 있다고 상상해 보십시오. 컴퓨터 과학자처럼 생각하면, 그 도형을 작은 정사각형들의 모음으로 분해할 수 있어서, 도형의 면적은 본질적으로 분해의 정사각형 수에 불과합니다. 이제 그 도형을 행렬로 변환하면, 이 정사각형 각각을 평행사변형으로 보내며, 각각은 행렬식에 의해 주어진 면적을 가집니다. 어떤 도형에 대해서든, 행렬식은 행렬이 어떤 도형의 면적을 스케일링하는 (부호 있는) 수를 제공함을 알 수 있습니다.

더 큰 행렬에 대한 행렬식을 계산하는 것은 힘들 수 있지만, 직관은 동일합니다. 행렬식은 $n\times n$ 행렬이 $n$차원 부피를 스케일링하는 인자로 남아 있습니다.

## 텐서와 일반적인 선형대수 연산

:numref:`sec_linear-algebra`에서 텐서의 개념이 소개되었습니다. 이 절에서는 텐서 축약(행렬 곱셈의 텐서 등가물)에 대해 더 깊이 다이빙하고, 그것이 어떻게 많은 행렬과 벡터 연산에 대한 통일된 관점을 제공할 수 있는지 볼 것입니다.

행렬과 벡터로 저희는 데이터를 변환하기 위해 그들을 곱하는 방법을 알았습니다. 텐서가 저희에게 유용하려면 유사한 정의가 필요합니다. 행렬 곱셈에 대해 생각해 보십시오.

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

또는 동등하게

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

이 패턴은 텐서에 대해 반복할 수 있는 것입니다. 텐서의 경우, 보편적으로 선택될 수 있는 합산 대상의 단일 경우가 없으므로, 정확히 어떤 인덱스를 합산할지 지정해야 합니다. 예를 들어, 다음을 고려할 수 있습니다.

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

이러한 변환은 *텐서 축약*이라고 합니다. 이는 행렬 곱셈만으로는 훨씬 더 유연한 변환의 가족을 나타낼 수 있습니다.

자주 사용되는 표기상의 단순화로, 합이 정확히 식에서 두 번 이상 나타나는 인덱스에 대한 것임을 알 수 있으므로, 사람들은 종종 *아인슈타인 표기법*으로 작업하며, 여기서 합산은 모든 반복된 인덱스에 대해 암묵적으로 취해집니다. 이는 다음과 같은 간결한 식을 제공합니다.

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### 선형대수의 일반적인 예

이전에 본 선형대수 정의의 많은 것들이 이 압축된 텐서 표기법으로 어떻게 표현될 수 있는지 보겠습니다.

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\textrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

이런 식으로, 저희는 무수히 많은 전문화된 표기법을 짧은 텐서 식으로 대체할 수 있습니다.

### 코드로 표현하기
텐서는 코드에서도 유연하게 연산될 수 있습니다. :numref:`sec_linear-algebra`에서 본 것처럼, 아래와 같이 텐서를 생성할 수 있습니다.

```{.python .input}
#@tab mxnet
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

아인슈타인 합산은 직접 구현되어 있습니다. 아인슈타인 합산에 나타나는 인덱스는 문자열로 전달될 수 있으며, 그 뒤에 작용을 받는 텐서가 옵니다. 예를 들어, 행렬 곱셈을 구현하려면, 위에서 본 아인슈타인 합산($\mathbf{A}\mathbf{v} = a_{ij}v_j$)을 고려하고 인덱스 자체를 떼어내어 구현을 얻을 수 있습니다.

```{.python .input}
#@tab mxnet
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

이것은 매우 유연한 표기법입니다. 예를 들어, 전통적으로 다음과 같이 쓰여졌을 것을 계산하고 싶다면

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

아인슈타인 합산을 통해 다음과 같이 구현될 수 있습니다.

```{.python .input}
#@tab mxnet
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

이 표기법은 인간에게는 읽기 쉽고 효율적이지만, 어떤 이유로든 프로그래밍적으로 텐서 축약을 생성해야 하는 경우 부피가 큽니다. 이런 이유로, `einsum`은 각 텐서에 대한 정수 인덱스를 제공함으로써 대안적인 표기법을 제공합니다. 예를 들어, 같은 텐서 축약은 다음과 같이도 쓸 수 있습니다.

```{.python .input}
#@tab mxnet
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch does not support this type of notation.
```

```{.python .input}
#@tab tensorflow
# TensorFlow does not support this type of notation.
```

어느 표기법이든 코드에서 텐서 축약의 간결하고 효율적인 표현을 허용합니다.

## 요약
* 벡터는 공간의 점 또는 방향으로 기하학적으로 해석될 수 있습니다.
* 내적은 임의로 높은 차원의 공간에 대한 각도의 개념을 정의합니다.
* 초평면은 직선과 평면의 고차원 일반화입니다. 이는 분류 작업의 마지막 단계로 자주 사용되는 결정 평면을 정의하는 데 사용될 수 있습니다.
* 행렬 곱셈은 기저 좌표의 균일한 왜곡으로 기하학적으로 해석될 수 있습니다. 이는 벡터를 변환하는 매우 제한적이지만 수학적으로 깔끔한 방법을 나타냅니다.
* 선형 종속은 벡터 모음이 저희가 예상하는 것보다 낮은 차원 공간에 있을 때를 알려주는 방법입니다(예: $2$차원 공간에 있는 $3$개의 벡터가 있다고 합시다). 행렬의 랭크는 선형 독립인 그 열의 가장 큰 부분집합의 크기입니다.
* 행렬의 역이 정의될 때, 행렬 역은 첫 번째 행렬의 작용을 되돌리는 다른 행렬을 찾을 수 있게 해줍니다. 행렬 역은 이론상으로는 유용하지만, 수치적 불안정성으로 인해 실제로는 주의가 필요합니다.
* 행렬식은 행렬이 공간을 얼마나 확장하거나 압축하는지 측정할 수 있게 해줍니다. 0이 아닌 행렬식은 가역(비특이) 행렬을 의미하고, 0 값의 행렬식은 행렬이 비가역(특이)임을 의미합니다.
* 텐서 축약과 아인슈타인 합산은 머신러닝에서 보이는 많은 계산을 표현하기 위한 깔끔하고 깨끗한 표기법을 제공합니다.

## 연습문제
1. 다음 사이의 각도는 무엇입니까?
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. 참 또는 거짓: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$와 $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$는 서로의 역행렬입니까?
3. 평면에 면적 $100\textrm{m}^2$인 도형을 그렸다고 가정해 보십시오. 다음 행렬로 도형을 변환한 후의 면적은 무엇입니까?
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. 다음 벡터 집합 중 어느 것이 선형 독립입니까?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. 어떤 값 $a, b, c$, 그리고 $d$의 선택에 대해 $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$로 쓰여진 행렬이 있다고 가정해 보십시오. 참 또는 거짓: 이러한 행렬의 행렬식은 항상 $0$입니까?
6. 벡터 $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$와 $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$는 직교합니다. $Ae_1$과 $Ae_2$가 직교하기 위한 행렬 $A$에 대한 조건은 무엇입니까?
7. 임의의 행렬 $A$에 대해 $\textrm{tr}(\mathbf{A}^4)$를 아인슈타인 표기법으로 어떻게 쓸 수 있습니까?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab:
