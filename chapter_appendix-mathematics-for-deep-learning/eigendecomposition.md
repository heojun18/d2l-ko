# 고유분해
:label:`sec_eigendecompositions`

고윳값은 선형대수를 공부할 때 마주칠 가장 유용한 개념 중 하나이지만, 초보자로서는 그 중요성을 간과하기 쉽습니다. 아래에서, 저희는 고유분해를 소개하고 그것이 왜 그렇게 중요한지에 대한 약간의 감을 전달하려고 합니다.

다음과 같은 항목을 가진 행렬 $A$가 있다고 가정해 보십시오.

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

$A$를 어떤 벡터 $\mathbf{v} = [x, y]^\top$에 적용하면, 벡터 $\mathbf{A}\mathbf{v} = [2x, -y]^\top$를 얻습니다. 이는 직관적인 해석을 가집니다. 벡터를 $x$ 방향으로 두 배 더 넓게 늘리고, 그런 다음 $y$ 방향으로 뒤집습니다.

그러나, *일부* 벡터의 경우 어떤 것도 변경되지 않은 채 남아 있습니다. 즉, $[1, 0]^\top$는 $[2, 0]^\top$로 보내지고, $[0, 1]^\top$는 $[0, -1]^\top$로 보내집니다. 이러한 벡터들은 여전히 같은 직선 위에 있고, 유일한 수정은 행렬이 각각 $2$와 $-1$의 인자로 그들을 늘린다는 것입니다. 저희는 이러한 벡터를 *고유벡터*라고 부르고, 그들이 늘려지는 인자를 *고윳값*이라고 부릅니다.

일반적으로, 다음과 같은 수 $\lambda$와 벡터 $\mathbf{v}$를 찾을 수 있다면

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

$\mathbf{v}$가 $A$에 대한 고유벡터이고 $\lambda$가 고윳값이라고 말합니다.

## 고윳값 찾기
이를 어떻게 찾는지 알아보겠습니다. 양변에서 $\lambda \mathbf{v}$를 빼고, 그런 다음 벡터를 인수분해하면, 위가 다음과 동치임을 알 수 있습니다.

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

:eqref:`eq_eigvalue_der`가 발생하려면, $(\mathbf{A} - \lambda \mathbf{I})$가 어떤 방향을 0으로 압축해야 한다는 것을 알 수 있고, 따라서 가역적이지 않으며, 그러므로 행렬식이 0입니다. 따라서, 저희는 $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$인 $\lambda$를 찾음으로써 *고윳값*을 찾을 수 있습니다. 고윳값을 찾으면, $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$를 풀어 연관된 *고유벡터(들)*을 찾을 수 있습니다.

### 예제
더 도전적인 행렬로 이것을 봅시다.

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$을 고려하면, 이것이 다항식 방정식 $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$와 동치임을 알 수 있습니다. 따라서 두 고윳값은 $4$와 $1$입니다. 연관된 벡터를 찾기 위해서는, 그러면 다음을 풀어야 합니다.

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

저희는 각각 벡터 $[1, -1]^\top$와 $[1, 2]^\top$로 이를 풀 수 있습니다.

내장된 `numpy.linalg.eig` 루틴을 사용하여 코드로 이를 확인할 수 있습니다.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

`numpy`는 고유벡터를 길이 1로 정규화하는 반면, 저희는 임의의 길이로 가져갔다는 점에 유의하십시오. 또한, 부호의 선택은 임의적입니다. 그러나, 계산된 벡터는 같은 고윳값을 가지고 저희가 손으로 찾은 것과 평행합니다.

## 행렬 분해
이전 예제를 한 단계 더 진행해 보겠습니다. 다음과 같이

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

열이 행렬 $\mathbf{A}$의 고유벡터인 행렬이라고 합시다. 다음과 같이

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

대각선에 연관된 고윳값이 있는 행렬이라고 합시다. 그러면 고윳값과 고유벡터의 정의는 저희에게 다음을 알려줍니다.

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

행렬 $W$는 가역적이므로, 오른쪽에서 양변에 $W^{-1}$를 곱할 수 있고, 저희는 다음과 같이 쓸 수 있음을 봅니다.

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

다음 절에서 이것의 몇 가지 좋은 결과를 볼 것이지만, 지금은 이러한 분해가 선형 독립인 고유벡터의 완전한 모음을 찾을 수 있는 한($W$가 가역적이 되도록) 존재할 것이라는 것만 알면 됩니다.

## 고유분해에 대한 연산
고유분해 :eqref:`eq_eig_decomp`의 좋은 점 중 하나는, 저희가 보통 마주치는 많은 연산을 고유분해의 관점에서 깔끔하게 쓸 수 있다는 것입니다. 첫 번째 예로, 다음을 고려해 보십시오.

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

이는 행렬의 어떤 양의 거듭제곱에 대해서도, 고유분해는 고윳값을 같은 거듭제곱으로 올리는 것만으로 얻어진다는 것을 알려줍니다. 음의 거듭제곱에 대해서도 같은 것이 보여질 수 있으므로, 행렬을 역으로 만들고 싶다면 다음만 고려하면 됩니다.

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

또는 다시 말해, 각 고윳값만 역으로 만들면 됩니다. 이는 각 고윳값이 0이 아닌 한 작동할 것이므로, 가역적이라는 것이 0인 고윳값이 없는 것과 같다는 것을 알 수 있습니다.

사실, 추가 작업으로 $\lambda_1, \ldots, \lambda_n$이 행렬의 고윳값이라면, 그 행렬의 행렬식이 다음과 같음을 보일 수 있습니다.

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

또는 모든 고윳값의 곱입니다. 이는 직관적으로 말이 되는데, 왜냐하면 $\mathbf{W}$가 어떤 늘리기를 하든, $W^{-1}$가 그것을 되돌리므로, 결국 발생하는 유일한 늘리기는 대각 행렬 $\boldsymbol{\Sigma}$에 의한 곱셈에 의한 것이며, 이는 대각 원소의 곱에 의해 부피를 늘립니다.

마지막으로, 랭크는 행렬의 선형 독립인 열의 최대 개수였음을 떠올리십시오. 고유분해를 자세히 살펴봄으로써, 랭크가 $\mathbf{A}$의 0이 아닌 고윳값의 개수와 같다는 것을 알 수 있습니다.

예제는 계속될 수 있지만, 요점은 분명하기를 바랍니다. 고유분해는 많은 선형대수적 계산을 단순화할 수 있고, 많은 수치적 알고리즘과 저희가 선형대수에서 하는 많은 분석의 기저에 있는 근본적인 연산입니다.

## 대칭 행렬의 고유분해
위의 과정이 작동하기에 충분한 선형 독립인 고유벡터를 찾는 것이 항상 가능한 것은 아닙니다. 예를 들어, 행렬

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

은 단일 고유벡터, 즉 $(1, 0)^\top$만 가집니다. 이러한 행렬을 다루려면, 저희가 다룰 수 있는 것보다 더 발전된 기법(예: 조르당 표준형 또는 특이값 분해)이 필요합니다. 저희는 종종 고유벡터의 완전한 집합의 존재를 보장할 수 있는 행렬에 주의를 제한할 필요가 있을 것입니다.

가장 일반적으로 마주치는 가족은 *대칭 행렬*인데, 이는 $\mathbf{A} = \mathbf{A}^\top$인 행렬입니다. 이 경우, 저희는 $W$를 *직교 행렬*(열이 모두 길이 1의 벡터이고 서로 직각을 이루며, $\mathbf{W}^\top = \mathbf{W}^{-1}$인 행렬)로 취할 수 있고, 모든 고윳값은 실수일 것입니다. 따라서, 이 특수한 경우에 :eqref:`eq_eig_decomp`를 다음과 같이 쓸 수 있습니다.

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## 게르슈고린 원 정리
고윳값은 종종 직관적으로 추론하기 어렵습니다. 임의의 행렬이 제시되면, 그것들을 계산하지 않고는 고윳값이 무엇인지에 대해 말할 수 있는 것이 거의 없습니다. 그러나, 가장 큰 값이 대각선에 있는 경우에 잘 근사하기 쉬운 한 가지 정리가 있습니다.

$\mathbf{A} = (a_{ij})$를 어떤 정사각 행렬($n\times n$)이라고 합시다. 저희는 $r_i = \sum_{j \neq i} |a_{ij}|$를 정의할 것입니다. $\mathcal{D}_i$를 중심 $a_{ii}$ 반지름 $r_i$인 복소 평면의 원반이라고 합시다. 그러면, $\mathbf{A}$의 모든 고윳값은 $\mathcal{D}_i$ 중 하나에 포함됩니다.

이는 풀어내기에 약간일 수 있으므로, 예제를 살펴봅시다. 행렬을 고려해 보십시오.

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

저희는 $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ 그리고 $r_4 = 0.9$를 가집니다. 행렬은 대칭이므로, 모든 고윳값은 실수입니다. 이는 저희의 모든 고윳값이 다음 범위 중 하나에 있을 것임을 의미합니다.

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


수치적 계산을 수행하면 고윳값이 대략 $0.99$, $2.97$, $4.95$, $9.08$임을 보여주는데, 모두 제공된 범위 안에 편안하게 들어있습니다.

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

이런 방식으로, 고윳값을 근사할 수 있고, 대각선이 다른 모든 원소보다 상당히 큰 경우 근사가 상당히 정확할 것입니다.

이는 작은 것이지만, 고유분해와 같은 복잡하고 미묘한 주제에서는 저희가 얻을 수 있는 어떤 직관적인 이해라도 얻는 것이 좋습니다.

## 유용한 응용: 반복 매핑의 성장

이제 저희는 고유벡터가 원칙적으로 무엇인지 이해했으므로, 신경망 동작의 중심 문제인 적절한 가중치 초기화에 대한 깊은 이해를 제공하는 데 어떻게 사용될 수 있는지 봅시다.

### 장기 동작으로서의 고유벡터

심층 신경망 초기화에 대한 완전한 수학적 조사는 본문의 범위를 벗어나지만, 여기서 장난감 버전을 봐서 고윳값이 이러한 모델이 어떻게 작동하는지 보는 데 어떻게 도움이 되는지 이해할 수 있습니다. 저희가 알다시피, 신경망은 선형 변환의 계층을 비선형 연산과 함께 끼워 넣음으로써 작동합니다. 여기서는 단순함을 위해, 비선형성이 없다고 가정하고, 변환이 단일 반복 행렬 연산 $A$라고 가정할 것이므로, 저희 모델의 출력은 다음과 같습니다.

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

이러한 모델이 초기화될 때, $A$는 가우시안 항목을 가진 랜덤 행렬로 취해지므로, 그 중 하나를 만들어 봅시다. 구체적으로 말하면, 평균 0, 분산 1 가우시안 분포를 가진 $5 \times 5$ 행렬로 시작합니다.

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### 무작위 데이터에 대한 동작
저희의 장난감 모델에서 단순함을 위해, 입력하는 데이터 벡터 $\mathbf{v}_{in}$이 무작위 5차원 가우시안 벡터라고 가정할 것입니다. 무슨 일이 일어나기를 원하는지 생각해 봅시다. 맥락을 위해, 일반적인 ML 문제, 즉 이미지와 같은 입력 데이터를 이미지가 고양이의 사진일 확률과 같은 예측으로 바꾸려고 시도하는 문제를 생각해 봅시다. 만약 $\mathbf{A}$의 반복 적용이 무작위 벡터를 매우 길게 늘인다면, 입력의 작은 변화가 출력의 큰 변화로 증폭될 것입니다(입력 이미지의 작은 수정이 크게 다른 예측으로 이어질 것입니다). 이는 옳지 않아 보입니다!

반대로, $\mathbf{A}$가 무작위 벡터를 더 짧게 줄인다면, 많은 계층을 통과한 후, 벡터는 본질적으로 아무것도 아닌 것으로 줄어들고, 출력은 입력에 의존하지 않을 것입니다. 이것도 분명히 옳지 않습니다!

저희는 출력이 입력에 따라 변하지만 많이는 변하지 않도록 하기 위해 성장과 감쇠 사이의 좁은 선을 걸어야 합니다!

저희가 무작위 입력 벡터에 대해 행렬 $\mathbf{A}$를 반복적으로 곱하고 노름을 추적할 때 무슨 일이 일어나는지 봅시다.

```{.python .input}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

노름이 통제할 수 없을 정도로 커지고 있습니다! 사실 몫의 리스트를 취하면, 패턴을 볼 것입니다.

```{.python .input}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

위 계산의 마지막 부분을 보면, 무작위 벡터가 `1.974459321485[...]`의 인자로 늘려진다는 것을 알 수 있는데, 끝부분이 약간 이동하지만, 늘리기 인자는 안정적입니다.

### 다시 고유벡터로 돌아가기

저희는 고유벡터와 고윳값이 어떤 것이 늘려지는 양에 해당함을 보았지만, 그것은 특정 벡터와 특정 늘리기에 대한 것이었습니다. $\mathbf{A}$에 대해 그것들이 무엇인지 살펴봅시다. 여기 약간의 주의사항이 있습니다. 그것들을 모두 보기 위해서는, 복소수로 가야 한다는 것이 밝혀집니다. 이것들을 늘리기와 회전으로 생각할 수 있습니다. 복소수의 노름(실수부와 허수부 제곱의 합의 제곱근)을 취함으로써 그 늘리기 인자를 측정할 수 있습니다. 또한 그들을 정렬합시다.

```{.python .input}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### 한 가지 관찰

저희는 여기서 약간 예상치 못한 일이 일어나는 것을 봅니다. 저희가 무작위 벡터에 적용된 행렬 $\mathbf{A}$의 장기 늘리기에 대해 이전에 식별한 그 숫자가 *정확히* (소수점 13자리까지 정확하게!) $\mathbf{A}$의 가장 큰 고윳값입니다. 이는 분명히 우연이 아닙니다!

그러나, 이제 기하학적으로 무슨 일이 일어나고 있는지 생각하면, 이는 말이 되기 시작합니다. 무작위 벡터를 고려해 보십시오. 이 무작위 벡터는 모든 방향으로 약간씩 가리키므로, 특히 가장 큰 고윳값과 연관된 $\mathbf{A}$의 고유벡터와 같은 방향으로 적어도 약간은 가리킵니다. 이는 너무 중요해서 *주 고윳값*과 *주 고유벡터*라고 불립니다. $\mathbf{A}$를 적용한 후, 저희의 무작위 벡터는 모든 가능한 방향으로 늘려지지만(모든 가능한 고유벡터와 연관되어), 이 주 고유벡터와 연관된 방향으로 가장 많이 늘려집니다. 이것이 의미하는 것은 $A$를 적용한 후, 저희의 무작위 벡터는 더 길어지고, 주 고유벡터와 정렬되는 데 더 가까운 방향을 가리킨다는 것입니다. 행렬을 여러 번 적용한 후, 주 고유벡터와의 정렬은 점점 더 가까워져, 모든 실용적인 목적상 저희의 무작위 벡터는 주 고유벡터로 변환되었습니다! 사실 이 알고리즘은 행렬의 가장 큰 고윳값과 고유벡터를 찾기 위한 *거듭제곱 반복*으로 알려진 것의 기초입니다. 자세한 내용은 예를 들어 :cite:`Golub.Van-Loan.1996`을 참조하십시오.

### 정규화 수정

이제, 위의 논의로부터, 저희는 무작위 벡터가 늘려지거나 줄어들기를 전혀 원하지 않으며, 전체 과정에서 무작위 벡터가 거의 같은 크기로 유지되기를 원한다고 결론지었습니다. 그렇게 하기 위해, 이제 가장 큰 고윳값이 이제 단지 1이 되도록 이 주 고윳값으로 행렬을 다시 스케일링합니다. 이 경우 무슨 일이 일어나는지 봅시다.

```{.python .input}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

이전과 같이 연속된 노름 사이의 비율을 플롯할 수도 있고 실제로 안정화되는 것을 볼 수 있습니다.

```{.python .input}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## 논의

이제 저희가 바랐던 것을 정확히 볼 수 있습니다! 주 고윳값으로 행렬을 정규화한 후, 무작위 데이터가 이전처럼 폭발하지 않고, 오히려 결국 특정 값으로 평형을 이루는 것을 봅니다. 첫 번째 원칙에서 이러한 일을 할 수 있다면 좋을 것이고, 그 수학을 깊이 들여다보면, 독립인 평균 0, 분산 1 가우시안 항목을 가진 큰 무작위 행렬의 가장 큰 고윳값이 평균적으로 약 $\sqrt{n}$, 또는 저희의 경우 $\sqrt{5} \approx 2.2$임을 알 수 있는데, 이는 *원형 법칙* :cite:`Ginibre.1965`으로 알려진 매혹적인 사실 때문입니다. 무작위 행렬의 고윳값(그리고 특이값이라고 불리는 관련 객체) 사이의 관계는 :citet:`Pennington.Schoenholz.Ganguli.2017`과 후속 작업에서 논의된 것처럼 신경망의 적절한 초기화와 깊은 연결이 있음이 보여졌습니다.

## 요약
* 고유벡터는 방향을 변경하지 않고 행렬에 의해 늘려지는 벡터입니다.
* 고윳값은 행렬의 적용에 의해 고유벡터가 늘려지는 양입니다.
* 행렬의 고유분해는 많은 연산이 고윳값에 대한 연산으로 축소될 수 있게 해줍니다.
* 게르슈고린 원 정리는 행렬의 고윳값에 대한 근사값을 제공할 수 있습니다.
* 반복된 행렬 거듭제곱의 동작은 주로 가장 큰 고윳값의 크기에 의존합니다. 이러한 이해는 신경망 초기화 이론에서 많은 응용을 가집니다.

## 연습문제
1. 다음의 고윳값과 고유벡터는 무엇입니까?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  다음 행렬의 고윳값과 고유벡터는 무엇이고, 이 예제가 이전 것과 비교하여 무엇이 이상합니까?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. 고윳값을 계산하지 않고, 다음 행렬의 가장 작은 고윳값이 $0.5$보다 작을 가능성이 있습니까? *참고*: 이 문제는 머리로 할 수 있습니다.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:
