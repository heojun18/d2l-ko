```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 문서
:begin_tab:`mxnet`
모든 MXNet 함수와 클래스를 일일이 소개하는 것은 불가능하며
(또한 그 정보는 금세 구식이 될 수 있습니다),
[API 문서](https://mxnet.apache.org/versions/1.8.0/api)와
추가적인 [튜토리얼](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) 및 예제가
그러한 문서를 제공합니다.
이 절에서는 MXNet API를 탐색하는 방법에 대한 몇 가지 지침을 제공합니다.
:end_tab:

:begin_tab:`pytorch`
모든 PyTorch 함수와 클래스를 일일이 소개하는 것은 불가능하며
(또한 그 정보는 금세 구식이 될 수 있습니다),
[API 문서](https://pytorch.org/docs/stable/index.html)와 추가적인 [튜토리얼](https://pytorch.org/tutorials/beginner/basics/intro.html) 및 예제가
그러한 문서를 제공합니다.
이 절에서는 PyTorch API를 탐색하는 방법에 대한 몇 가지 지침을 제공합니다.
:end_tab:

:begin_tab:`tensorflow`
모든 TensorFlow 함수와 클래스를 일일이 소개하는 것은 불가능하며
(또한 그 정보는 금세 구식이 될 수 있습니다),
[API 문서](https://www.tensorflow.org/api_docs)와 추가적인 [튜토리얼](https://www.tensorflow.org/tutorials) 및 예제가
그러한 문서를 제공합니다.
이 절에서는 TensorFlow API를 탐색하는 방법에 대한 몇 가지 지침을 제공합니다.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## 모듈 안의 함수와 클래스

모듈에서 어떤 함수와 클래스를 호출할 수 있는지 알기 위해서,
저희는 `dir` 함수를 사용합니다. 예를 들어,
(**난수 생성을 위한 모듈의 모든 속성을 조회**)할 수 있습니다.

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

일반적으로 `__`로 시작하고 끝나는 함수(Python의 특수 객체)나
단일 `_`로 시작하는 함수(주로 내부 함수)는 무시할 수 있습니다.
남은 함수나 속성 이름을 바탕으로,
이 모듈이 균등 분포(`uniform`), 정규 분포(`normal`),
다항 분포(`multinomial`)에서의 샘플링을 포함하여
난수를 생성하는 다양한 방법을 제공한다고
추측해 볼 수 있습니다.

## 특정 함수와 클래스

특정 함수나 클래스를 사용하는 방법에 대한 구체적인 안내를 위해서는
`help` 함수를 호출할 수 있습니다. 예를 들어,
[**텐서의 `ones` 함수 사용법을 살펴봅시다**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

문서를 통해, `ones` 함수가 지정된 모양을 가진
새로운 텐서를 생성하고 모든 원소를 1의 값으로
설정한다는 것을 알 수 있습니다.
가능하다면 언제든지 해석을 확인하기 위해
(**간단한 테스트를 실행**)해 보아야 합니다.

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

Jupyter 노트북에서는 `?`를 사용하여 다른 창에 문서를 표시할 수 있습니다.
예를 들어, `list?`는 `help(list)`와 거의 동일한 내용을 생성하여
새로운 브라우저 창에 표시합니다.
또한 `list??`처럼 물음표 두 개를 사용하면,
함수를 구현하는 Python 코드도 함께 표시됩니다.

공식 문서는 이 책의 범위를 벗어나는 많은 설명과 예제를 제공합니다.
저희는 다루는 범위의 완전성보다는,
여러분이 실용적인 문제에 빠르게 착수할 수 있도록 해 주는
중요한 사용 사례를 강조합니다.
또한 라이브러리의 소스 코드를 공부하여
프로덕션 코드의 고품질 구현 예제를 살펴볼 것을 권장합니다.
이를 통해 여러분은 더 나은 과학자가 되는 것에 더해
더 나은 엔지니어가 될 수 있을 것입니다.

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17972)
:end_tab:
