# 순환 신경망을 처음부터 구현하기
:label:`sec_rnn-scratch`

저희는 이제 RNN을 처음부터 구현할 준비가 되었습니다.
특히, 저희는 이 RNN을
문자 수준 언어 모델로서 기능하도록 학습시킬 것이며
(:numref:`sec_rnn` 참고),
:numref:`sec_text-sequence`에 개략적으로 설명된 데이터 처리 단계에 따라,
H. G. 웰스의 *The Time Machine* 전체 텍스트로 구성된 코퍼스로
학습시킬 것입니다.
저희는 데이터셋을 로드하는 것으로 시작합니다.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNN 모델

저희는 RNN 모델을 구현하기 위한 클래스를 정의하는 것으로 시작합니다
(:numref:`subsec_rnn_w_hidden_states`).
은닉 유닛의 수 `num_hiddens`는
조정 가능한 하이퍼파라미터라는 점에 유의하세요.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**아래의 `forward` 메서드는
현재 입력과 이전 타임스텝에서의 모델의 상태가 주어졌을 때,
어떤 타임스텝에서든 출력과 은닉 상태를
어떻게 계산하는지 정의합니다.**]
RNN 모델은 `inputs`의 가장 바깥쪽 차원을 따라 반복하며,
한 번에 하나의 타임스텝씩
은닉 상태를 업데이트한다는 점에 유의하세요.
여기서 모델은 $\tanh$ 활성화 함수를 사용합니다 (:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

저희는 다음과 같이 입력 시퀀스의 미니배치를 RNN 모델에 공급할 수 있습니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

은닉 상태의 차원이 변하지 않은 채로
유지되는지 확인하기 위해,
RNN 모델이 올바른 모양의 결과를
만들어내는지 확인해 봅시다.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## RNN 기반 언어 모델

다음 `RNNLMScratch` 클래스는
RNN 기반 언어 모델을 정의하는데,
저희는 `__init__` 메서드의
`rnn` 인자를 통해 RNN을 전달합니다.
언어 모델을 학습시킬 때,
입력과 출력은 같은 어휘에서 옵니다.
따라서, 그것들은 같은 차원을 가지며,
이는 어휘 크기와 같습니다.
저희는 모델을 평가하는 데 펄플렉서티를 사용한다는 점에 유의하세요.
:numref:`subsec_perplexity`에서 논의된 바와 같이, 이는
다양한 길이의 시퀀스가 비교 가능하도록 보장합니다.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**원-핫 인코딩**]

각 토큰이 해당 단어/문자/워드피스의
어휘 내 위치를 나타내는
수치 인덱스로 표현된다는 점을 떠올려 보세요.
여러분은 (각 타임스텝에서) 단일 입력 노드를 가진 신경망을 만들고 싶은
유혹을 받을 수도 있는데,
거기서 인덱스가 스칼라 값으로 공급될 수 있습니다.
이는 충분히 가까운 두 값이
비슷하게 다루어져야 하는
가격이나 온도 같은 수치 입력을
다룰 때는 잘 작동합니다.
그러나 이는 잘 말이 되지 않습니다.
저희 어휘에서 $45^{\textrm{th}}$번째 단어와 $46^{\textrm{th}}$번째 단어는
공교롭게도 "their"와 "said"이며,
그 의미는 전혀 비슷하지 않습니다.

그러한 범주형 데이터를 다룰 때,
가장 흔한 전략은 각 항목을
*원-핫 인코딩(one-hot encoding)* 으로 표현하는 것입니다
(:numref:`subsec_classification-problem`에서 떠올려 보세요).
원-핫 인코딩은 길이가 어휘 크기 $N$으로 주어지는 벡터이며,
저희의 토큰에 해당하는 항목만 $1$로 설정되고
나머지 모든 항목은 $0$으로 설정됩니다.
예를 들어, 어휘에 다섯 개의 요소가 있다면,
인덱스 0과 2에 해당하는 원-핫 벡터는 다음과 같을 것입니다.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**각 반복(iteration)에서 저희가 샘플링하는 미니배치는
(배치 크기, 타임스텝 수)의 모양을 가질 것입니다.
각 입력을 원-핫 벡터로 표현하고 나면,
저희는 각 미니배치를 세 번째 축을 따른 길이가
어휘 크기(`len(vocab)`)로 주어지는
3차원 텐서로 생각할 수 있습니다.**)
저희는 종종 입력을 전치하여
(타임스텝 수, 배치 크기, 어휘 크기) 모양의 출력을 얻습니다.
이것은 저희가 한 미니배치의 은닉 상태를 타임스텝마다 업데이트하기 위해
가장 바깥쪽 차원을 따라 더 편리하게 반복할 수 있게 해 줄 것입니다
(예: 위의 `forward` 메서드에서).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### RNN 출력 변환

언어 모델은 각 타임스텝에서 RNN 출력을 토큰 예측으로 변환하기 위해 완전 연결 출력 층을 사용합니다.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

[**순전파 계산이 올바른 모양의 출력을
만들어내는지 확인해 봅시다.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**그래디언트 클리핑**]


여러분은 이미 신경망을 단일 타임스텝 내에서조차
많은 층이 입력과 출력을 분리한다는 의미에서
"깊다"고 생각하는 것에 익숙해져 있겠지만,
시퀀스의 길이는
새로운 깊이의 개념을 도입합니다.
입력에서 출력 방향으로 신경망을 통과하는 것에 더해,
첫 번째 타임스텝의 입력은
마지막 타임스텝에서의 모델 출력에 영향을 주기 위해
타임스텝을 따라 $T$개 층의 사슬을
통과해야 합니다.
거꾸로 본 관점에서, 각 반복에서,
저희는 시간을 거슬러 그래디언트를 역전파하며,
이는 길이 $\mathcal{O}(T)$의 행렬 곱의 사슬을
만들어냅니다.
:numref:`sec_numerical_stability`에서 언급한 바와 같이,
이는 가중치 행렬의 성질에 따라
그래디언트가 폭발하거나 소실되어
수치 불안정성을 초래할 수 있습니다.

그래디언트 소실과 폭발을 다루는 것은
RNN을 설계할 때 근본적인 문제이며
현대 신경망 아키텍처에서의 가장 큰 발전 일부에
영감을 주었습니다.
다음 장에서, 저희는 그래디언트 소실 문제를 완화하려는
희망으로 설계된 전문 아키텍처들에 대해
이야기할 것입니다.
그러나, 현대의 RNN조차도 종종
폭발하는 그래디언트로 어려움을 겪습니다.
한 가지 우아하지 않지만 보편적인 해결책은
단순히 그래디언트를 클리핑하여
결과적으로 "클리핑된" 그래디언트가
더 작은 값을 가지도록 강제하는 것입니다.


일반적으로 말해, 어떤 목적을 경사 하강법으로
최적화할 때, 저희는 관심 있는 파라미터, 예를 들어 벡터 $\mathbf{x}$를
반복적으로 업데이트하지만,
음의 그래디언트 $\mathbf{g}$의 방향으로 그것을 밀어냅니다
(확률적 경사 하강법에서, 저희는
무작위로 샘플링된 미니배치에서 이 그래디언트를 계산합니다).
예를 들어, 학습률 $\eta > 0$에 대해,
각 업데이트는
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$의 형태를 띕니다.
또한 목적 함수 $f$가
충분히 매끄럽다고 가정해 봅시다.
공식적으로, 저희는 목적이
상수 $L$로 *립시츠 연속(Lipschitz continuous)* 이라고 말하는데,
이는 어떤 $\mathbf{x}$와 $\mathbf{y}$에 대해서도, 다음이 성립한다는 의미입니다.

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

보시다시피, 저희가 $\eta \mathbf{g}$를 빼서 파라미터 벡터를 업데이트할 때,
목적 값의 변화는
다음과 같이 학습률, 그래디언트의 노름,
그리고 $L$에 의존합니다.

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

다시 말해, 목적은 $L \eta \|\mathbf{g}\|$보다 더 많이
변할 수 없습니다.
이 상한이 작은 값을 가지는 것은
좋게 보일 수도 있고 나쁘게 보일 수도 있습니다.
나쁜 측면에서는, 저희가 목적의 값을 줄일 수 있는
속도를 제한하고 있습니다.
좋은 측면에서는, 이는 어떤 한 그래디언트 스텝에서든
저희가 얼마나 잘못될 수 있는지를 제한합니다.


저희가 그래디언트가 폭발한다고 말할 때,
이는 $\|\mathbf{g}\|$가 지나치게 커진다는 의미입니다.
이 최악의 경우에는, 단일 그래디언트 스텝에서
수천 번의 학습 반복 동안 이루어진
모든 진전을 되돌릴 수 있을 만큼
큰 손상을 줄 수도 있습니다.
그래디언트가 그렇게 커질 수 있을 때,
신경망 학습은 종종 발산하며,
목적의 값을 줄이는 데 실패합니다.
다른 때에는, 학습이 결국 수렴하지만
손실의 큰 급증으로 인해 불안정합니다.


$L \eta \|\mathbf{g}\|$의 크기를 제한하는 한 가지 방법은
학습률 $\eta$를 작은 값으로 줄이는 것입니다.
이는 저희가 업데이트를 편향시키지 않는다는 이점이 있습니다.
그러나 큰 그래디언트가 *드물게만* 발생한다면 어떨까요?
이 극단적인 조치는 드문 그래디언트 폭발 사건을 다루기 위해서만
모든 스텝에서 저희의 진전을 느리게 만듭니다.
인기 있는 대안은 다음과 같이
그래디언트 $\mathbf{g}$를 어떤 주어진 반지름 $\theta$의 공으로
투영하는 *그래디언트 클리핑(gradient clipping)* 휴리스틱을 채택하는 것입니다.

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

이는 그래디언트 노름이 결코 $\theta$를 초과하지 않고
업데이트된 그래디언트가 $\mathbf{g}$의 원래 방향과
완전히 정렬되도록 보장합니다.
또한 어떤 주어진 미니배치(그리고 그 안의 어떤 주어진 샘플)가
파라미터 벡터에 행사할 수 있는 영향을 제한하는
바람직한 부수 효과를 갖습니다.
이는 모델에 어느 정도의 견고함을 부여합니다.
명확히 하자면, 이것은 일종의 편법(hack)입니다.
그래디언트 클리핑은 저희가 항상
참된 그래디언트를 따르고 있지는 않다는 의미이며 가능한 부수 효과에 대해 분석적으로 추론하는 것이 어렵습니다.
그러나, 이는 매우 유용한 편법이며,
대부분의 딥러닝 프레임워크의 RNN 구현에서 널리 채택됩니다.


아래에서 저희는 그래디언트를 클리핑하는 메서드를 정의하며,
이는 `d2l.Trainer` 클래스의 `fit_epoch` 메서드에 의해
호출됩니다 (:numref:`sec_linear_scratch` 참고).
그래디언트 노름을 계산할 때,
저희는 모든 모델 파라미터를 연결하여
하나의 거대한 파라미터 벡터로 다룬다는 점에 유의하세요.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## 학습

*The Time Machine* 데이터셋(`data`)을 사용하여,
저희는 처음부터 구현된 RNN(`rnn`)을 기반으로 한
문자 수준 언어 모델(`model`)을 학습시킵니다.
저희가 먼저 그래디언트를 계산한 다음,
그것들을 클리핑하고, 마지막으로
클리핑된 그래디언트를 사용하여
모델 파라미터를 업데이트한다는 점에 유의하세요.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## 디코딩

언어 모델이 학습되고 나면,
저희는 그것을 다음 토큰을 예측하는 데뿐만 아니라
이전에 예측된 토큰을 마치
입력의 다음 토큰인 것처럼 다루며,
계속해서 후속 각 토큰을 예측하는 데 사용할 수 있습니다.
때때로 저희는 문서의 시작 부분에서
시작하는 것처럼 텍스트를 생성하기를 원할 것입니다.
그러나, 종종 언어 모델을 사용자가 제공한 접두사에
조건화하는 것이 유용합니다.
예를 들어, 검색 엔진을 위한 자동 완성 기능이나
사용자가 이메일 작성을 돕는 기능을 개발한다면,
저희는 그들이 지금까지 작성한 것(접두사)을 공급한 다음,
가능성 있는 이어짐을 생성하기를 원할 것입니다.


[**다음 `predict` 메서드는
사용자가 제공한 `prefix`를 받아들인 후
한 번에 한 문자씩 이어짐을 생성합니다**].
`prefix`의 문자들을 반복하는 동안,
저희는 다음 타임스텝으로 은닉 상태를 계속 전달하지만
어떤 출력도 생성하지 않습니다.
이를 *워밍업(warm-up)* 기간이라고 합니다.
접두사를 받아들인 후, 저희는 이제
후속 문자들을 방출하기 시작할 준비가 되어 있으며,
각 문자는 다음 타임스텝에서의 입력으로
모델에 다시 공급될 것입니다.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

다음에서, 저희는 접두사를 지정하고
그것이 20개의 추가 문자를 생성하게 합니다.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

위의 RNN 모델을 처음부터 구현하는 것은 교육적이지만, 편리하지는 않습니다.
다음 절에서, 저희는 표준 아키텍처를 사용하여 RNN을 빠르게 만들어내고,
고도로 최적화된 라이브러리 함수에 의존하여
성능 향상을 거두기 위해 딥러닝 프레임워크를 어떻게 활용하는지 볼 것입니다.


## 요약

저희는 RNN 기반 언어 모델을 학습시켜 사용자가 제공한 텍스트 접두사를 따르는 텍스트를 생성할 수 있습니다.
간단한 RNN 언어 모델은 입력 인코딩, RNN 모델링, 출력 생성으로 구성됩니다.
학습 동안, 그래디언트 클리핑은 그래디언트 폭발 문제를 완화할 수 있지만 그래디언트 소실 문제는 다루지 않습니다. 실험에서, 저희는 간단한 RNN 언어 모델을 구현하고 문자 수준에서 토큰화된 텍스트 시퀀스에 대해 그래디언트 클리핑으로 학습시켰습니다. 접두사로 조건화함으로써, 저희는 언어 모델을 사용하여 가능성 있는 이어짐을 생성할 수 있으며, 이는 자동 완성 기능과 같은 많은 응용에서 유용함이 입증됩니다.


## 연습문제

1. 구현된 언어 모델은 *The Time Machine*의 맨 첫 토큰까지 모든 과거 토큰을 바탕으로 다음 토큰을 예측하나요?
1. 어떤 하이퍼파라미터가 예측에 사용되는 이력의 길이를 제어하나요?
1. 원-핫 인코딩이 각 객체에 대해 다른 임베딩을 고르는 것과 동등함을 보이세요.
1. 펄플렉서티를 개선하기 위해 하이퍼파라미터(예: 에포크 수, 은닉 유닛 수, 미니배치의 타임스텝 수, 학습률)를 조정하세요. 이 간단한 아키텍처를 고수하면서 얼마나 낮게 갈 수 있나요?
1. 원-핫 인코딩을 학습 가능한 임베딩으로 대체하세요. 이것이 더 나은 성능으로 이어지나요?
1. *The Time Machine*에서 학습된 이 언어 모델이 H. G. 웰스의 다른 책, 예를 들어
   *The War of the Worlds*에서 얼마나 잘 작동하는지 결정하기 위한 실험을 수행하세요.
1. 이 모델의 펄플렉서티를 다른 저자가 쓴 책에서 평가하기 위한
   또 다른 실험을 수행하세요.
1. 가장 가능성 있는 다음 문자를 고르는 대신
   샘플링을 사용하도록 예측 메서드를 수정하세요.
    * 무슨 일이 일어나나요?
    * 더 가능성 있는 출력 쪽으로 모델을 편향시키세요, 예를 들어,
    $\alpha > 1$에 대해 $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$로부터 샘플링함으로써.
1. 그래디언트를 클리핑하지 않고 이 절의 코드를 실행하세요. 무슨 일이 일어나나요?
1. 이 절에서 사용된 활성화 함수를 ReLU로 대체하고
   이 절의 실험을 반복하세요. 저희가 여전히 그래디언트 클리핑이 필요한가요? 왜요?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18014)
:end_tab:
