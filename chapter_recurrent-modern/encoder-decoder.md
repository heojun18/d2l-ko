```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 인코더-디코더 아키텍처 (The Encoder-Decoder Architecture)
:label:`sec_encoder-decoder`

기계 번역(:numref:`sec_machine_translation`)과 같은 일반적인 시퀀스 대 시퀀스 문제에서, 입력과 출력은 정렬되지 않은 다양한 길이를 가집니다. 이런 종류의 데이터를 다루기 위한 표준 접근 방식은 두 개의 주요 구성 요소로 이루어진 *인코더-디코더(encoder-decoder)* 아키텍처(:numref:`fig_encoder_decoder`)를 설계하는 것입니다. 가변 길이 시퀀스를 입력으로 받는 *인코더(encoder)*와, 조건부 언어 모델로 작동하여 인코딩된 입력과 대상 시퀀스의 왼쪽 문맥을 받아 대상 시퀀스에서의 다음 토큰을 예측하는 *디코더(decoder)*입니다.


![인코더-디코더 아키텍처.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

영어에서 프랑스어로의 기계 번역을 예시로 들어봅시다. 영어로 된 입력 시퀀스 "They", "are", "watching", "."가 주어지면, 이 인코더-디코더 아키텍처는 먼저 가변 길이 입력을 하나의 상태로 인코딩한 다음, 그 상태를 디코딩하여 번역된 시퀀스를 토큰별로 출력으로 생성합니다. "Ils", "regardent", ".". 인코더-디코더 아키텍처는 이후 절들에서 다양한 시퀀스 대 시퀀스 모델의 기반을 형성하기 때문에, 이 절에서는 이 아키텍처를 이후에 구현될 인터페이스로 변환할 것입니다.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## (**인코더**)

인코더 인터페이스에서, 저희는 단지 인코더가 가변 길이 시퀀스를 입력 `X`로 받는다는 것만 명시합니다. 구현은 이 기본 `Encoder` 클래스를 상속하는 어떤 모델에 의해서든 제공될 것입니다.

```{.python .input}
%%tab mxnet
class Encoder(nn.Block):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Encoder(tf.keras.layers.Layer):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def call(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## [**디코더**]

다음 디코더 인터페이스에서, 저희는 인코더 출력(`enc_all_outputs`)을 인코딩된 상태로 변환하기 위한 추가적인 `init_state` 메서드를 더합니다. 이 단계는 :numref:`sec_machine_translation`에서 설명되었던 입력의 유효 길이와 같은 추가 입력을 요구할 수 있다는 점에 유의하십시오. 가변 길이 시퀀스를 토큰별로 생성하기 위해, 매번 디코더는 입력(예: 이전 시간 단계에서 생성된 토큰)과 인코딩된 상태를 현재 시간 단계의 출력 토큰으로 매핑할 수 있습니다.

```{.python .input}
%%tab mxnet
class Decoder(nn.Block):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Decoder(tf.keras.layers.Layer):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## [**인코더와 디코더를 함께 결합하기**]

순방향 전파에서, 인코더의 출력은 인코딩된 상태를 생성하는 데 사용되며, 이 상태는 디코더의 입력 중 하나로 디코더에 의해 추가로 사용될 것입니다.

```{.python .input}
%%tab mxnet, pytorch
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]
```

```{.python .input}
%%tab jax
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

다음 절에서, 저희는 이 인코더-디코더 아키텍처에 기반한 시퀀스 대 시퀀스 모델을 설계하기 위해 RNN을 적용하는 방법을 보게 될 것입니다.


## 요약

인코더-디코더 아키텍처는 둘 다 가변 길이 시퀀스로 구성된 입력과 출력을 다룰 수 있으며, 따라서 기계 번역과 같은 시퀀스 대 시퀀스 문제에 적합합니다. 인코더는 가변 길이 시퀀스를 입력으로 받아 고정된 형태의 상태로 변환합니다. 디코더는 고정된 형태의 인코딩된 상태를 가변 길이 시퀀스로 매핑합니다.


## 연습문제

1. 저희가 인코더-디코더 아키텍처를 구현하기 위해 신경망을 사용한다고 가정해 봅시다. 인코더와 디코더가 같은 유형의 신경망이어야 합니까?
1. 기계 번역 외에, 인코더-디코더 아키텍처가 적용될 수 있는 다른 응용 사례를 생각해 볼 수 있습니까?

:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/3864)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/18021)
:end_tab:
