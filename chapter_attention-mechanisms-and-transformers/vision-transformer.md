```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# 비전을 위한 트랜스포머
:label:`sec_vision-transformer`

트랜스포머 아키텍처는 처음에 기계 번역에 초점을 맞춰 시퀀스 투 시퀀스 학습을 위해 제안되었습니다.
이어서 트랜스포머는 다양한 자연어 처리 작업에서 선택되는 모델로 부상했습니다 :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`.
그러나 컴퓨터 비전 분야에서는 지배적인 아키텍처가 CNN(:numref:`chap_modern_cnn`)으로 유지되었습니다.
자연스럽게 연구자들은 트랜스포머 모델을 이미지 데이터에 적응시킴으로써 더 잘할 수 있을지 궁금해하기 시작했습니다.
이 질문은 컴퓨터 비전 커뮤니티에서 엄청난 관심을 불러일으켰습니다.
최근 :citet:`ramachandran2019stand` 은 합성곱을 셀프 어텐션으로 대체하는 방식을 제안했습니다.
그러나 그 어텐션의 특수화된 패턴 사용은 하드웨어 가속기에서 모델을 확장하기 어렵게 만듭니다.
그런 다음 :citet:`cordonnier2020relationship` 은 셀프 어텐션이 합성곱과 유사하게 동작하도록 학습할 수 있음을 이론적으로 증명했습니다.
경험적으로, 이미지에서 $2 \times 2$ 패치가 입력으로 사용되었지만, 작은 패치 크기는 모델이 저해상도의 이미지 데이터에만 적용 가능하게 합니다.

패치 크기에 특정한 제약 없이, *비전 트랜스포머(vision Transformers, ViTs)* 는 이미지에서 패치를 추출하여 트랜스포머 인코더에 공급해 전역 표현을 얻고, 이는 마지막으로 분류를 위해 변환됩니다 :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
특히 트랜스포머는 CNN보다 더 나은 확장성을 보입니다: 그리고 더 큰 데이터셋에서 더 큰 모델을 학습시킬 때, 비전 트랜스포머는 ResNet을 상당한 차이로 능가합니다.
자연어 처리에서의 네트워크 아키텍처 설계 지형과 유사하게, 트랜스포머는 컴퓨터 비전에서도 게임 체인저가 되었습니다.

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 모델

:numref:`fig_vit` 은 비전 트랜스포머의 모델 아키텍처를 묘사합니다.
이 아키텍처는 이미지를 패치화하는 줄기(stem), 다층 트랜스포머 인코더에 기반한 본체(body), 그리고 전역 표현을 출력 레이블로 변환하는 헤드(head)로 구성됩니다.

![비전 트랜스포머 아키텍처. 이 예시에서 이미지는 9개 패치로 분할됩니다. 특수한 "&lt;cls&gt;" 토큰과 9개의 평탄화된 이미지 패치는 패치 임베딩과 $\mathit{n}$ 개의 트랜스포머 인코더 블록을 통해 각각 10개의 표현으로 변환됩니다. "&lt;cls&gt;" 표현은 추가로 출력 레이블로 변환됩니다.](../img/vit.svg)
:label:`fig_vit`

높이 $h$, 너비 $w$, $c$ 개의 채널을 가진 입력 이미지를 고려합시다.
패치 높이와 너비를 모두 $p$ 로 지정하면, 이미지는 $m = hw/p^2$ 개의 패치 시퀀스로 분할되며, 각 패치는 길이 $cp^2$ 의 벡터로 평탄화됩니다.
이런 방식으로 이미지 패치는 트랜스포머 인코더에 의해 텍스트 시퀀스의 토큰과 유사하게 다뤄질 수 있습니다.
특수한 "&lt;cls&gt;"(클래스) 토큰과 $m$ 개의 평탄화된 이미지 패치는 $m+1$ 개의 벡터 시퀀스로 선형 사영되어, 학습 가능한 위치 임베딩과 합산됩니다.
다층 트랜스포머 인코더는 $m+1$ 개의 입력 벡터를 같은 길이의 같은 수의 출력 벡터 표현으로 변환합니다.
이는 :numref:`fig_transformer` 의 원래 트랜스포머 인코더와 정확히 같은 방식으로 작동하며, 단지 정규화의 위치만 다릅니다.
"&lt;cls&gt;" 토큰이 셀프 어텐션을 통해 모든 이미지 패치에 주의를 기울이므로(:numref:`fig_cnn-rnn-self-attention` 참조), 트랜스포머 인코더 출력에서 그 표현은 추가로 출력 레이블로 변환됩니다.

## 패치 임베딩

비전 트랜스포머를 구현하기 위해, :numref:`fig_vit` 의 패치 임베딩으로 시작합시다.
이미지를 패치로 분할하고 이러한 평탄화된 패치를 선형 사영하는 것은 단일 합성곱 연산으로 단순화될 수 있으며, 여기서 커널 크기와 스트라이드 크기 모두 패치 크기로 설정됩니다.

```{.python .input}
%%tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

다음 예시에서, 높이와 너비가 `img_size` 인 이미지를 입력으로 받아, 패치 임베딩은 길이 `num_hiddens` 의 벡터로 선형 사영되는 `(img_size//patch_size)**2` 개의 패치를 출력합니다.

```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## 비전 트랜스포머 인코더
:label:`subsec_vit-encoder`

비전 트랜스포머 인코더의 MLP는 원래 트랜스포머 인코더의 위치별 FFN(:numref:`subsec_positionwise-ffn` 참조)과 약간 다릅니다.
첫째, 여기서 활성화 함수는 가우시안 오차 선형 유닛(GELU)을 사용하는데, 이는 ReLU의 더 매끄러운 버전으로 간주될 수 있습니다 :cite:`Hendrycks.Gimpel.2016`.
둘째, 정규화를 위해 MLP의 각 완전 연결 계층 출력에 드롭아웃이 적용됩니다.

```{.python .input}
%%tab pytorch
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

비전 트랜스포머 인코더 블록 구현은 :numref:`fig_vit` 의 사전 정규화(pre-normalization) 설계를 그대로 따르며, 여기서 정규화는 멀티헤드 어텐션이나 MLP 바로 *전(before)* 에 적용됩니다.
정규화가 잔차 연결 바로 *후(after)* 에 배치되는 사후 정규화(:numref:`fig_transformer` 의 "add & norm")와는 대조적으로, 사전 정규화는 트랜스포머에 대해 더 효과적이거나 효율적인 학습으로 이어집니다 :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`.

```{.python .input}
%%tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

:numref:`subsec_transformer-encoder` 에서와 마찬가지로, 어떤 비전 트랜스포머 인코더 블록도 입력 모양을 변경하지 않습니다.

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## 모두 합치기

아래 비전 트랜스포머의 순전파는 간단합니다.
먼저, 입력 이미지가 `PatchEmbedding` 인스턴스에 공급되고, 그 출력은 "&lt;cls&gt;" 토큰 임베딩과 연결됩니다.
이들은 드롭아웃 전에 학습 가능한 위치 임베딩과 합산됩니다.
그런 다음 출력은 `ViTBlock` 클래스의 `num_blks` 개 인스턴스를 쌓는 트랜스포머 인코더에 공급됩니다.
마지막으로, "&lt;cls&gt;" 토큰의 표현이 네트워크 헤드에 의해 사영됩니다.

```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """Vision Transformer."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## 학습

Fashion-MNIST 데이터셋에서 비전 트랜스포머를 학습시키는 것은 :numref:`chap_modern_cnn` 에서 CNN이 학습된 방식과 비슷합니다.

```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## 요약 및 논의

여러분은 Fashion-MNIST 같은 작은 데이터셋에 대해, 저희가 구현한 비전 트랜스포머가 :numref:`sec_resnet` 의 ResNet을 능가하지 못한다는 것을 알아챘을 수 있습니다.
ImageNet 데이터셋(120만 개 이미지)에서조차도 유사한 관찰이 이루어질 수 있습니다.
이는 트랜스포머가 합성곱의 평행이동 불변성과 국소성(:numref:`sec_why-conv`) 같은 유용한 원리들을 *결여*하고 있기 때문입니다.
그러나 더 큰 데이터셋(예: 3억 개 이미지)에서 더 큰 모델을 학습시킬 때 그림이 바뀌는데, 여기서 비전 트랜스포머는 이미지 분류에서 ResNet을 큰 차이로 능가하며, 확장성에서 트랜스포머의 본질적인 우수성을 입증합니다 :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
비전 트랜스포머의 도입은 이미지 데이터를 모델링하기 위한 네트워크 설계의 지형을 바꿨습니다.
그것들은 곧 DeiT의 데이터 효율적 학습 전략으로 ImageNet 데이터셋에서 효과적임을 보였습니다 :cite:`touvron2021training`.
그러나 셀프 어텐션의 이차 복잡도(:numref:`sec_self-attention-and-positional-encoding`)는 트랜스포머 아키텍처를 더 높은 해상도의 이미지에 덜 적합하게 만듭니다.
컴퓨터 비전에서 범용 백본 네트워크를 향해, Swin 트랜스포머는 이미지 크기에 대한 이차 계산 복잡도(:numref:`subsec_cnn-rnn-self-attention`)를 해결하고 합성곱과 유사한 사전 정보를 복원하여, 최첨단 결과로 이미지 분류를 넘어 다양한 컴퓨터 비전 작업에 트랜스포머의 적용성을 확장했습니다 :cite:`liu2021swin`.

## 연습문제

1. `img_size` 의 값이 학습 시간에 어떻게 영향을 미칩니까?
1. "&lt;cls&gt;" 토큰 표현을 출력으로 사영하는 대신, 평균화된 패치 표현을 어떻게 사영하시겠습니까? 이 변경을 구현하고 정확도에 어떻게 영향을 미치는지 보십시오.
1. 비전 트랜스포머의 정확도를 개선하기 위해 초매개변수를 수정할 수 있습니까?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8943)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18032)
:end_tab:
