# `d2l` API 문서
:label:`sec_d2l`

이 절에서는 `d2l` 패키지의 클래스와 함수(알파벳순으로 정렬)를 보여 주며, 더 자세한 구현과 설명을 찾을 수 있도록 책에서 어디에 정의되어 있는지를 표시합니다.
[GitHub 저장소](https://github.com/d2l-ai/d2l-en/tree/master/d2l)의 소스 코드도 참조하세요.

:begin_tab:`pytorch`

```eval_rst

.. currentmodule:: d2l.torch

```

:begin_tab:`mxnet`

```eval_rst

.. currentmodule:: d2l.mxnet

```

:end_tab:


:begin_tab:`tensorflow`

```eval_rst

.. currentmodule:: d2l.torch

```


:end_tab:

## 클래스

```eval_rst 

.. autoclass:: AdditiveAttention
   :members:
   
.. autoclass:: AddNorm
   :members:

.. autoclass:: AttentionDecoder
   :members: 

.. autoclass:: Classifier
   :members: 
   
.. autoclass:: DataModule
   :members: 
   
.. autoclass:: Decoder
   :members: 
   
.. autoclass:: DotProductAttention
   :members:
   
.. autoclass:: Encoder
   :members:
   
.. autoclass:: EncoderDecoder
   :members:
   
.. autoclass:: FashionMNIST
   :members: 
   
.. autoclass:: GRU
   :members: 
   
.. autoclass:: HyperParameters
   :members: 
   
.. autoclass:: LeNet
   :members: 
   
.. autoclass:: LinearRegression
   :members: 
   
.. autoclass:: LinearRegressionScratch
   :members: 
   
.. autoclass:: Module
   :members: 
   
.. autoclass:: MTFraEng
   :members: 
   
.. autoclass:: MultiHeadAttention
   :members:
   
.. autoclass:: PositionalEncoding
   :members:
   
.. autoclass:: PositionWiseFFN
   :members:
   
.. autoclass:: ProgressBoard
   :members: 
   
.. autoclass:: Residual
   :members: 
   
.. autoclass:: ResNeXtBlock
   :members:
   
.. autoclass:: RNN
   :members: 
   
.. autoclass:: RNNLM
   :members:
   
.. autoclass:: RNNLMScratch
   :members:
   
.. autoclass:: RNNScratch
   :members: 
   
.. autoclass:: Seq2Seq
   :members:  
   
.. autoclass:: Seq2SeqEncoder
   :members:
   
.. autoclass:: SGD
   :members: 
   
.. autoclass:: SoftmaxRegression
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: TimeMachine
   :members: 

.. autoclass:: Trainer
   :members: 

.. autoclass:: TransformerEncoder 
   :members:

.. autoclass:: TransformerEncoderBlock
   :members:

.. autoclass:: Vocab
   :members: 
```


## 함수

```eval_rst 

.. autofunction:: add_to_class

.. autofunction:: bleu

.. autofunction:: check_len

.. autofunction:: check_shape

.. autofunction:: corr2d

.. autofunction:: cpu

.. autofunction:: gpu

.. autofunction:: init_cnn

.. autofunction:: init_seq2seq

.. autofunction:: masked_softmax

.. autofunction:: num_gpus

.. autofunction:: plot

.. autofunction:: set_axes

.. autofunction:: set_figsize

.. autofunction:: show_heatmaps

.. autofunction:: show_list_len_pair_hist

.. autofunction:: try_all_gpus

.. autofunction:: try_gpu

.. autofunction:: use_svg_display

```

