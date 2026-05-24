# 자연어 처리: 사전 학습
:label:`chap_nlp_pretrain`


인간은 의사소통이 필요합니다.
인간이라는 존재의 이 기본적 필요로부터, 매일같이 방대한 양의 글이 만들어집니다.
소셜 미디어, 채팅 앱, 이메일, 제품 리뷰, 뉴스 기사, 연구 논문, 책에 풍부한 텍스트가 존재한다는 점을 고려할 때, 컴퓨터가 이를 이해하여 인간 언어에 기반한 도움을 제공하거나 의사 결정을 내릴 수 있도록 하는 것은 매우 중요해집니다.

*자연어 처리(natural language processing)* 는 자연어를 사용한 컴퓨터와 인간 사이의 상호작용을 연구합니다.
실제로 자연어 처리 기법을 사용해 텍스트(인간의 자연어) 데이터를 처리하고 분석하는 것은 매우 흔합니다. 예를 들어 :numref:`sec_language-model`의 언어 모델이나 :numref:`sec_machine_translation`의 기계 번역 모델이 있습니다.

텍스트를 이해하기 위해,
저희는 그 표현(representation)을 학습하는 것에서부터 시작할 수 있습니다.
대규모 말뭉치에 존재하는 기존 텍스트 시퀀스를 활용하여,
*자기 지도 학습(self-supervised learning)* 은
텍스트 표현을 사전 학습하는 데 폭넓게 사용되어 왔습니다.
예를 들어 텍스트의 어떤 부분을 그 주변 텍스트의 다른 부분을 사용해 예측하는 방식이 있습니다.
이러한 방식으로,
모델은 *값비싼* 레이블링 작업 없이도
*방대한* 텍스트 데이터로부터의 지도를 통해 학습할 수 있습니다!


이 장에서 살펴보겠지만,
각 단어 또는 서브워드를 하나의 개별 토큰으로 다룰 때,
각 토큰의 표현은 word2vec, GloVe 또는 서브워드 임베딩 모델을 사용하여
대규모 말뭉치에서 사전 학습될 수 있습니다.
사전 학습 후, 각 토큰의 표현은 하나의 벡터가 될 수 있지만,
문맥이 무엇이든 그 값은 동일하게 유지됩니다.
예를 들어, "bank"의 벡터 표현은
"go to the bank to deposit some money"와
"go to the bank to sit down"
양쪽에서 모두 동일합니다.
따라서 더 최근의 많은 사전 학습 모델은 같은 토큰의 표현을
서로 다른 문맥에 적응시킵니다.
그중에는 트랜스포머 인코더에 기반한 훨씬 더 깊은 자기 지도 모델인 BERT가 있습니다.
이 장에서는 :numref:`fig_nlp-map-pretrain`에 강조된 것처럼,
텍스트에 대해 이러한 표현을 어떻게 사전 학습하는지에 초점을 맞출 것입니다.

![사전 학습된 텍스트 표현은 다양한 다운스트림 자연어 처리 응용을 위해 여러 딥러닝 아키텍처에 입력될 수 있습니다. 이 장은 업스트림 텍스트 표현 사전 학습에 초점을 맞춥니다.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`


큰 그림을 보면,
:numref:`fig_nlp-map-pretrain`은
사전 학습된 텍스트 표현이
다양한 다운스트림 자연어 처리 응용을 위해 여러 딥러닝 아키텍처에 입력될 수 있음을 보여줍니다.
저희는 이를 :numref:`chap_nlp_app`에서 다룰 것입니다.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```

