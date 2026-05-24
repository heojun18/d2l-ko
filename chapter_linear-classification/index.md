# 분류를 위한 선형 신경망
:label:`chap_classification`

이제 여러분은 모든 메커니즘을 익혔으므로
배운 기술을 더 폭넓은 종류의 작업에 적용할 준비가 되었습니다.
저희가 분류로 전환하더라도,
대부분의 기본 구조는 동일하게 유지됩니다.
데이터를 불러오고, 모델에 통과시키고,
출력을 생성하고, 손실을 계산하고,
가중치에 대한 기울기를 구하고,
모델을 업데이트합니다.
다만, 타깃의 정확한 형태,
출력 층의 파라미터화,
그리고 손실 함수의 선택은
*분류* 환경에 맞게 적응됩니다.

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```

