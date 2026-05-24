# Amazon SageMaker 사용하기
:label:`sec_sagemaker`

딥러닝 애플리케이션은
로컬 머신이 제공할 수 있는 것을 쉽게 넘어설 만큼
매우 많은 계산 자원을 요구할 수 있습니다.
클라우드 컴퓨팅 서비스를 사용하면
더 강력한 컴퓨터를 사용하여
이 책의 GPU 집약적인 코드를
더 쉽게 실행할 수 있습니다.
이 절에서는 Amazon SageMaker를 사용하여
이 책의 코드를 실행하는 방법을 소개하겠습니다.

## 회원 가입

먼저 https://aws.amazon.com/ 에서 계정을 등록해야 합니다.
추가 보안을 위해
이중 인증(two-factor authentication)을 사용하는 것이
권장됩니다.
또한 인스턴스 중지를 잊어버리는 등의 예상치 못한 상황을 피하기 위해
상세한 결제 및 지출 알림을 설정하는 것도
좋은 방법입니다.
AWS 계정에 로그인한 후,
[콘솔](http://console.aws.amazon.com/)로 이동하여 "Amazon SageMaker"를 검색하고(:numref:`fig_sagemaker` 참조),
이를 클릭하여 SageMaker 패널을 엽니다.

![SageMaker 패널 검색 및 열기.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## SageMaker 인스턴스 생성하기

다음으로 :numref:`fig_sagemaker-create`에 설명된 대로 노트북 인스턴스를 만들어 보겠습니다.

![SageMaker 인스턴스 만들기.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker는 다양한 계산 성능과 가격을 가진 여러 [인스턴스 타입](https://aws.amazon.com/sagemaker/pricing/instance-types/)을 제공합니다.
노트북 인스턴스를 만들 때
이름과 타입을 지정할 수 있습니다.
:numref:`fig_sagemaker-create-2`에서 저희는 `ml.p3.2xlarge`를 선택합니다. Tesla V100 GPU 1개와 8코어 CPU를 갖춘 이 인스턴스는 이 책의 대부분 내용을 다루기에 충분히 강력합니다.

![인스턴스 타입 선택하기.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
SageMaker로 실행하기 위한 ipynb 형식의 책 전체는 https://github.com/d2l-ai/d2l-en-sagemaker 에서 이용할 수 있습니다. 이 GitHub 저장소 URL(:numref:`fig_sagemaker-create-3`)을 지정하여 인스턴스를 만들 때 SageMaker가 이를 복제하도록 할 수 있습니다.
:end_tab:

:begin_tab:`pytorch`
SageMaker로 실행하기 위한 ipynb 형식의 책 전체는 https://github.com/d2l-ai/d2l-pytorch-sagemaker 에서 이용할 수 있습니다. 이 GitHub 저장소 URL(:numref:`fig_sagemaker-create-3`)을 지정하여 인스턴스를 만들 때 SageMaker가 이를 복제하도록 할 수 있습니다.
:end_tab:

:begin_tab:`tensorflow`
SageMaker로 실행하기 위한 ipynb 형식의 책 전체는 https://github.com/d2l-ai/d2l-tensorflow-sagemaker 에서 이용할 수 있습니다. 이 GitHub 저장소 URL(:numref:`fig_sagemaker-create-3`)을 지정하여 인스턴스를 만들 때 SageMaker가 이를 복제하도록 할 수 있습니다.
:end_tab:

![GitHub 저장소 지정하기.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## 인스턴스 실행 및 중지하기

인스턴스 생성에는
몇 분이 걸릴 수 있습니다.
준비가 되면,
옆에 있는 "Open Jupyter" 링크를 클릭하여(:numref:`fig_sagemaker-open`) 이 인스턴스에서
이 책의 모든 Jupyter 노트북을
편집하고 실행할 수 있습니다
(:numref:`sec_jupyter`의 단계와 유사합니다).

![생성된 SageMaker 인스턴스에서 Jupyter 열기.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`


작업을 마치신 후에는
추가 요금이 청구되는 것을 피하기 위해
인스턴스를 중지하는 것을 잊지 마세요(:numref:`fig_sagemaker-stop`).

![SageMaker 인스턴스 중지하기.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## 노트북 업데이트하기

:begin_tab:`mxnet`
이 오픈 소스 책의 노트북은 GitHub의 [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker) 저장소에서
정기적으로 업데이트될 것입니다.
최신 버전으로 업데이트하려면,
SageMaker 인스턴스에서 터미널을 여시면 됩니다(:numref:`fig_sagemaker-terminal`).
:end_tab:

:begin_tab:`pytorch`
이 오픈 소스 책의 노트북은 GitHub의 [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker) 저장소에서
정기적으로 업데이트될 것입니다.
최신 버전으로 업데이트하려면,
SageMaker 인스턴스에서 터미널을 여시면 됩니다(:numref:`fig_sagemaker-terminal`).
:end_tab:


:begin_tab:`tensorflow`
이 오픈 소스 책의 노트북은 GitHub의 [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker) 저장소에서
정기적으로 업데이트될 것입니다.
최신 버전으로 업데이트하려면,
SageMaker 인스턴스에서 터미널을 여시면 됩니다(:numref:`fig_sagemaker-terminal`).
:end_tab:


![SageMaker 인스턴스에서 터미널 열기.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

원격 저장소에서 업데이트를 가져오기 전에 로컬 변경 사항을 커밋하고 싶을 수 있습니다.
그렇지 않다면 터미널에서 다음 명령을 사용하여
모든 로컬 변경 사항을 그냥 폐기하세요.

:begin_tab:`mxnet`

```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`pytorch`

```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`tensorflow`

```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```


:end_tab:

## 요약

* Amazon SageMaker를 사용하여 노트북 인스턴스를 만들어 이 책의 GPU 집약적인 코드를 실행할 수 있습니다.
* Amazon SageMaker 인스턴스의 터미널을 통해 노트북을 업데이트할 수 있습니다.


## 연습문제


1. Amazon SageMaker를 사용하여 GPU가 필요한 절을 편집하고 실행해 보세요.
1. 이 책의 모든 노트북을 호스팅하는 로컬 디렉터리에 접근하기 위해 터미널을 열어 보세요.


[토론](https://discuss.d2l.ai/t/422)
