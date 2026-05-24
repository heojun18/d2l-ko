# 설치
:label:`chap_installation`

본격적으로 시작하려면,
Python을 실행할 수 있는 환경과
Jupyter Notebook, 관련 라이브러리,
그리고 이 책 자체를 실행하는 데 필요한 코드가 필요합니다.

## Miniconda 설치

가장 간단한 방법은
[Miniconda](https://conda.io/en/latest/miniconda.html)를 설치하는 것입니다.
Python 3.x 버전이 필요하다는 점에 유의하세요.
이미 conda가 설치되어 있는 컴퓨터라면
다음 단계는 건너뛰어도 됩니다.

Miniconda 웹사이트를 방문하여
Python 3.x 버전과 컴퓨터 아키텍처에 맞는
적절한 버전을 확인합니다.
Python 버전이 3.9(저희가 테스트한 버전)라고
가정하겠습니다.
macOS를 사용하고 있다면,
파일 이름에 "MacOSX" 문자열이 포함된
bash 스크립트를 다운로드한 뒤,
다운로드 위치로 이동해서
다음과 같이 설치를 실행합니다
(Intel Mac을 예로 들면).

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linux 사용자라면
파일 이름에 "Linux" 문자열이 포함된
파일을 다운로드한 뒤
다운로드 위치에서 다음을 실행합니다.

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows 사용자는 [온라인 안내](https://conda.io/en/latest/miniconda.html)에 따라 Miniconda를 다운로드하고 설치하면 됩니다.
Windows에서는 `cmd`를 검색해 명령 프롬프트(명령줄 인터프리터)를 열고 명령을 실행할 수 있습니다.

다음으로, `conda`를 바로 실행할 수 있도록 셸을 초기화합니다.

```bash
~/miniconda3/bin/conda init
```


그런 다음 현재 셸을 닫고 다시 엽니다.
이제 다음과 같이
새 환경을 생성할 수 있습니다.

```bash
conda create --name d2l python=3.9 -y
```


이제 `d2l` 환경을 활성화할 수 있습니다.

```bash
conda activate d2l
```


## 딥러닝 프레임워크와 `d2l` 패키지 설치

딥러닝 프레임워크를 설치하기 전에,
먼저 컴퓨터에 적절한 GPU가 있는지
확인해 보시기 바랍니다
(일반 노트북의 디스플레이 출력을 담당하는 GPU는
저희의 목적과 관련이 없습니다).
예를 들어,
컴퓨터에 NVIDIA GPU가 있고 [CUDA](https://developer.nvidia.com/cuda-downloads)가 설치되어 있다면,
모든 준비가 끝난 것입니다.
컴퓨터에 GPU가 전혀 없더라도,
아직 걱정할 필요는 없습니다.
처음 몇 장을 진행하기에는
CPU의 성능만으로도 충분하고도 남습니다.
다만 더 큰 모델을 실행하기 전에는
GPU에 접근하고 싶어진다는 점만 기억해 두세요.


:begin_tab:`mxnet`

GPU를 지원하는 MXNet 버전을 설치하려면,
어떤 버전의 CUDA가 설치되어 있는지 먼저 확인해야 합니다.
`nvcc --version`이나
`cat /usr/local/cuda/version.txt`를 실행해 확인할 수 있습니다.
CUDA 11.2가 설치되어 있다고 가정하면,
다음 명령을 실행합니다.

```bash
# For macOS and Linux users
pip install mxnet-cu112==1.9.1

# For Windows users
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


마지막 숫자는 CUDA 버전에 맞게 바꿀 수 있습니다. 예를 들어
CUDA 10.1의 경우 `cu101`, CUDA 9.0의 경우 `cu90`을 사용합니다.


컴퓨터에 NVIDIA GPU나 CUDA가 없다면,
다음과 같이 CPU 버전을
설치할 수 있습니다.

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

PyTorch(명시된 버전은 집필 시점에 테스트된 버전입니다)는 CPU 또는 GPU 지원과 함께 다음과 같이 설치할 수 있습니다.

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
TensorFlow는 CPU 또는 GPU 지원과 함께 다음과 같이 설치할 수 있습니다.

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
JAX와 Flax는 CPU 또는 GPU 지원과 함께 다음과 같이 설치할 수 있습니다.

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


컴퓨터에 NVIDIA GPU나 CUDA가 없다면,
다음과 같이 CPU 버전을
설치할 수 있습니다.

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


다음 단계는 이 책 전반에 걸쳐 자주 사용되는
함수와 클래스를 캡슐화하기 위해
저희가 개발한
`d2l` 패키지를 설치하는 것입니다.

```bash
pip install d2l==1.0.3
```


## 코드 다운로드 및 실행

다음으로, 책의 각 코드 블록을 실행할 수 있도록
노트북을 다운로드해야 합니다.
[D2L.ai 웹사이트](https://d2l.ai/)의 어떤 HTML 페이지에서든
상단의 "Notebooks" 탭을 클릭하기만 하면
코드를 다운로드한 뒤 압축을 풀 수 있습니다.
또는 다음과 같이 명령줄에서
노트북을 받아올 수도 있습니다.

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

`unzip`이 아직 설치되어 있지 않다면, 먼저 `sudo apt-get install unzip`을 실행하세요.
이제 다음을 실행하여 Jupyter Notebook 서버를 시작할 수 있습니다.

```bash
jupyter notebook
```


이 시점에 웹 브라우저에서 http://localhost:8888 을
열 수 있습니다(이미 자동으로 열려 있을 수도 있습니다).
그런 다음 책의 각 절에 해당하는 코드를 실행할 수 있습니다.
새 명령줄 창을 열 때마다,
D2L 노트북을 실행하거나
(딥러닝 프레임워크 또는 `d2l` 패키지 같은)
패키지를 업데이트하기 전에
런타임 환경을 활성화하기 위해
`conda activate d2l`을 실행해야 합니다.
환경에서 빠져나가려면
`conda deactivate`를 실행합니다.


:begin_tab:`mxnet`
[토론](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[토론](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[토론](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[토론](https://discuss.d2l.ai/t/17964)
:end_tab:
