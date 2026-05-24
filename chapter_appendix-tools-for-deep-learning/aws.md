# AWS EC2 인스턴스 사용하기
:label:`sec_aws`

이 절에서는 순수한 Linux 머신에 모든 라이브러리를 설치하는 방법을 보여 드리겠습니다. :numref:`sec_sagemaker`에서 Amazon SageMaker를 사용하는 방법을 논의했음을 기억하시기 바랍니다. 인스턴스를 직접 구축하면 AWS에서 비용이 더 적게 듭니다. 이 안내는 세 단계로 구성됩니다.

1. AWS EC2에서 GPU Linux 인스턴스를 요청합니다.
1. CUDA를 설치합니다(또는 CUDA가 사전 설치된 Amazon Machine Image를 사용합니다).
1. 책의 코드를 실행하기 위한 딥러닝 프레임워크와 기타 라이브러리를 설치합니다.

이 과정은 약간의 수정과 함께 다른 인스턴스(및 다른 클라우드)에도 적용됩니다. 진행하기 전에 AWS 계정을 만들어야 합니다. 자세한 내용은 :numref:`sec_sagemaker`를 참조하세요.


## EC2 인스턴스 생성 및 실행하기

AWS 계정에 로그인한 후 "EC2"(:numref:`fig_aws`)를 클릭하여 EC2 패널로 이동합니다.

![EC2 콘솔 열기.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2`는 EC2 패널을 보여 줍니다.

![EC2 패널.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### 위치 사전 설정
지연 시간을 줄이기 위해 근처의 데이터 센터를 선택합니다(예: "Oregon", :numref:`fig_ec2`의 오른쪽 상단 빨간 박스로 표시됨). 만약 중국에 위치해 있다면,
서울이나 도쿄와 같은 근처 아시아 태평양 지역을 선택할 수 있습니다. 일부 데이터 센터에는 GPU 인스턴스가 없을 수도 있다는 점에 주의하세요.


### 한도 늘리기

인스턴스를 선택하기 전에,
:numref:`fig_ec2`와 같이 왼쪽 바의 "Limits" 라벨을 클릭하여 수량 제한이 있는지 확인합니다.
:numref:`fig_limits`는 이러한 제한의 예를 보여 줍니다. 이 계정은 현재 해당 지역에 따라 "p2.xlarge" 인스턴스를 열 수 없습니다.
하나 이상의 인스턴스를 열어야 한다면, "Request limit increase" 링크를 클릭하여
더 높은 인스턴스 할당량을 신청하세요.
일반적으로 신청 처리에는 영업일 기준 하루가 소요됩니다.

![인스턴스 수량 제한.](../img/limits.png)
:width:`700px`
:label:`fig_limits`


### 인스턴스 시작하기

다음으로, :numref:`fig_ec2`의 빨간 박스로 표시된 "Launch Instance" 버튼을 클릭하여 인스턴스를 시작합니다.

먼저 적합한 Amazon Machine Image(AMI)를 선택합니다. Ubuntu 인스턴스를 선택합니다(:numref:`fig_ubuntu`).


![AMI 선택하기.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2는 선택할 수 있는 다양한 인스턴스 구성을 제공합니다. 이는 초보자에게는 다소 부담스러울 수 있습니다. :numref:`tab_ec2`는 다양한 적합한 머신을 나열합니다.

:다양한 EC2 인스턴스 타입
:label:`tab_ec2`

| 이름  | GPU         | 비고                          |
|------|-------------|-------------------------------|
| g2   | Grid K520   | 오래됨                        |
| p2   | Kepler K80  | 오래되었지만 스팟으로 종종 저렴 |
| g3   | Maxwell M60 | 좋은 절충안                   |
| p3   | Volta V100  | FP16에 대한 높은 성능          |
| p4   | Ampere A100 | 대규모 훈련을 위한 높은 성능   |
| g4   | Turing T4   | FP16/INT8 추론 최적화         |


이 모든 서버는 사용된 GPU 수를 나타내는 여러 형태로 제공됩니다. 예를 들어, p2.xlarge는 GPU 1개, p2.16xlarge는 GPU 16개와 더 많은 메모리를 가집니다. 자세한 내용은 [AWS EC2 문서](https://aws.amazon.com/ec2/instance-types/) 또는 [요약 페이지](https://www.ec2instances.info)를 참조하세요. 예시를 위해서는 p2.xlarge면 충분합니다(:numref:`fig_p2x`의 빨간 박스로 표시).

![인스턴스 선택하기.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

적절한 드라이버와 GPU 지원 딥러닝 프레임워크가 있는 GPU 지원 인스턴스를 사용해야 한다는 점에 유의하세요. 그렇지 않으면 GPU를 사용하는 이점을 전혀 볼 수 없습니다.

다음으로 인스턴스에 접근하는 데 사용할 키 페어를 선택합니다.
키 페어가 없다면 :numref:`fig_keypair`에서 "Create new key pair"를 클릭하여 키 페어를 생성합니다. 그 후
이전에 생성한 키 페어를 선택할 수 있습니다.
새 키 페어를 생성한 경우 다운로드하여 안전한 위치에 보관해야 합니다. 이것이
서버에 SSH로 접근할 수 있는 유일한 방법입니다.

![키 페어 선택하기.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

이 예제에서 저희는 "Network settings"의 기본 구성을 그대로 유지하겠습니다("Edit" 버튼을 클릭하여 서브넷 및 보안 그룹과 같은 항목을 구성합니다). 기본 하드 디스크 크기를 64 GB로 늘리기만 합니다(:numref:`fig_disk`). CUDA 자체만으로도 이미 4 GB를 차지한다는 점에 유의하세요.

![하드 디스크 크기 수정하기.](../img/disk.png)
:width:`700px`
:label:`fig_disk`


"Launch Instance"를 클릭하여 생성된 인스턴스를 시작합니다.
이 인스턴스의 상태를 확인하려면
:numref:`fig_launching`에 표시된 인스턴스 ID를 클릭합니다.

![인스턴스 ID 클릭하기.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### 인스턴스에 연결하기

:numref:`fig_connect`에 표시된 것처럼, 인스턴스 상태가 초록색으로 바뀐 후 인스턴스를 마우스 오른쪽 버튼으로 클릭하고 `Connect`를 선택하여 인스턴스 접근 방법을 확인합니다.

![인스턴스 접근 방법 보기.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

이것이 새 키라면 SSH가 작동하려면 공개적으로 볼 수 없도록 해야 합니다. `D2L_key.pem`을 저장한 폴더로 이동하여
다음 명령을 실행하여
키를 공개적으로 볼 수 없도록 만듭니다.

```bash
chmod 400 D2L_key.pem
```


![인스턴스 접근 및 시작 방법 보기.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


이제 :numref:`fig_chmod`의 아래쪽 빨간 박스에서 SSH 명령을 복사하여 명령줄에 붙여 넣습니다.

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```


명령줄에서 "Are you sure you want to continue connecting (yes/no)"라는 메시지가 표시되면, "yes"를 입력하고 Enter 키를 눌러 인스턴스에 로그인합니다.

이제 서버가 준비되었습니다.


## CUDA 설치하기

CUDA를 설치하기 전에, 최신 드라이버로 인스턴스를 업데이트하시기 바랍니다.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


여기서 저희는 CUDA 12.1을 다운로드합니다. NVIDIA의 [공식 저장소](https://developer.nvidia.com/cuda-toolkit-archive)를 방문하여 :numref:`fig_cuda`에 표시된 것처럼 다운로드 링크를 찾으세요.

![CUDA 12.1 다운로드 주소 찾기.](../img/cuda121.png)
:width:`500px`
:label:`fig_cuda`

지침을 복사하여 터미널에 붙여 넣어 CUDA 12.1을 설치합니다.

```bash
# 링크와 파일명은 변경될 수 있습니다
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```


프로그램 설치 후 다음 명령을 실행하여 GPU를 확인합니다.

```bash
nvidia-smi
```


마지막으로, 다른 라이브러리가 CUDA를 찾을 수 있도록 라이브러리 경로에 CUDA를 추가합니다. 예를 들어 `~/.bashrc` 끝에 다음 줄을 추가합니다.

```bash
export PATH="/usr/local/cuda-12.1/bin:$PATH"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-12.1/lib64
```


## 코드 실행을 위한 라이브러리 설치하기

이 책의 코드를 실행하려면,
EC2 인스턴스에서 Linux 사용자를 위한 :ref:`chap_installation`의 단계를 따르고
원격 Linux 서버에서 작업할 때
다음 팁을 사용하세요.

* Miniconda 설치 페이지에서 bash 스크립트를 다운로드하려면, 다운로드 링크를 마우스 오른쪽 버튼으로 클릭하고 "Copy Link Address"를 선택한 다음, `wget [복사된 링크 주소]`를 실행합니다.
* `~/miniconda3/bin/conda init`을 실행한 후, 현재 셸을 닫고 다시 열 필요 없이 `source ~/.bashrc`를 실행할 수 있습니다.


## 원격으로 Jupyter 노트북 실행하기

Jupyter 노트북을 원격으로 실행하려면 SSH 포트 포워딩을 사용해야 합니다. 결국 클라우드의 서버에는 모니터나 키보드가 없으니까요. 이를 위해, 데스크톱(또는 노트북)에서 다음과 같이 서버에 로그인합니다.

```
# 이 명령은 로컬 명령줄에서 실행해야 합니다
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```


다음으로, EC2 인스턴스에서 다운로드한 이 책 코드의 위치로 이동한 후,
다음을 실행합니다.

```
conda activate d2l
jupyter notebook
```


:numref:`fig_jupyter`는 Jupyter 노트북을 실행한 후의 가능한 출력을 보여 줍니다. 마지막 행은 포트 8888의 URL입니다.

![Jupyter 노트북 실행 후 출력. 마지막 행은 포트 8888의 URL입니다.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

8889 포트로 포트 포워딩을 사용했기 때문에,
:numref:`fig_jupyter`의 빨간 박스 안 마지막 행을 복사한 다음,
URL의 "8888"을 "8889"로 교체하고
로컬 브라우저에서 엽니다.


## 사용하지 않는 인스턴스 닫기

클라우드 서비스는 사용 시간에 따라 청구되므로, 사용하지 않는 인스턴스는 닫아야 합니다. 다음과 같은 대안이 있다는 점에 유의하세요.

* 인스턴스를 "중지"(Stopping)하면 다시 시작할 수 있습니다. 이는 일반 서버의 전원을 끄는 것과 비슷합니다. 그러나 중지된 인스턴스도 보존되는 하드 디스크 공간에 대해 소액이 청구됩니다.
* 인스턴스를 "종료"(Terminating)하면 관련된 모든 데이터가 삭제됩니다. 여기에는 디스크도 포함되므로 다시 시작할 수 없습니다. 향후에 필요 없을 것임을 알 때만 이렇게 하세요.

인스턴스를 더 많은 인스턴스를 위한 템플릿으로 사용하려면,
:numref:`fig_connect`의 예시를 마우스 오른쪽 버튼으로 클릭하고 "Image" $\rightarrow$
"Create"를 선택하여 인스턴스의 이미지를 만듭니다. 완료되면,
"Instance State" $\rightarrow$ "Terminate"를 선택하여 인스턴스를 종료합니다. 다음번에
이 인스턴스를 사용하고 싶을 때는, 이 절의 단계를 따라
저장된 이미지를 기반으로 인스턴스를 만들 수 있습니다. 유일한 차이점은
:numref:`fig_ubuntu`에 표시된 "1. Choose AMI"에서 왼쪽의 "My AMIs" 옵션을 사용하여 저장된 이미지를 선택해야 한다는 것입니다.
생성된 인스턴스는 이미지 하드 디스크에 저장된 정보를 보존합니다.
예를 들어, CUDA와 기타 런타임 환경을 다시 설치할 필요가 없습니다.


## 요약

* 우리는 우리 자신의 컴퓨터를 구매하고 구축할 필요 없이 필요에 따라 인스턴스를 시작하고 중지할 수 있습니다.
* GPU 지원 딥러닝 프레임워크를 사용하기 전에 CUDA를 설치해야 합니다.
* 원격 서버에서 Jupyter 노트북을 실행하기 위해 포트 포워딩을 사용할 수 있습니다.


## 연습문제

1. 클라우드는 편리함을 제공하지만 저렴하지는 않습니다. 비용을 줄이는 방법을 알아보기 위해 [스팟 인스턴스](https://aws.amazon.com/ec2/spot/)를 시작하는 방법을 찾아보세요.
1. 다양한 GPU 서버를 실험해 보세요. 얼마나 빠른가요?
1. 멀티 GPU 서버를 실험해 보세요. 얼마나 잘 확장할 수 있나요?


[토론](https://discuss.d2l.ai/t/423)
