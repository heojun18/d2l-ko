# Jupyter 노트북 사용하기
:label:`sec_jupyter`


이 절에서는 Jupyter 노트북을 사용하여
이 책의 각 절에 있는 코드를 편집하고 실행하는 방법을 설명합니다.
:ref:`chap_installation`에 설명된 대로 Jupyter를 설치하고
코드를 다운로드해 두셨는지 확인하시기 바랍니다.
Jupyter에 대해 더 자세히 알고 싶으시다면
[공식 문서](https://jupyter.readthedocs.io/en/latest/)의 훌륭한 튜토리얼을 참고하세요.


## 로컬에서 코드 편집 및 실행하기

이 책의 코드의 로컬 경로가 `xx/yy/d2l-en/`이라고 가정하겠습니다. 셸을 사용하여 디렉터리를 이 경로로 변경하고(`cd xx/yy/d2l-en`) `jupyter notebook` 명령을 실행합니다. 브라우저가 자동으로 열리지 않는다면 http://localhost:8888 을 열어 보세요. 그러면 :numref:`fig_jupyter00`에 표시된 것처럼 Jupyter 인터페이스와 이 책의 코드를 담고 있는 모든 폴더가 표시됩니다.

![이 책의 코드를 담고 있는 폴더들.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


웹 페이지에 표시되는 폴더를 클릭하여 노트북 파일에 접근할 수 있습니다.
이 파일들은 일반적으로 ".ipynb" 접미사를 가집니다.
간결성을 위해 임시 "test.ipynb" 파일을 만들겠습니다.
클릭한 후 표시되는 내용은
:numref:`fig_jupyter01`에 나와 있습니다.
이 노트북에는 마크다운 셀과 코드 셀이 포함되어 있습니다. 마크다운 셀의 내용은 "This Is a Title"과 "This is text."를 포함합니다.
코드 셀에는 두 줄의 Python 코드가 포함되어 있습니다.

!["text.ipynb" 파일의 마크다운 셀과 코드 셀.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


마크다운 셀을 더블 클릭하여 편집 모드로 들어갑니다.
:numref:`fig_jupyter02`에 표시된 것처럼 셀 끝에 새로운 텍스트 "Hello world."를 추가하세요.

![마크다운 셀 편집하기.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


:numref:`fig_jupyter03`에서 보여 주는 것처럼,
편집한 셀을 실행하려면 메뉴 바에서 "Cell" $\rightarrow$ "Run Cells"를 클릭합니다.

![셀 실행하기.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

실행 후, 마크다운 셀은 :numref:`fig_jupyter04`와 같이 표시됩니다.

![실행 후의 마크다운 셀.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


다음으로 코드 셀을 클릭합니다. :numref:`fig_jupyter05`에 표시된 것처럼 마지막 코드 줄 뒤에 요소를 2로 곱하는 코드를 추가합니다.

![코드 셀 편집하기.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


단축키(기본값: "Ctrl + Enter")를 사용하여 셀을 실행할 수도 있으며, :numref:`fig_jupyter06`에서 출력 결과를 확인할 수 있습니다.

![코드 셀을 실행하여 출력 얻기.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


노트북에 더 많은 셀이 포함되어 있을 때는 메뉴 바에서 "Kernel" $\rightarrow$ "Restart & Run All"을 클릭하여 전체 노트북의 모든 셀을 실행할 수 있습니다. 메뉴 바에서 "Help" $\rightarrow$ "Edit Keyboard Shortcuts"를 클릭하여 단축키를 선호에 맞게 편집할 수 있습니다.

## 고급 옵션

로컬 편집 외에 두 가지가 매우 중요합니다. 마크다운 형식으로 노트북을 편집하는 것과 Jupyter를 원격으로 실행하는 것입니다.
후자는 더 빠른 서버에서 코드를 실행하려는 경우에 중요합니다.
전자는 Jupyter의 기본 ipynb 형식이 내용과는 무관한,
주로 코드가 어떻게 어디서 실행되는지와 관련된
많은 보조 데이터를 저장하기 때문에 중요합니다.
이는 Git에 혼란을 일으켜
기여를 검토하기 매우 어렵게 만듭니다.
다행히도 대안이 있습니다(마크다운 형식의 네이티브 편집).

### Jupyter의 마크다운 파일

이 책의 내용에 기여하고 싶으시다면, GitHub에서 소스 파일(ipynb 파일이 아닌 md 파일)을 수정해야 합니다.
notedown 플러그인을 사용하면
Jupyter에서 md 형식의 노트북을 직접 수정할 수 있습니다.


먼저 notedown 플러그인을 설치하고, Jupyter 노트북을 실행한 다음, 플러그인을 로드합니다.

```
pip install d2l-notedown  # 원본 notedown을 제거해야 할 수도 있습니다.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


Jupyter 노트북을 실행할 때마다 기본적으로 notedown 플러그인을 켜 둘 수도 있습니다.
먼저 Jupyter 노트북 설정 파일을 생성합니다(이미 생성한 경우 이 단계는 건너뛸 수 있습니다).

```
jupyter notebook --generate-config
```


그런 다음 Jupyter 노트북 설정 파일의 끝에 다음 줄을 추가합니다(Linux나 macOS의 경우 보통 `~/.jupyter/jupyter_notebook_config.py` 경로에 있습니다).

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


그 후에는 `jupyter notebook` 명령을 실행하기만 하면 기본적으로 notedown 플러그인이 켜집니다.

### 원격 서버에서 Jupyter 노트북 실행하기

때로는 Jupyter 노트북을 원격 서버에서 실행하고 로컬 컴퓨터의 브라우저를 통해 접근하고 싶을 수 있습니다. 로컬 머신에 Linux나 macOS가 설치되어 있다면(Windows에서도 PuTTY 같은 서드파티 소프트웨어를 통해 이 기능을 지원할 수 있습니다) 포트 포워딩을 사용할 수 있습니다.

```
ssh myserver -L 8888:localhost:8888
```


위의 `myserver` 문자열은 원격 서버의 주소입니다.
그러면 http://localhost:8888 을 사용하여 Jupyter 노트북을 실행하는 원격 서버 `myserver`에 접근할 수 있습니다. AWS 인스턴스에서 Jupyter 노트북을 실행하는 방법은 이 부록의 뒷부분에서 자세히 다루겠습니다.

### 시간 측정

`ExecuteTime` 플러그인을 사용하여 Jupyter 노트북에서 각 코드 셀의 실행 시간을 측정할 수 있습니다.
플러그인을 설치하려면 다음 명령을 사용하세요.

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## 요약

* Jupyter 노트북 도구를 사용하여 이 책의 각 절을 편집하고, 실행하며, 기여할 수 있습니다.
* 포트 포워딩을 사용하여 원격 서버에서 Jupyter 노트북을 실행할 수 있습니다.


## 연습문제

1. 로컬 머신에서 Jupyter 노트북으로 이 책의 코드를 편집하고 실행해 보세요.
1. 포트 포워딩을 통해 *원격으로* Jupyter 노트북으로 이 책의 코드를 편집하고 실행해 보세요.
1. $\mathbb{R}^{1024 \times 1024}$의 두 정사각 행렬에 대해 $\mathbf{A}^\top \mathbf{B}$와 $\mathbf{A} \mathbf{B}$ 연산의 실행 시간을 비교해 보세요. 어느 것이 더 빠른가요?


[토론](https://discuss.d2l.ai/t/421)
