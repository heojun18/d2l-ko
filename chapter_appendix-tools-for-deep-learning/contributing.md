# 이 책에 기여하기
:label:`sec_how_to_contribute`

[독자](https://github.com/d2l-ai/d2l-en/graphs/contributors)들의 기여는 저희가 이 책을 개선하는 데 도움이 됩니다. 오타, 오래된 링크, 누락된 인용이라고 생각되는 곳, 코드가 우아해 보이지 않거나 설명이 명확하지 않은 곳을 발견하시면, 다시 기여해 주셔서 저희가 독자들을 돕는 데 도움을 주세요. 일반 책에서는 인쇄 간격(따라서 오타 수정 간격)을 연 단위로 측정할 수 있지만, 이 책에서는 일반적으로 개선 사항을 반영하는 데 몇 시간에서 며칠이 걸립니다. 이 모든 것은 버전 관리와 지속적 통합(CI) 테스트 덕분에 가능합니다. 그렇게 하려면 GitHub 저장소에 [pull request](https://github.com/d2l-ai/d2l-en/pulls)를 제출해야 합니다. pull request가 저자들에 의해 코드 저장소에 병합되면, 여러분은 기여자가 됩니다.

## 작은 변경 사항 제출하기

가장 흔한 기여는 한 문장을 편집하거나 오타를 고치는 것입니다. 저희는 [GitHub 저장소](https://github.com/d2l-ai/d2l-en)에서 소스 파일을 찾아 파일을 직접 편집하는 것을 권장합니다. 예를 들어, [Find file](https://github.com/d2l-ai/d2l-en/find/master) 버튼(:numref:`fig_edit_file`)을 통해 파일을 검색하여 소스 파일(마크다운 파일)을 찾을 수 있습니다. 그런 다음 오른쪽 상단의 "Edit this file" 버튼을 클릭하여 마크다운 파일을 변경하세요.

![GitHub에서 파일 편집하기.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

완료한 후, 페이지 하단의 "Propose file change" 패널에 변경 설명을 작성하고 "Propose file change" 버튼을 클릭합니다. 그러면 변경 사항을 검토할 수 있는 새 페이지로 리디렉션됩니다(:numref:`fig_git_createpr`). 모든 것이 좋다면, "Create pull request" 버튼을 클릭하여 pull request를 제출할 수 있습니다.

## 큰 변경 사항 제안하기

만약 텍스트나 코드의 큰 부분을 업데이트할 계획이라면, 이 책이 사용하는 형식에 대해 조금 더 알아야 합니다. 소스 파일은 방정식, 이미지, 장, 인용을 참조하는 것과 같은 [D2L-Book](http://book.d2l.ai/user/markdown.html) 패키지를 통한 일련의 확장과 함께 [마크다운 형식](https://daringfireball.net/projects/markdown/syntax)을 기반으로 합니다. 어떤 마크다운 편집기를 사용해서든 이 파일들을 열고 변경할 수 있습니다.

코드를 변경하고 싶다면, :numref:`sec_jupyter`에 설명된 대로 Jupyter 노트북을 사용하여 이 마크다운 파일들을 여는 것을 권장합니다. 그래야 변경 사항을 실행하고 테스트할 수 있습니다. 저희 CI 시스템이 업데이트한 절을 실행하여 출력을 생성하므로, 변경 사항을 제출하기 전에 모든 출력을 지우는 것을 기억하세요.

일부 절은 여러 프레임워크 구현을 지원할 수도 있습니다.
새 코드 블록을 추가하는 경우, 시작 줄에서 `%%tab`을 사용하여 이 블록을 표시하세요. 예를 들어,
PyTorch 코드 블록의 경우 `%%tab pytorch`, TensorFlow 코드 블록의 경우 `%%tab tensorflow`, 또는 모든 구현에 대한 공유 코드 블록의 경우 `%%tab all`을 사용합니다. 더 자세한 정보는 `d2lbook` 패키지를 참조하실 수 있습니다.

## 큰 변경 사항 제출하기

저희는 큰 변경 사항을 제출하기 위해 표준 Git 프로세스를 사용하는 것을 권장합니다. 간단히 말해, 프로세스는 :numref:`fig_contribute`에 설명된 대로 작동합니다.

![책에 기여하기.](../img/contribute.svg)
:label:`fig_contribute`

저희가 단계를 자세히 안내해 드리겠습니다. 이미 Git에 익숙하다면 이 절을 건너뛰셔도 됩니다. 구체적으로 기여자의 사용자 이름이 "astonzhang"이라고 가정하겠습니다.

### Git 설치하기

Git 오픈 소스 책은 [Git 설치 방법](https://git-scm.com/book/en/v2)을 설명합니다. 이는 일반적으로 Ubuntu Linux에서는 `apt install git`, macOS에서는 Xcode 개발자 도구 설치, 또는 GitHub의 [데스크톱 클라이언트](https://desktop.github.com)를 사용하여 작동합니다. GitHub 계정이 없다면 가입해야 합니다.

### GitHub에 로그인하기

브라우저에 책의 코드 저장소 [주소](https://github.com/d2l-ai/d2l-en/)를 입력하세요. :numref:`fig_git_fork`의 오른쪽 상단 빨간 박스에 있는 `Fork` 버튼을 클릭하여 이 책의 저장소 사본을 만드세요. 이는 이제 *여러분의 사본*이며, 원하는 어떤 방식으로든 변경할 수 있습니다.

![코드 저장소 페이지.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


이제, 이 책의 코드 저장소는 :numref:`fig_git_forked`의 왼쪽 상단에 표시된 `astonzhang/d2l-en`처럼 여러분의 사용자 이름으로 fork(즉, 복사)될 것입니다.

![Fork된 코드 저장소.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### 저장소 복제하기

저장소를 복제하려면(즉, 로컬 사본을 만들려면) 저장소 주소를 가져와야 합니다. :numref:`fig_git_clone`의 초록색 버튼이 이를 표시합니다. 이 fork를 더 오래 유지하기로 결정한 경우 로컬 사본이 메인 저장소와 최신 상태인지 확인하세요. 지금은 시작하기 위해 :ref:`chap_installation`의 지침을 따르기만 하세요. 주된 차이점은 이제 저장소의 *여러분의 fork*를 다운로드한다는 것입니다.

![저장소 복제하기.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# your_github_username을 여러분의 GitHub 사용자 이름으로 교체하세요
git clone https://github.com/your_github_username/d2l-en.git
```


### 편집 및 푸시

이제 책을 편집할 시간입니다. :numref:`sec_jupyter`의 지침에 따라 Jupyter 노트북에서 편집하는 것이 가장 좋습니다. 변경 사항을 적용하고 문제가 없는지 확인하세요. `~/d2l-en/chapter_appendix-tools-for-deep-learning/contributing.md` 파일에서 오타를 수정했다고 가정해 보겠습니다.
그러면 어떤 파일이 변경되었는지 확인할 수 있습니다.

이 시점에서 Git은 `chapter_appendix-tools-for-deep-learning/contributing.md` 파일이 수정되었다고 알려 줄 것입니다.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix-tools-for-deep-learning/contributing.md
```


이것이 원하는 것임을 확인한 후 다음 명령을 실행합니다.

```
git add chapter_appendix-tools-for-deep-learning/contributing.md
git commit -m 'Fix a typo in git documentation'
git push
```


그러면 변경된 코드가 저장소의 개인 fork에 있게 됩니다. 변경 사항의 추가를 요청하려면, 책의 공식 저장소에 대한 pull request를 생성해야 합니다.

### Pull Request 제출하기

:numref:`fig_git_newpr`에 표시된 것처럼, GitHub의 fork된 저장소로 이동하여 "New pull request"를 선택하세요. 그러면 여러분의 편집과 책의 메인 저장소에 있는 현재 내용 사이의 변경 사항을 보여 주는 화면이 열립니다.

![새 pull request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`


마지막으로, :numref:`fig_git_createpr`에 표시된 버튼을 클릭하여 pull request를 제출하세요. pull request에 만든 변경 사항을 설명해 주세요.
이렇게 하면 저자들이 검토하고 책에 병합하기가 더 쉬워집니다. 변경 사항에 따라 즉시 수락될 수도, 거부될 수도, 또는 보다 가능성이 높게는 변경 사항에 대한 피드백을 받을 수도 있습니다. 일단 피드백을 반영하면 준비가 된 것입니다.

![Pull request 생성하기.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`


## 요약

* GitHub를 사용하여 이 책에 기여할 수 있습니다.
* 작은 변경 사항의 경우 GitHub에서 파일을 직접 편집할 수 있습니다.
* 큰 변경 사항의 경우, 저장소를 fork하고 로컬에서 편집한 다음, 준비가 되었을 때만 다시 기여하세요.
* Pull request는 기여가 묶이는 방식입니다. 거대한 pull request는 이해하고 반영하기 어렵게 만드니 가급적 제출하지 마세요. 여러 개의 더 작은 것을 보내는 것이 좋습니다.


## 연습문제

1. `d2l-ai/d2l-en` 저장소에 별을 누르고 fork해 보세요.
1. 개선이 필요한 것(예: 참고 자료 누락)을 발견하면, pull request를 제출해 보세요.
1. 새 브랜치를 사용하여 pull request를 만드는 것이 일반적으로 더 좋은 방법입니다. [Git 브랜칭](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)으로 이를 수행하는 방법을 배워 보세요.

[토론](https://discuss.d2l.ai/t/426)
