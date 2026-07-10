$$Phase 1 - Step 1$$

# Mac OS (Apple Silicon) AI 개발 환경 완벽 세팅 가이드

인공지능(AI) 개발을 시작하기 위한 첫걸음, 바로 **견고한 개발 환경을 구축**하는 것입니다. 특히 최근의 Mac(M1, M2, M3 등 Apple Silicon) 환경은 뛰어난 전성비와 성능을 자랑하지만, 기존 Intel 기반의 세팅 방식과는 조금 다른 접근이 필요합니다.

이 글에서는 Mac OS 환경에서 파이썬(Python)을 기반으로, GPU 가속(MPS)까지 완벽하게 활용할 수 있는 최적의 AI 개발 환경 세팅 방법을 단계별로 알아봅니다.

## 1. 터미널의 친구, Homebrew 설치하기
개발에 필요한 각종 패키지와 소프트웨어를 손쉽게 설치하고 관리하게 해주는 패키지 관리자 Homebrew를 가장 먼저 설치합니다.

1. `Terminal` 앱을 실행합니다. (Spotlight 검색 `cmd + space` 에서 'terminal' 입력)
2. 아래 명령어를 복사하여 터미널에 붙여넣고 실행(Enter)합니다.
```
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
```
3.설치가 완료된 후, 터미널 화면 마지막에 나오는 `Next steps`: 아래의 두 줄(환경 변수 설정 명령어)을 복사하여 실행합니다. (Apple Silicon Mac에만 해당)
4. 설치 확인: 터미널에 `brew --version`을 입력하여 버전 정보가 나오면 성공입니다.

## 2. 파이썬 환경 관리: Miniforge 설치 (권장)
AI 개발 시 프로젝트마다 필요한 라이브러리 버전이 다를 수 있습니다. 이를 격리하여 관리하기 위해 '가상 환경'을 사용합니다.

과거에는 Anaconda를 많이 사용했지만, Apple Silicon 환경에서는 네이티브 지원이 더 뛰어나고 가벼운 Miniforge 사용을 강력히 권장합니다.

1. Homebrew를 이용해 Miniforge를 설치합니다.
```
brew install miniforge
```
2. 설치가 끝나면 터미널을 완전히 종료(`cmd + Q`)했다가 다시 실행합니다.
3. 터미널 명령줄 앞에 (`base`)라는 글자가 생겼다면 Miniforge가 성공적으로 활성화된 것입니다.

## 3. 나만의 AI 가상 환경 만들기
이제 본격적인 AI 프로젝트를 위한 독립적인 방(가상 환경)을 만들어보겠습니다. 이름은 `ai_env`로 하고, 파이썬 버전은 가장 안정적인 `3.10`으로 설정하겠습니다.

1. 가상 환경 생성:
```
conda create -n ai_env python=3.10
```
(중간에 `Proceed (\[y\]/n)?` 가 나오면 `y`를 입력하고 Enter)

2. 가상 환경 활성화:
```
conda activate ai_env
```
(명령줄 앞이 (base)에서 (ai_env)로 바뀐 것을 확인하세요.)

💡 **팁**: 프로젝트 작업을 끝내고 기본 환경으로 돌아가려면 conda deactivate를 입력합니다.

## 4. 필수 라이브러리 및 PyTorch (MPS 지원) 설치
가상 환경이 활성화된 상태((`ai_env`))에서, 데이터 분석과 딥러닝에 필요한 핵심 라이브러리들을 설치합니다.

#### 4.1 데이터 사이언스 기본 패키지
수치 연산, 데이터 처리, 시각화, 그리고 코딩 환경(Jupyter)을 설치합니다.
```
conda install numpy pandas matplotlib seaborn jupyter jupyterlab
```

#### 4.2 PyTorch 설치 (핵심!)
Apple Silicon의 GPU(Metal Performance Shaders, MPS)를 활용하기 위해 공식 가이드에 맞게 PyTorch를 설치합니다. MPS를 사용하면 CPU만 사용할 때보다 딥러닝 모델 학습 속도를 획기적으로 높일 수 있습니다.
```
conda install pytorch torchvision torchaudio -c pytorch
```


## 5. 최종 점검: GPU 가속(MPS) 확인하기
모든 설치가 끝났습니다! 이제 실제로 Mac의 GPU가 PyTorch와 잘 연동되어 작동하는지 테스트해 볼 차례입니다.

1. 터미널(`ai_env` 상태)에서 `jupyter lab`을 입력하여 실행합니다.
2. 웹 브라우저가 열리면, 우측의 `Notebook` 하위의 `Python 3`를 클릭하여 새 노트북을 만듭니다.
3. 아래 코드를 복사하여 셀에 붙여넣고 실행(`Shift + Enter)`해 보세요.

```python
import torch

print(f"PyTorch Version: {torch.__version__}")

# MPS(Metal Performance Shaders) 지원 여부 확인
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("🚀 성공! Apple Silicon GPU(MPS)를 사용할 수 있습니다.")
    
    # 간단한 텐서 생성 및 MPS 장치로 이동 테스트
    x = torch.ones(1, device=mps_device)
    print(f"텐서 x: {x}")
else:
    print("⚠️ MPS를 사용할 수 없습니다. 일반 CPU를 사용합니다.")
```

출력 결과에 "🚀 성**공! Apple Silicon GPU(MPS)를 사용할 수 있습니다**." 가 보인다면, 완벽한 AI 개발 환경이 갖춰진 것입니다! 🎉

## 마치며

축하합니다! 가장 귀찮지만 중요한 첫 단추인 '환경 세팅'을 무사히 마쳤습니다. 다음 Step에서는 오늘 설치한 NumPy와 Pandas를 활용하여 **실제 데이터를 요리(전처리)하는 방법**을 알아보겠습니다.
