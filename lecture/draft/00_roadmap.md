# 🚀 단계별 AI 개발자 블로그 튜토리얼 로드맵
Mac OS 환경에서 파이썬을 기반으로, 실무에서 가장 많이 쓰이는 기술과 알고리즘을 직접 구현하며 익힐 수 있는 '실무형 AI 개발자 성장 로드맵'을 제안해 드립니다. 

## Phase 1: 개발 환경 구축 및 데이터 다루기 (기초 공사)

AI 개발의 8할은 데이터 전처리와 환경 설정입니다. 특히 Mac(Apple Silicon) 환경에 맞춘 최적화 세팅을 첫 글로 작성하면 많은 유입을 기대할 수 있습니다.

#### Step 1: Mac OS AI 개발 환경 완벽 세팅**
- Homebrew 및 Miniforge(또는 Anaconda) 설치.
- 가상환경 구성 및 Jupyter Notebook 연동.
- **실무 팁:** Apple Silicon(M칩)에서 GPU 가속을 사용하기 위한 MPS(Metal Performance Shaders) 세팅 방법.

#### Step 2: 데이터 사이언스 필수 라이브러리 마스터**
- NumPy로 행렬 연산 이해하기 (선형대수 기초).
- Pandas로 실전 데이터 셋(CSV) 불러오고 결측치 처리하기.
- Matplotlib/Seaborn으로 데이터 시각화하기.

## Phase 2: 머신러닝 핵심 알고리즘 구현 (원리 이해)
실무에서는 Scikit-learn을 쓰지만, 알고리즘을 직접 파이썬으로 구현해 보는 것은 훌륭한 블로그 콘텐츠이자 면접 대비용 지식이 됩니다.

#### Step 3: 선형 회귀(Linear Regression)와 경사하강법(Gradient Descent)**
- 수식 없이 이해하는 경사하강법 개념.
- **구현:** 파이썬으로 경사하강법 직접 코딩하여 1차 함수 $y = wx + b$의 최적 가중치 찾기.

#### Step 4: 분류의 기초, 로지스틱 회귀(Logistic Regression)**
- Sigmoid 함수의 역할과 분류 문제의 이해.
- **구현:** 이진 분류기 만들기 및 평가지표(Accuracy, Precision, Recall) 직접 계산해보기.

#### Step 5: 앙상블 기법 (Random Forest & XGBoost)**
- 실무 정형 데이터 대회(Kaggle 등)를 휩쓰는 트리 기반 모델 이해.
- **실전:** Scikit-learn을 활용한 타이타닉(Titanic) 생존자 예측 모델 만들기.

## Phase 3: 딥러닝과 PyTorch (본격적인 AI 모델링)
실무 점유율이 압도적인 **PyTorch**를 메인 프레임워크로 선택하는 것을 추천합니다.

#### Step 6: PyTorch 텐서(Tensor)와 Autograd 이해하기**
- NumPy 배열을 PyTorch 텐서로 변환하고 MPS 장치(Mac GPU)에 할당하기.

#### Step 7: 인공신경망(MLP) 밑바닥부터 구축하기**
- 퍼셉트론(Perceptron)의 한계와 다층 퍼셉트론(MLP)의 등장.
-  **구현:** PyTorch `nn.Module`을 상속받아 간단한 손글씨(MNIST) 분류 모델 만들기.

#### Step 8: 최적화(Optimizer)와 손실 함수(Loss Function)**
- SGD, Adam, RMSprop 등 다양한 옵티마이저의 차이 시각화.
- Cross Entropy Loss 등 문제에 맞는 손실 함수 선택법.

## Phase 4: 최신 트렌드 실무 (비전, NLP, 그리고 LLM)
가장 수요가 많은 분야를 선택해 깊게 파고드는 단계입니다. 최근 실무에서는 LLM(거대 언어 모델) 활용 능력이 필수적입니다.

#### Step 9: 컴퓨터 비전(CV) - 합성곱 신경망(CNN) 구현**
- 이미지 특징을 추출하는 필터와 풀링(Pooling)의 이해.
- **구현:** 전이 학습(Transfer Learning)을 활용해 ResNet 모델로 개와 고양이 분류기 만들기.


#### Step 10: 자연어 처리(NLP)와 트랜스포머(Transformer)**
- 순환 신경망(RNN/LSTM)의 개념과 한계.
- 현대 AI의 핵심인 **어텐션(Attention)** 메커니즘 뜯어보기.


#### Step 11: 🚀 실무 0순위, LLM과 RAG (검색 증강 생성) 구축하기**
- OpenAI API 또는 로컬 LLM(Ollama 등) 연동하기.
- **구현:** LangChain을 활용해 '내 블로그 글을 읽고 답변해 주는 커스텀 AI 챗봇(RAG)' 만들기.

## Phase 5: MLOps와 배포 (엔지니어링의 완성)
모델을 만드는 것을 넘어, 서비스로 배포할 줄 아는 개발자가 실무에서 환영받습니다.

#### Step 12: 모델 서빙(Model Serving) 기초**
- FastAPI를 이용해 내가 만든 딥러닝 모델을 REST API로 만들기.

#### Step 13: Docker를 활용한 환경 격리**
 Mac 환경에서 작업한 서버를 Docker 이미지로 빌드하고 컨테이너로 띄우기.


이 로드맵은 기초 알고리즘의 '직접 구현'으로 시작해, 최신 실무 트렌드인 'LLM/RAG 구현'과 '서버 배포'까지 이어지도록 기획되었습니다.
