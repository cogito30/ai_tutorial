$$Phase 3 - Step 8$$

# 딥러닝 튜닝의 핵심: 옵티마이저(Optimizer)와 손실 함수(Loss Function)
지난 Step 7에서 우리는 PyTorch를 이용해 손글씨 숫자(MNIST)를 분류하는 인공신경망을 멋지게 완성했습니다.

그때 코드를 유심히 보셨다면, 아래와 같이 두 줄의 코드를 무심코 지나쳤을 것입니다.

```python
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

사실 이 두 줄은 딥러닝 모델의 성능을 결정짓는 가장 핵심적인 요소입니다. 모델이 얼마나 똑똑해질지, 얼마나 빠르게 학습할지가 바로 여기서 결정됩니다. 오늘은 이 '손실 함수(Loss Function)'와 '옵티마이저(Optimizer)'의 정체를 파헤쳐 보겠습니다.

## 1. 손실 함수 (Loss Function): AI의 채점 기준표
학습(Training)이란 모델이 내놓은 예측값과 실제 정답 사이의 **오차를 줄여나가는 과정**입니다. 이때 '오차를 어떻게 계산할 것인가?'를 정의하는 것이 바로 손실 함수입니다. 문제의 종류에 따라 채점 방식이 달라져야 합니다.

#### 📍 대표적인 손실 함수 3가지
1. MSE (Mean Squared Error) / L2 Loss

- **언제 쓰나요?** 연속적인 값을 예측하는 회귀(Regression) 문제 (예: 집값 예측, 주식 가격 예측)
- **원리**: 정답과 예측값의 차이를 제곱하여 평균을 냅니다. 오차가 클수록 페널티를 크게 줍니다.
- **PyTorch 코드**: nn.MSELoss()

2. BCE (Binary Cross Entropy)
- **언제 쓰나요?** 정답이 0 아니면 1인 이진 분류(Binary Classification) 문제 (예: 스팸 메일 구분, 개/고양이 분류)
- **원리**: 모델의 출력값(시그모이드 함수를 통과한 0~1 사이의 확률)과 실제 정답(0 또는 1) 간의 정보량 차이를 계산합니다.
- **PyTorch 코드**: nn.BCELoss() 또는 nn.BCEWithLogitsLoss()

3. Cross Entropy
- **언제 쓰나요?** 정답이 3개 이상인 다중 분류(Multi-class Classification) 문제 (예: 0~9 손글씨 분류, 1000가지 사물 인식)
- **원리**: 모델의 출력값을 '소프트맥스(Softmax)' 함수를 통해 각 클래스별 확률로 변환한 뒤, 정답 클래스에 대한 확률을 극대화하도록 유도합니다.
- **PyTorch 코드**: nn.CrossEntropyLoss()

💡 **실무 꿀팁**: 분류 문제를 풀고 있다면 CrossEntropyLoss를, 수치를 예측하고 있다면 MSELoss를 선택하는 것이 가장 기본적인 공식입니다.

## 2. 옵티마이저 (Optimizer): 산 정상에서 길을 찾는 방법
손실 함수로 오차를 구했다면, 이제 가중치(w, b)를 수정하여 오차를 줄여야 합니다. Step 3에서 우리는 '경사하강법(Gradient Descent)'을 배웠습니다. 산의 기울기(미분)를 따라 아래로 내려가는 방법이죠.

하지만 기본적인 확률적 경사하강법(SGD, Stochastic Gradient Descent)에는 치명적인 단점들이 있습니다.
- 지그재그로 왔다 갔다 하며 내려가서 시간이 너무 오래 걸립니다.
- '지역 최솟값(Local Minima)'이라는 작은 웅덩이에 빠지면 거기가 진짜 바닥(Global Minimum)인 줄 알고 멈춰버립니다.

이를 해결하기 위해 똑똑한 최적화 알고리즘(Optimizer)들이 등장했습니다.

#### 📍 옵티마이저의 진화 과정
1. **Momentum (관성)**: "내려오던 탄력을 유지하자!"
- 공을 언덕에서 굴리면 작은 웅덩이는 관성에 의해 튕겨서 빠져나올 수 있습니다. 이전 이동 방향을 기억하여 학습 속도를 높이고 웅덩이를 탈출하게 해줍니다.
2. **RMSprop**: "보폭을 유연하게 조절하자!"
- 경사가 가파른 곳(많이 움직인 곳)은 보폭을 줄여서 조심스럽게 내려가고, 경사가 완만한 곳은 보폭을 넓혀서 듬성듬성 빠르게 내려갑니다.
3. 🌟 **Adam (Adaptive Moment Estimation): 실무 점유율 1위** 🌟
- Momentum(관성) + RMSprop(보폭 조절)의 장점만 합친 완전체입니다. 방향도 유지하고 보폭도 상황에 맞게 조절합니다.
- 딥러닝 프로젝트를 시작할 때 "일단 Adam으로 시작해라"라는 격언이 있을 정도로 압도적인 표준입니다.

## 3. PyTorch에서 옵티마이저 변경해보기
PyTorch에서는 단 한 줄만 수정하면 세상의 거의 모든 최적화 알고리즘을 테스트해 볼 수 있습니다.
```python
import torch
import torch.nn as nn

# 가상의 모델 생성
model = nn.Linear(10, 2) 

# 1. 고전적인 방법: SGD
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01)

# 2. 관성을 더한 방법: SGD with Momentum
optimizer_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. 보폭을 조절하는 방법: RMSprop
optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.001)

# 4. 실무 절대 강자: Adam
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
```
학습률(`lr`, Learning Rate)은 딥러닝에서 가장 중요한 하이퍼파라미터(우리가 직접 튜닝해야 하는 값) 중 하나입니다. 주로 `1e-3` (0.001) 이나 `1e-4` (0.0001) 부터 테스트를 시작하는 것이 일반적입니다.

## 마치며
오늘 내용을 한 줄로 요약하면 다음과 같습니다.

**"정형 데이터 수치 예측은 MSELoss, 분류는 CrossEntropyLoss를 쓰며, 옵티마이저는 무조건 Adam(lr=0.001)으로 시작하라."**

딥러닝의 기초와 핵심 튜닝 요소(Phase 3)까지 모두 마스터하셨습니다!

이제 우리는 텍스트나 이미지 같은 '비정형 데이터'를 다룰 준비가 완벽하게 끝났습니다. 다음 Phase 4 - Step 9에서는 컴퓨터 비전(Computer Vision)의 핵심이자 인공지능이 눈을 뜨게 만든 혁명적인 기술, '합성곱 신경망(CNN)'을 구현해 보겠습니다!
