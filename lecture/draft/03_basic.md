$$Phase 2 - Step 3$$

# 머신러닝의 시작: 선형 회귀와 경사하강법 밑바닥부터 구현하기
데이터를 다루는 기초 체력을 길렀으니, 이제 드디어 **'기계(Machine)'에게 '학습(Learning)'**을 시켜볼 차례입니다.

머신러닝의 수많은 알고리즘 중 가장 기본이 되면서도 딥러닝까지 관통하는 핵심 원리를 담고 있는 것이 바로 '선형 회귀(Linear Regression)'와 '경사하강법(Gradient Descent)'입니다. 오늘은 복잡한 라이브러리 없이 NumPy만 사용하여 이 원리를 직접 코딩하며 완벽하게 이해해 보겠습니다.

## 1. 선형 회귀 (Linear Regression)란?
쉽게 말해 "데이터들을 가장 잘 설명하는 하나의 직선을 긋는 작업"입니다.
우리가 중학교 수학 시간에 배웠던 1차 함수 식을 떠올려 볼까요?

$y = wx + b$

- $x$: 입력 데이터 (예: 공부 시간)
- $y$: 예측하려는 정답 (예: 시험 점수)
- $w$ (Weight, 가중치): 직선의 기울기. $x$가 $y$에 미치는 영향력.
- $b$ (Bias, 편향): 직선의 y절편. 기본적으로 깔고 가는 값.

우리의 목표는 **주어진 데이터($x, y$)에 가장 잘 맞는 최적의 $w$와 $b$를 찾는 것**입니다.

## 2. 경사하강법 (Gradient Descent) - 산 내려가기
그렇다면 최적의 $w$와 $b$는 어떻게 찾을까요? 무작정 숫자를 대입해 볼 수는 없습니다. 이때 사용하는 방법이 **경사하강법**입니다.

눈을 가린 채 산 정상에서 가장 깊은 골짜기(최저점)로 내려가야 한다고 상상해 보세요. 어떻게 할까요? 발로 바닥의 경사(기울기)를 더듬어보고, **경사가 아래로 향하는 쪽으로 한 걸음씩 내딛을 것**입니다.

머신러닝도 똑같습니다.
1. 현재 $w, b$로 예측값을 만듭니다.
2. 실제 정답과 비교해 오차(Error)를 구합니다.
3. 오차를 줄이는 방향(기울기의 반대 방향)으로 $w, b$를 조금씩 수정(Update)합니다.

이 과정을 수백, 수천 번 반복하는 것이 바로 '학습(Training)'입니다.

## 3. 파이썬(NumPy)으로 직접 구현하기
자, 이제 Jupyter Notebook을 열고 직접 코드를 작성해 봅시다! 이전 Step에서 배운 NumPy와 Matplotlib을 활용합니다.
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 가상 데이터 준비 (공부 시간과 시험 점수)
# x: 공부 시간 (1~5시간)
X = np.array([1, 2, 3, 4, 5])
# y: 실제 시험 점수 (대략 y = 20x 에 가까운 데이터)
y = np.array([18, 41, 58, 83, 105]) 

# 2. 파라미터(w, b) 초기화
w = 0.0  # 가중치 (초기값 0)
b = 0.0  # 편향 (초기값 0)

# 3. 하이퍼파라미터 설정 (우리가 직접 세팅하는 값)
learning_rate = 0.01  # 산을 내려가는 보폭 (너무 크면 산을 벗어나고, 너무 작으면 오래 걸림)
epochs = 1000         # 전체 데이터를 반복 학습할 횟수

# 4. 훈련 루프 (Gradient Descent)
for i in range(epochs):
    # ① 예측 (Forward Pass)
    y_pred = w * X + b
    
    # ② 오차 계산 (실제값과 예측값의 차이)
    error = y_pred - y
    
    # ③ 기울기(Gradient) 계산 (오차 함수의 미분값)
    # 수학적으로 평균 제곱 오차(MSE)를 미분한 결과입니다.
    w_grad = (2/len(X)) * np.sum(error * X)
    b_grad = (2/len(X)) * np.sum(error)
    
    # ④ 파라미터 업데이트 (경사의 반대 방향으로 이동)
    w = w - learning_rate * w_grad
    b = b - learning_rate * b_grad
    
    # 과정 출력 (200번마다)
    if i % 200 == 0:
        print(f"Epoch {i}: w = {w:.4f}, b = {b:.4f}")

print(f"\n최종 학습 결과: y = {w:.4f}x + {b:.4f}")
```

## 4. 학습 결과 시각화하기
인공지능이 데이터를 보고 스스로 훌륭한 직선을 그어냈는지 눈으로 확인해 보겠습니다.
```python
# 5. 시각화 (Matplotlib)
plt.figure(figsize=(8, 5))

# 실제 데이터 흩뿌리기 (산점도)
plt.scatter(X, y, color='blue', label='Real Data')

# AI가 학습한 예측 직선 그리기
plt.plot(X, w * X + b, color='red', label='AI Prediction Line')

plt.title('Linear Regression with Gradient Descent')
plt.xlabel('Study Hours (x)')
plt.ylabel('Test Score (y)')
plt.legend()
plt.grid(True)
plt.show()

# 예측해보기
test_hours = 6
predicted_score = w * test_hours + b
print(f"😎 6시간 공부했을 때 예상 점수: {predicted_score:.1f}점")
```

그래프를 보면 붉은 선(AI가 학습한 선)이 파란 점(실제 데이터)들의 중앙을 멋지게 관통하는 것을 볼 수 있습니다. 여러분은 방금 가장 기초적인 인공지능을 바닥부터 직접 만들어 낸 것입니다!

## 마치며

오늘 배운 **"예측 ➔ 오차 계산 ➔ 기울기 계산 ➔ 파라미터 업데이트"** 라는 반복 루프는 놀랍게도 가장 최신 기술인 챗GPT(LLM)나 이미지 생성 AI 모델을 학습시킬 때도 똑같이 적용되는 대원칙입니다.

연속적인 숫자를 예측하는 '회귀(Regression)'를 마스터했으니, 다음 Step 4에서는 데이터를 A와 B로 나누는 '분류(Classification)'의 기초, 로지스틱 회귀(Logistic Regression)에 대해 알아보겠습니다!
