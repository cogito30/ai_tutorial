$$Phase 2 - Step 4$$

# 0과 1을 나누는 마법: 로지스틱 회귀와 분류 알고리즘 기초
지난 시간에 우리는 선형 회귀(Linear Regression)를 통해 '연속적인 숫자(예: 시험 점수)'를 예측하는 방법을 배웠습니다. 하지만 현실 세계의 많은 문제는 숫자를 맞추는 것이 아니라 '선택'하는 문제입니다.

- 이 이메일은 스팸인가? (스팸 O / 스팸 X)
- 이 환자는 암인가? (양성 / 음성)
- 이 고객이 대출을 갚을 것인가? (상환 / 미상환)

이렇게 데이터를 A와 B, 혹은 0과 1로 나누는 작업을 '분류(Classification)'라고 합니다. 오늘은 분류 알고리즘의 기초이자 딥러닝 신경망의 핵심 구성 요소인 '로지스틱 회귀(Logistic Regression)'를 직접 코딩하며 이해해 보겠습니다.

## 1. 이름은 '회귀'인데 왜 '분류' 알고리즘일까?
직선($y = wx + b$)을 긋는 선형 회귀를 분류 문제에 그대로 쓰면 어떻게 될까요? 선형 회귀의 결과값은 마이너스 무한대에서 플러스 무한대까지 끝없이 뻗어 나갑니다. 확률은 0%에서 100%(0과 1 사이)여야 하는데 말이죠.

그래서 우리는 직선을 구부려서 S자 형태의 곡선으로 만들어주는 마법의 함수를 도입합니다. 이것이 바로 딥러닝에서도 필수적으로 쓰이는 시그모이드 함수(Sigmoid Function)입니다.

$S(x) = \frac{1}{1 + e^{-x}}$

어떤 숫자를 넣어도 시그모이드 함수를 통과하면 그 결과값이 항상 0과 1 사이로 압축됩니다. 즉, "정답이 1(예: 합격)일 확률"로 해석할 수 있게 됩니다.

## 2. 파이썬(NumPy)으로 로지스틱 회귀 구현하기
Jupyter Notebook을 열고, 지난 시간처럼 오직 NumPy만을 사용해 경사하강법으로 로지스틱 회귀를 학습시켜 봅시다.
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 준비: 공부 시간(X)에 따른 시험 합격 여부(y)
# 0: 불합격, 1: 합격
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1]) 

# 2. 파라미터 초기화
w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 2000

# 3. 시그모이드 함수 정의 (핵심!)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. 훈련 루프 (Gradient Descent)
for i in range(epochs):
    # ① 직선의 방정식 통과
    z = w * X + b
    
    # ② 시그모이드 함수를 통과시켜 0~1 사이의 확률로 변환 (Forward Pass)
    y_pred = sigmoid(z)
    
    # ③ 오차 계산
    error = y_pred - y
    
    # ④ 기울기 계산 (Cross Entropy Loss의 미분 결과)
    w_grad = (1/len(X)) * np.sum(error * X)
    b_grad = (1/len(X)) * np.sum(error)
    
    # ⑤ 파라미터 업데이트
    w -= learning_rate * w_grad
    b -= learning_rate * b_grad
    
    if i % 400 == 0:
        print(f"Epoch {i}: w = {w:.4f}, b = {b:.4f}")

print(f"\n최종 파라미터: w = {w:.4f}, b = {b:.4f}")
```

#### 시각화로 S자 곡선 확인하기
우리의 AI 모델이 데이터를 어떻게 나누고 있는지 그래프로 확인해 보겠습니다.

```python
plt.figure(figsize=(8, 5))

# 원본 데이터 스캐터 플롯
plt.scatter(X, y, color='blue', label='Real Data (0=Fail, 1=Pass)')

# AI가 만들어낸 S자 곡선 (시그모이드)
X_test = np.linspace(0, 10, 100) # 0부터 10까지 촘촘한 테스트 데이터
y_test_pred = sigmoid(w * X_test + b)

plt.plot(X_test, y_test_pred, color='red', label='Logistic Regression Curve')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary (50%)')

plt.title('Logistic Regression')
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()

# 4.5시간 공부했을 때의 합격 확률 예측
prob = sigmoid(w * 4.5 + b)
print(f"🧐 4.5시간 공부 시 합격 확률: {prob*100:.1f}%")
print("결과:", "합격!" if prob >= 0.5 else "불합격 ㅠㅠ")
```

그래프를 보면 모델이 확률 50%(0.5)를 기준으로 합격과 불합격을 유연하게 가르고 있는 것을 볼 수 있습니다.

## 3. 실무 필수 지식: 분류 평가지표 (Accuracy, Precision, Recall)
모델을 만들었다면 '이 모델이 얼마나 똑똑한가?'를 평가해야 합니다. 단순한 "정확도"만으로는 실무에서 큰 낭패를 볼 수 있습니다.
- **정확도 (Accuracy)**: 전체 예측 중 맞춘 비율. (가장 직관적이지만, 데이터가 불균형할 때 위험함)
- **정밀도 (Precision)**: AI가 "1이다(합격이다/암이다)"라고 예측한 것 중 실제로 1인 비율. (오진으로 인한 비용이 클 때 중요)
- **재현율 (Recall)**: 실제 1인 정답 중에서 AI가 1이라고 잘 찾아낸 비율. (실제 암 환자를 놓치면 안 될 때 가장 중요)

파이썬으로 이 지표들을 직접 계산해 봅시다.
```python
# 임의의 테스트 결과라고 가정해 봅시다.
y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0]) # 실제 정답
y_pred_probs = np.array([0.1, 0.4, 0.8, 0.6, 0.9, 0.7, 0.3, 0.2]) # AI가 예측한 확률

# 확률이 0.5 이상이면 1(합격), 아니면 0(불합격)으로 판단
y_pred_class = (y_pred_probs >= 0.5).astype(int)

# True Positive, False Positive 등 계산
TP = np.sum((y_true == 1) & (y_pred_class == 1)) # 실제 1, 예측 1
TN = np.sum((y_true == 0) & (y_pred_class == 0)) # 실제 0, 예측 0
FP = np.sum((y_true == 0) & (y_pred_class == 1)) # 실제 0인데, 1이라고 우김
FN = np.sum((y_true == 1) & (y_pred_class == 0)) # 실제 1인데, 0이라고 놓침

accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print(f"정확도 (Accuracy) : {accuracy*100:.1f}%")
print(f"정밀도 (Precision): {precision*100:.1f}%")
print(f"재현율 (Recall)   : {recall*100:.1f}%")
```

## 마치며

오늘 배운 분류 평가지표(Accuracy, Precision, Recall)는 데이터 분석가와 AI 엔지니어 면접에서 **단골로 등장하는 핵심 질문**입니다. 각 지표가 어떤 상황에서 더 중요한지(예: 스팸 필터는 Precision, 암 진단은 Recall) 꼭 기억해 두세요.

여기까지 수식과 코드로 머신러닝의 '기본기'를 다졌습니다. 다음 Step 5에서는 실무 캐글(Kaggle) 대회 같은 곳에서 무조건 쓰이는 강력한 알고리즘인 트리 기반 앙상블(Random Forest & XGBoost)을 Scikit-learn을 활용해 실전처럼 사용해 보겠습니다!
