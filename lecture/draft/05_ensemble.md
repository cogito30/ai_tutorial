$$Phase 2 - Step 5$$

# 실전 데이터 분석의 제왕: 랜덤 포레스트와 XGBoost (feat. 타이타닉)
지금까지 우리는 파이썬과 NumPy만을 이용해 선형 회귀와 로지스틱 회귀의 작동 원리를 밑바닥부터 뜯어보았습니다. 알고리즘의 원리를 이해하는 것은 매우 중요합니다.

하지만 실무에서 매번 수식과 미분을 직접 코딩하지는 않습니다. 수많은 천재 개발자들이 이미 최적화해 놓은 '라이브러리'를 가져다 쓰는 것이 실무의 방식입니다.

오늘은 엑셀과 같은 표 형태의 데이터(정형 데이터)를 다룰 때 딥러닝보다도 강력한 성능을 내는 **'트리 기반 앙상블(Ensemble)' 모델**을 실전처럼 다뤄보겠습니다.

## 1. 실습 준비: Scikit-learn과 XGBoost 설치
터미널을 열고 우리가 만든 `ai_env` 가상 환경에 머신러닝 필수 라이브러리인 `scikit-learn`과 `xgboost`를 설치합니다.
```
conda activate ai_env
conda install scikit-learn xgboost -c conda-forge
```
(Mac Apple Silicon 환경에서는 `conda-forge` 채널을 이용해 설치하는 것이 의존성 충돌을 막고 안정적입니다.)

## 2. 앙상블(Ensemble) 기법이란?
앙상블의 뜻은 '조화', '합창'입니다. 머신러닝에서 앙상블은 "약한 AI 여러 마리를 모아 집단 지성으로 강력한 AI를 만드는 기법"을 의미합니다. 그 중심에 '의사결정나무(Decision Tree)'가 있습니다.
- **의사결정나무**: 스무고개 게임과 같습니다. "나이가 30살 이상인가?", "남자인가?" 등의 질문을 통해 데이터를 분류합니다. 직관적이지만 훈련 데이터에만 너무 과적합(Overfitting)된다는 단점이 있습니다.
- **랜덤 포레스트 (Random Forest)**: 나무가 모이면 숲이 되죠! 의사결정나무를 수백 개 만들어 각자의 결과를 '다수결'로 투표하게 만듭니다. 가장 대중적이고 안정적인 성능을 냅니다.
- **XGBoost (eXtreme Gradient Boosting)**: 앞선 나무가 틀린 문제를 다음 나무가 집중적으로 고쳐나가는 '부스팅(Boosting)' 방식을 씁니다. Kaggle(데이터 사이언스 대회 플랫폼) 우승자들의 단골 무기입니다.

## 3. 실전 예제: 타이타닉 생존자 예측하기
데이터 사이언스의 'Hello World'로 불리는 타이타닉 생존자 예측을 직접 해보겠습니다. 승객의 나이, 성별, 티켓 등급 등의 데이터를 보고 이 승객이 살았을지 죽었을지 AI가 예측하는 프로젝트입니다.

Jupyter Notebook을 열고 코드를 작성해 봅시다!

#### 단계 1: 데이터 로드 및 전처리

```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Seaborn에 내장된 타이타닉 데이터 불러오기
titanic = sns.load_dataset('titanic')

# 2. 실무의 핵심, 데이터 전처리 (결측치 처리 및 숫자로 변환)
# 예측에 사용할 핵심 컬럼만 추출
df = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].copy()

# 나이(age)의 빈칸(NaN)을 전체 나이 평균으로 채우기
df['age'] = df['age'].fillna(df['age'].mean())

# 성별(sex)을 컴퓨터가 이해할 수 있게 숫자(남:0, 여:1)로 변환 (One-hot encoding 기초)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# 3. 문제(X)와 정답(y) 분리
X = df.drop('survived', axis=1) # 생존 여부를 제외한 나머지 속성들
y = df['survived']              # 정답 (0: 사망, 1: 생존)

# 4. 학습용 데이터와 테스트용 데이터 분리 (8:2 비율)
# AI가 본 적 없는 데이터로 평가하기 위해 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"학습용 데이터: {X_train.shape}, 테스트용 데이터: {X_test.shape}")
```

#### 단계 2: 랜덤 포레스트와 XGBoost 모델 훈련
`scikit-learn`을 사용하면 이토록 강력한 알고리즘을 단 세 줄이면 학습시킬 수 있습니다.

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- 1. 랜덤 포레스트 학습 ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # 100개의 나무 생성
rf_model.fit(X_train, y_train) # 학습해라!
rf_pred = rf_model.predict(X_test) # 테스트 데이터를 예측해라!
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"🌲 랜덤 포레스트 정확도: {rf_accuracy * 100:.2f}%")

# --- 2. XGBoost 학습 ---
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train) # 학습해라!
xgb_pred = xgb_model.predict(X_test) # 예측해라!
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"🚀 XGBoost 정확도: {xgb_accuracy * 100:.2f}%")
```

데이터 전처리만 잘 해두면, `fit()`과 `predict()` 단 두 개의 함수로 AI 모델링이 끝납니다. 참 쉽죠?

## 4. 모델 해석: AI는 무엇을 보고 생존을 판단했을까? (Feature Importance)
실무에서는 "예측을 잘하는 것"만큼 "왜 그렇게 예측했는지 설명하는 것"도 중요합니다. 트리 기반 모델들은 어떤 데이터(특성)가 예측에 가장 큰 영향을 미쳤는지 알려주는 **특성 중요도(Feature Importance)** 기능을 제공합니다.

```python
import matplotlib.pyplot as plt

# XGBoost 모델의 특성 중요도 추출
importance = xgb_model.feature_importances_
feature_names = X.columns

# 시각화
plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=feature_names, hue=feature_names, palette='viridis', legend=False)
plt.title('XGBoost Feature Importance (What determined survival?)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
```
그래프를 확인해보면 'sex(성별)'과 'fare(티켓 요금 - 부의 척도)', 'pclass(객실 등급)'가 생존에 가장 압도적인 영향을 미친 것을 볼 수 있습니다. 영화 '타이타닉'에서 여성과 어린이, 그리고 1등석 승객을 먼저 구명보트에 태웠던 역사적 사실을 AI가 데이터만 보고 정확하게 짚어낸 것입니다!

## 마치며 (Phase 2 완료)

여기까지 오신 것을 진심으로 축하합니다! 🎉 여러분은 이제 실무에서 가장 많이 쓰이는 머신러닝 라이브러리와 앙상블 기법까지 마스터했습니다. 표 형태의 데이터 분석에서는 이미 중급 이상의 지식을 갖추셨습니다.

지금까지가 머신러닝(Machine Learning)의 세계였다면, 다음 Phase 3 (Step 6) 부터는 본격적인 딥러닝(Deep Learning)의 세계로 진입합니다. 실무 점유율 1위 프레임워크인 PyTorch를 이용해 신경망을 직접 구축해 보는 흥미진진한 여정이 기다리고 있습니다!
