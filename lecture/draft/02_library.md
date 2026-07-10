$ Phase 1 - Step 2 $$ 

# AI 개발의 뼈대: NumPy, Pandas, 데이터 시각화 완벽 가이드
저번 시간에 완벽한 AI 개발 환경을 구축하셨나요? 그렇다면 이제 본격적으로 코드를 작성해 볼 차례입니다.

AI 모델은 '데이터'를 먹고 자랍니다. 아무리 뛰어난 딥러닝 모델도 정제되지 않은 쓰레기 데이터를 넣으면 쓰레기를 뱉어냅니다(Garbage In, Garbage Out). 따라서 데이터를 자유자재로 다루는 능력은 AI 개발자의 필수 소양입니다.

오늘 다룰 NumPy, Pandas, Matplotlib/Seaborn은 전 세계 모든 데이터 사이언티스트와 AI 엔지니어가 매일 숨 쉬듯이 사용하는 3대장 라이브러리입니다. 실무에서 가장 많이 쓰이는 핵심 기능 위주로 빠르게 알아보겠습니다.

## 1. NumPy: AI 연산의 심장 (수치 계산)
NumPy(넘파이)는 파이썬에서 대규모 다차원 배열과 행렬 연산을 처리하는 핵심 라이브러리입니다. 나중에 배우게 될 PyTorch나 TensorFlow의 '텐서(Tensor)'도 모두 이 NumPy 배열을 기반으로 디자인되었습니다.

Jupyter Notebook을 열고 아래 코드를 따라 쳐보세요!

```python
import numpy as np

# 1. 배열(Array) 생성하기
# 파이썬 기본 리스트와 비슷해 보이지만, 연산 속도가 압도적으로 빠릅니다.
data = [1, 2, 3, 4, 5]
arr = np.array(data)
print(f"NumPy 배열: {arr}")
print(f"배열의 형태(Shape): {arr.shape}") # 딥러닝에서 가장 많이 쓰게 될 속성입니다.

# 2. 행렬(Matrix) 연산 (선형대수 기초)
# 2x2 행렬 생성
matrix_A = np.array([[1, 2], [3, 4]])
matrix_B = np.array([[5, 6], [7, 8]])

print("\n--- 행렬 덧셈 ---")
print(matrix_A + matrix_B)

print("\n--- 행렬 곱셈 (Dot Product) ---")
# 딥러닝의 신경망은 본질적으로 수많은 행렬 곱셈의 연속입니다.
print(np.dot(matrix_A, matrix_B)) 
```

💡 **실무 포인트**: AI 모델을 만들 때 `ValueError: shapes (A,B) and (C,D) not aligned` 같은 에러를 정말 자주 만나게 됩니다. 이때 `np.shape`와 `np.reshape()`를 활용해 데이터의 차원을 맞추는 것이 일상입니다.

## 2. Pandas: 파이썬계의 엑셀 (데이터 전처리)
NumPy가 계산기라면, Pandas(판다스)는 엑셀(Excel)입니다. 표 형태의 데이터(표형 데이터, Tabular Data)를 불러오고, 탐색하고, 정제하는 데 특화되어 있습니다.
```python
import pandas as pd
import numpy as np # 결측치 생성을 위해 임포트

# 1. 데이터프레임(DataFrame) 만들기
# 실무에서는 보통 pd.read_csv("data.csv") 로 불러옵니다.
data = {
    'Name': ['AI_Bot_1', 'AI_Bot_2', 'AI_Bot_3', 'AI_Bot_4'],
    'Age': [1, 2, np.nan, 4], # np.nan은 '값이 없음(결측치)'을 의미합니다.
    'Accuracy': [0.85, 0.90, 0.78, 0.95]
}
df = pd.DataFrame(data)
print("--- 원본 데이터 ---")
display(df) # Jupyter에서는 print 대신 display를 쓰면 표가 예쁘게 출력됩니다.

# 2. 데이터 탐색하기
print("\n--- 데이터 기본 정보 요약 ---")
print(df.info())

# 3. 결측치(Missing Value) 처리하기 (매우 중요!)
# 실전 데이터는 구멍(결측치)이 뚫려있는 경우가 대부분입니다.
print("\n--- 결측치 확인 ---")
print(df.isna().sum())

print("\n--- 결측치 채우기 (나이의 평균값으로 채움) ---")
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
display(df)
```

💡 **실무 포인트**: 머신러닝 모델은 '빈칸(NaN)'을 이해하지 못합니다. Pandas의 `fillna()`로 적절한 값(평균, 중앙값 등)을 채우거나, `dropna()`로 아예 해당 데이터를 날려버리는 판단을 하는 것이 데이터 전처리의 핵심입니다.

## 3. Matplotlib & Seaborn: 데이터 시각화

데이터를 숫자로만 보면 패턴을 파악하기 어렵습니다. Matplotlib과 이를 기반으로 더 예쁘고 편하게 만들어진 Seaborn을 활용해 데이터를 시각화해 봅시다.
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화 스타일 설정
sns.set_theme(style="whitegrid")

# 임의의 데이터 생성 (정규분포를 따르는 데이터 1000개)
np.random.seed(42)
ages = np.random.normal(loc=35, scale=10, size=1000)
salaries = ages * 300 + np.random.normal(loc=0, scale=2000, size=1000)

# Pandas 데이터프레임으로 묶기
hr_data = pd.DataFrame({'Age': ages, 'Salary': salaries})

# 1. 데이터 분포 확인 (히스토그램)
plt.figure(figsize=(8, 4))
sns.histplot(hr_data['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.show()

# 2. 두 변수 간의 관계 확인 (산점도)
plt.figure(figsize=(8, 4))
sns.scatterplot(x='Age', y='Salary', data=hr_data, alpha=0.5)
plt.title('Age vs Salary')
plt.show()
```

데이터의 분포를 보고 이상치(Outlier)를 찾아내거나, 두 변수(예: 나이와 연봉) 사이에 어떤 상관관계가 있는지 눈으로 확인하는 과정은 AI 모델링 방향을 설정하는 데 큰 힌트를 줍니다.

## 마치며

오늘 우리는 AI의 연료인 '데이터'를 다루는 기초 체력을 길렀습니다. NumPy로 계산하고, Pandas로 정제하며, Seaborn으로 확인하는 이 흐름은 앞으로 어떤 거대한 AI 프로젝트를 하든 변하지 않는 기본기입니다.

다음 Step 3에서는 드디어 '머신러닝'의 세계로 진입합니다. 오늘 배운 라이브러리들을 활용해 **선형 회귀(Linear Regression)** 알고리즘의 원리를 알아보고 파이썬으로 직접 구현해 보겠습니다!
